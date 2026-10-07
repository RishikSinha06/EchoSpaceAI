"""Deterministic frozen-sample B0/A/P runs with validation-selected checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from echospace.models import SpatialConfig, SpatialUNet, WallConfig, WallExtrapolation
from .frozen import FrozenDataset
from .losses import masked_bce_dice, masked_loss_per_sample


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 10
    min_delta: float = 0.0001
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    amp: bool = True
    device: str = "cpu"
    threads: int = 2

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.patience, self.threads) < 1 or self.seed < 0:
            raise ValueError("epochs, batch size, patience and threads must be positive; seed nonnegative")
        if (self.learning_rate <= 0 or self.weight_decay < 0 or self.min_delta < 0
                or self.bce_weight < 0 or self.dice_weight < 0 or self.bce_weight + self.dice_weight <= 0):
            raise ValueError("invalid optimizer, early stopping or loss settings")
        if not all(np.isfinite(v) for v in (self.learning_rate, self.weight_decay, self.min_delta,
                                          self.bce_weight, self.dice_weight)):
            raise ValueError("training settings must be finite")


def seed_everything(seed):
    # Set before CUDA initialization. Unsupported nondeterministic operations
    # raise rather than quietly relaxing the reproducibility requirement.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_model(variant, spatial=SpatialConfig(), wall=WallConfig()):
    if variant == "B0":
        return WallExtrapolation(wall)
    return SpatialUNet(variant, spatial)


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = handle.name
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    with tempfile.NamedTemporaryFile(dir=Path(path).parent, delete=False) as handle:
        temporary = handle.name
        torch.save(value, handle)
    os.replace(temporary, path)


def load_checkpoint(path, expected_frozen_checksum=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "p7.checkpoint.1":
        raise ValueError("unknown checkpoint format")
    if expected_frozen_checksum is not None and payload["frozen_checksum"] != expected_frozen_checksum:
        raise ValueError("checkpoint belongs to another frozen dataset")
    spec = payload["model"]
    model = make_model(spec["variant"], SpatialConfig(**spec["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def _device_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _metrics(output, batch):
    mask = (batch["observed_cells"] == 0) & (batch["valid_cells"] == 1)
    result = {}
    for head in ("occupancy", "boundary"):
        predicted = output[f"{head}_logits"].float().sigmoid() >= 0.5
        target = batch[f"target_{head}"].bool()
        intersection = ((predicted & target) & mask).sum((1, 2))
        union = ((predicted | target) & mask).sum((1, 2))
        result[f"{head}_iou"] = torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union, dtype=torch.float32))
    return result


def evaluate(model, dataset, config, device, output_dir=None, checkpoint_sha256=None):
    model.eval()
    sums, count, exported = {"loss": 0.0, "occupancy_iou": 0.0, "boundary_iou": 0.0}, 0, []
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            batch = _device_batch(batch, device)
            output = model(batch)
            losses, active = masked_loss_per_sample(output, batch, config.bce_weight, config.dice_weight)
            metrics = _metrics(output, batch)
            sums["loss"] += float(losses[active].sum())
            for key in metrics:
                sums[key] += float(metrics[key][active].sum())
            count += int(active.sum())
            if output_dir is not None:
                for i in range(len(active)):
                    entry = dataset.entries[offset + i]
                    prediction = {f"{head}_probability": output[f"{head}_logits"][i].float().sigmoid().cpu().numpy()
                                  for head in ("occupancy", "boundary")}
                    if any(a.shape != (64, 64) or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any()
                           for a in prediction.values()):
                        raise ValueError("invalid prediction probabilities")
                    name = f"{offset+i:06d}.npz"
                    np.savez_compressed(output_dir / name, **prediction)
                    exported.append({"sample_id": entry["metadata"]["sample_id"], "sample_sha256": entry["sha256"],
                                     "room_id": entry["metadata"]["room_id"], "room_group": entry["metadata"]["room_group"],
                                     "grid": entry["metadata"]["grid"], "spatial_transform": entry["metadata"]["spatial_transform"],
                                     "prediction_path": name, "checkpoint_sha256": checkpoint_sha256,
                                     "prediction_file_sha256": hashlib.sha256((output_dir / name).read_bytes()).hexdigest()})
            offset += len(active)
    if count == 0:
        raise ValueError("evaluation has no unobserved supervision")
    return {key: value / count for key, value in sums.items()} | {"samples": count}, exported


def run_baseline(variant, frozen_manifest, output_dir, config=TrainConfig(), spatial=SpatialConfig(), wall=WallConfig()):
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("run destination must be empty; never overwrite a recorded run")
    datasets = {s: FrozenDataset(frozen_manifest, s) for s in ("train", "val", "test")}
    checksum = datasets["train"].manifest["checksum"]
    root.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model = make_model(variant, spatial, wall).to(device)
    use_amp = bool(config.amp and device.type == "cuda" and variant != "B0")
    try:
        revision = subprocess.check_output(["git", "-c", f"safe.directory={Path(__file__).resolve().parents[3].as_posix()}",
                                            "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    manifest = {"format": "p7.run.1", "status": "running", "variant": variant,
                "model": model.specification(), "config": asdict(config), "amp_enabled": use_amp,
                "frozen_checksum": checksum, "source": datasets["train"].manifest["source"],
                "code_revision": revision, "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "cuda": torch.version.cuda},
                "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                "loss_domain": "valid_cells=1 AND observed_cells=0; per-sample BCE+Dice, mean of two heads",
                "split_sample_ids": {s: [e["metadata"]["sample_id"] for e in d.entries] for s, d in datasets.items()},
                "split_room_groups": {s: sorted({str(e["metadata"]["room_group"]) for e in d.entries}) for s, d in datasets.items()},
                "parameter_count": sum(p.numel() for p in model.parameters()), "history": []}
    package = Path(__file__).resolve().parents[1]
    manifest["source_files_sha256"] = {
        file.relative_to(package).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for folder in (package / "models", package / "training") for file in sorted(folder.glob("*.py"))
    }
    manifest["determinism"] = {"algorithms": True, "cudnn_benchmark": False, "tf32": False,
                                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                                "scope": "repeatable on the same software/hardware; no cross-platform bit equality claim"}
    atomic_json(root / "run.json", manifest)
    try:
        checkpoint_sha = None
        if variant != "B0":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            best, bad, best_epoch = float("inf"), 0, 0
            for epoch in range(1, config.epochs + 1):
                generator = torch.Generator().manual_seed(config.seed + epoch)
                train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True,
                                          num_workers=0, generator=generator)
                model.train()
                train_sum, train_count = 0.0, 0
                for batch in train_loader:
                    batch = _device_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        output = model(batch)
                        loss = masked_bce_dice(output, batch, config.bce_weight, config.dice_weight)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_sum += float(loss.detach()) * len(batch["observed_cells"])
                    train_count += len(batch["observed_cells"])
                validation, _ = evaluate(model, datasets["val"], config, device)
                row = {"epoch": epoch, "train_loss": train_sum / train_count, "validation": validation}
                manifest["history"].append(row)
                improved = validation["loss"] < best - config.min_delta
                if improved:
                    best, best_epoch, bad = validation["loss"], epoch, 0
                else:
                    bad += 1
                payload = {"format": "p7.checkpoint.1", "epoch": epoch, "model": model.specification(),
                           "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                           "config": asdict(config), "frozen_checksum": checksum, "best_epoch": best_epoch,
                           "best_validation_loss": best, "bad_epochs": bad, "torch_rng_state": torch.get_rng_state(),
                           "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
                save_checkpoint(root / "last.pt", payload)
                if improved:
                    save_checkpoint(root / "best.pt", payload)
                atomic_json(root / "run.json", manifest)
                if bad >= config.patience:
                    break
            model, best_payload = load_checkpoint(root / "best.pt", checksum)
            model.to(device)
            checkpoint_sha = hashlib.sha256((root / "best.pt").read_bytes()).hexdigest()
            manifest.update(best_epoch=best_payload["epoch"], checkpoint_sha256=checkpoint_sha,
                            stop_reason="early_stopping" if bad >= config.patience else "epoch_limit")
        else:
            manifest["stop_reason"] = "parameter_free_baseline"
        # Test data are first read here, after all optimizer/selection decisions.
        for split in ("val", "test"):
            directory = root / f"predictions_{split}"
            directory.mkdir()
            metrics, predictions = evaluate(model, datasets[split], config, device, directory, checkpoint_sha)
            atomic_json(directory / "manifest.json", {"format": "p7.predictions.1", "variant": variant, "split": split,
                                                       "model": model.specification(), "run_manifest": "../run.json",
                                                       "frozen_checksum": checksum, "predictions": predictions})
            manifest[f"{split}_metrics"] = metrics
        manifest["status"] = "complete"
        atomic_json(root / "run.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(root / "run.json", manifest)
        raise
