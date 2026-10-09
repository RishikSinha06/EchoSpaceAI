"""P8 training and evaluation for EchoFusion variants (plan, Stages 2-3).

Reuses P7's masked BCE + Dice loss (valid unobserved cells only), seeding,
atomic writers and IoU metric. Adds what the plan specifies for fusion
training: AdamW (lr 3e-4, weight decay 1e-4), 5-epoch linear warm-up then
cosine decay, random K uniform in 1..8 per sample and epoch, early stopping
on inner-validation missing-region IoU (patience 15), mixed precision on
CUDA, and held-out evaluation at K = 1, 2, 4, 8 on the same fixed masks.

Validation alone selects the checkpoint; test samples are read only after
training has finished. A trivial "mean room shape" predictor (plan's
learning gate for S) is computed from training targets only.
"""

from __future__ import annotations

import hashlib
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from echospace.models.echofusion import EchoFusion, build

from .audio_frozen import FrozenAudioDataset
from .losses import masked_bce_dice, masked_loss_per_sample
from .runner import atomic_json, save_checkpoint, seed_everything

CHECKPOINT_FORMAT = "p8.checkpoint.1"
RUN_FORMAT = "p8.run.1"


@dataclass(frozen=True)
class FusionTrainConfig:
    seed: int = 0
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 15
    min_delta: float = 1e-4
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    k_range: tuple[int, int] = (1, 8)
    selection_k: int = 8
    eval_k: tuple[int, ...] = (1, 2, 4, 8)
    amp: bool = True
    device: str = "cpu"
    threads: int = 2
    workers: int = 0
    max_train_batches: int | None = None  # smoke runs only; recorded in the run manifest

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.patience, self.threads) < 1 or self.seed < 0 or self.workers < 0:
            raise ValueError("epochs, batch size, patience, threads must be positive; seed and workers nonnegative")
        if not (0 < self.learning_rate and self.weight_decay >= 0 and self.warmup_epochs >= 0):
            raise ValueError("invalid optimiser or warm-up settings")
        if not 1 <= self.k_range[0] <= self.k_range[1] <= 8 or not all(1 <= k <= 8 for k in (*self.eval_k, self.selection_k)):
            raise ValueError("K values must lie in 1..8")


def schedule(config: FusionTrainConfig, steps_per_epoch: int) -> Callable[[int], float]:
    """Linear warm-up over ``warmup_epochs``, then cosine decay to zero at ``epochs``."""
    warmup, total = config.warmup_epochs * steps_per_epoch, config.epochs * steps_per_epoch

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return factor


def _device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def masked_iou(probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample IoU at 0.5 over masked cells; empty union counts as 1 (as in P7)."""
    predicted = probability >= 0.5
    target = target.bool()
    intersection = (predicted & target & mask).sum((1, 2)).float()
    union = ((predicted | target) & mask).sum((1, 2)).float()
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))


def evaluate(model: torch.nn.Module, dataset: FrozenAudioDataset, config: FusionTrainConfig, device: torch.device,
             keep_rows: bool = False) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    sums = {"loss": 0.0, "occupancy_iou": 0.0, "boundary_iou": 0.0}
    rows: list[dict[str, Any]] = []
    count, offset = 0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = _device(batch, device)
            output = model(batch)
            losses, active = masked_loss_per_sample(output, batch, config.bce_weight, config.dice_weight)
            mask = (batch["observed_cells"] == 0) & (batch["valid_cells"] == 1)
            ious = {head: masked_iou(output[f"{head}_logits"].float().sigmoid(), batch[f"target_{head}"], mask)
                    for head in ("occupancy", "boundary")}
            sums["loss"] += float(losses[active].sum())
            for head, value in ious.items():
                sums[f"{head}_iou"] += float(value[active].sum())
            count += int(active.sum())
            if keep_rows:
                for i in range(len(active)):
                    meta = dataset.entries[offset + i]["metadata"]
                    rows.append({"sample_id": meta["sample_id"], "room_id": meta["room_id"], "room_group": meta["room_group"],
                                 "occlusion_type": meta["occlusion_type"], "coverage_bin": meta["coverage_bin"],
                                 "k": int(batch["k"][i]), "loss": float(losses[i]),
                                 "occupancy_iou": float(ious["occupancy"][i]), "boundary_iou": float(ious["boundary"][i])})
            offset += len(active)
    if count == 0:
        raise ValueError("evaluation has no unobserved supervision")
    return {key: value / count for key, value in sums.items()} | {"samples": count}, rows


def mean_shape_baseline(train: FrozenAudioDataset, test: FrozenAudioDataset) -> dict[str, float]:
    """Trivial predictor: per-cell mean training target, thresholded at 0.5, scored like the models."""
    totals = {head: torch.zeros(64, 64, dtype=torch.float64) for head in ("occupancy", "boundary")}
    for epoch in train.epochs:
        train.set_epoch(epoch)
        for i in range(len(train)):
            item = train[i]
            for head in totals:
                totals[head] += item[f"target_{head}"].double()
    n = sum(1 for _ in train.epochs) * len(train)
    means = {head: (value / n).float() for head, value in totals.items()}
    train.set_epoch(0)
    sums = {"occupancy_iou": 0.0, "boundary_iou": 0.0}
    for i in range(len(test)):
        item = test[i]
        mask = ((item["observed_cells"] == 0) & (item["valid_cells"] == 1))[None]
        for head in ("occupancy", "boundary"):
            sums[f"{head}_iou"] += float(masked_iou(means[head][None], item[f"target_{head}"][None], mask)[0])
    return {key: value / len(test) for key, value in sums.items()} | {"samples": len(test)}


def load_fusion_checkpoint(path: str | Path, expected_frozen_checksum: str | None = None) -> tuple[EchoFusion, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unknown checkpoint format")
    if expected_frozen_checksum is not None and payload["frozen_checksum"] != expected_frozen_checksum:
        raise ValueError("checkpoint belongs to another frozen dataset")
    model = build(payload["model"]["variant"], payload["model"]["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {file.relative_to(package).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
            for folder in (package / "models", package / "training") for file in sorted(folder.glob("*.py"))}


def run_fusion(variant: str, frozen_manifest: str | Path, output_dir: str | Path,
               config: FusionTrainConfig = FusionTrainConfig(), model_config: dict | None = None,
               progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("run destination must be empty; never overwrite a recorded run")
    seed_everything(config.seed)
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    train = FrozenAudioDataset(frozen_manifest, "train", random_k=config.k_range, seed=config.seed)
    val = FrozenAudioDataset(frozen_manifest, "val", k=config.selection_k)
    checksum = train.checksum
    root.mkdir(parents=True, exist_ok=True)
    model = build(variant, model_config).to(device)
    use_amp = bool(config.amp and device.type == "cuda")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
                                           stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    manifest: dict[str, Any] = {
        "format": RUN_FORMAT, "status": "running", "variant": variant, "model": model.specification(),
        "parameter_count": sum(p.numel() for p in model.parameters()), "config": asdict(config),
        "amp_enabled": use_amp, "frozen_checksum": checksum, "source": train.manifest["source"],
        "code_revision": revision, "source_files_sha256": _source_hashes(),
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                     "cuda": torch.version.cuda},
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "loss_domain": "valid_cells=1 AND observed_cells=0; per-sample BCE+Dice, mean of two heads (P7 loss)",
        "selection": f"inner-validation occupancy IoU on unobserved cells at K={config.selection_k}",
        "smoke_run": config.max_train_batches is not None, "history": [],
    }
    atomic_json(root / "run.json", manifest)
    try:
        steps = len(train) // config.batch_size + (len(train) % config.batch_size > 0)
        if config.max_train_batches is not None:
            steps = min(steps, config.max_train_batches)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(config, steps))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best, best_epoch, bad = -math.inf, 0, 0
        for epoch in range(1, config.epochs + 1):
            train.set_epoch(epoch - 1)
            generator = torch.Generator().manual_seed(config.seed * 1000 + epoch)
            loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator,
                                num_workers=config.workers, persistent_workers=False)
            model.train()
            total, seen = 0.0, 0
            for step, batch in enumerate(loader):
                if config.max_train_batches is not None and step >= config.max_train_batches:
                    break
                batch = _device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    output = model(batch)
                    loss = masked_bce_dice(output, batch, config.bce_weight, config.dice_weight)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total += float(loss.detach()) * len(batch["k"])
                seen += len(batch["k"])
            validation, _ = evaluate(model, val, config, device)
            score = validation["occupancy_iou"]
            improved = score > best + config.min_delta
            if improved:
                best, best_epoch, bad = score, epoch, 0
            else:
                bad += 1
            row = {"epoch": epoch, "train_loss": total / max(1, seen), "learning_rate": scheduler.get_last_lr()[0],
                   "validation": validation, "improved": improved}
            manifest["history"].append(row)
            payload = {"format": CHECKPOINT_FORMAT, "epoch": epoch, "model": model.specification(),
                       "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "config": asdict(config),
                       "frozen_checksum": checksum, "best_epoch": best_epoch, "best_validation_iou": best,
                       "bad_epochs": bad, "torch_rng_state": torch.get_rng_state()}
            save_checkpoint(root / "last.pt", payload)
            if improved:
                save_checkpoint(root / "best.pt", payload)
            atomic_json(root / "run.json", manifest)
            if progress is not None:
                progress(row)
            if bad >= config.patience:
                break
        model, chosen = load_fusion_checkpoint(root / "best.pt", checksum)
        model.to(device)
        manifest.update(best_epoch=chosen["epoch"], best_validation_iou=chosen["best_validation_iou"],
                        checkpoint_sha256=hashlib.sha256((root / "best.pt").read_bytes()).hexdigest(),
                        stop_reason="early_stopping" if bad >= config.patience else "epoch_limit")
        # Held-out data are first read here, after every selection decision.
        results: dict[str, Any] = {}
        for k in config.eval_k:
            test = FrozenAudioDataset(frozen_manifest, "test", k=k)
            metrics, rows = evaluate(model, test, config, device, keep_rows=True)
            results[f"K={k}"] = metrics
            atomic_json(root / f"test_rows_K{k}.json", {"variant": variant, "k": k, "frozen_checksum": checksum,
                                                         "checkpoint_sha256": manifest["checkpoint_sha256"], "rows": rows})
        manifest["test_metrics"] = results
        manifest["status"] = "complete"
        atomic_json(root / "run.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(root / "run.json", manifest)
        raise
