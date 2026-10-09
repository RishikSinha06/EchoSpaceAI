"""P8 frozen samples with RIR waveforms, and random-K training views.

P7's ``p7.frozen.1`` stores no audio. EchoFusion needs the P5 waveforms, so P8
materialises its own immutable set, format ``p8.frozen.1``: P7's arrays plus
``rir`` (8 x 1280, float16, the plan's cache precision), with occlusion type
and coverage bin in the metadata for per-bin reporting.

Two training sources, recorded in the manifest:

- ``online``: E epochs of P6 online draws (fresh scan, +-1 m shift, P5 and
  spatial augmentation) for training rooms, materialised once so a GPU run
  does not wait on scan simulation. Epoch e of training reads draw e mod E.
- ``fixed-masks``: the committed P6 fixed masks of the training rooms, no
  augmentation (for machines that only have the evaluation archive).

Validation and test always use the committed P6 fixed masks, verified against
``eval_masks.json``. Every frozen sample stores up to 8 nested pairs; random K
(plan: uniform 1..8 per sample) is applied at load time by zeroing rows >= K,
which keeps P5's nested prefix: K = 4 is exactly the first four of K = 8.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .frozen import json_sha, validate_item

FORMAT = "p8.frozen.1"
ARRAY_KEYS = ("observed_cells", "observed_free", "observed_wall", "valid_cells", "target_occupancy",
              "target_boundary", "src_pos", "mic_pos", "rir_valid", "rir")
META_KEYS = ("sample_id", "room_id", "room_group", "grid", "spatial_transform", "occlusion_type", "coverage_bin",
             "coverage")


def sample_sha(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(json_sha(metadata).encode())
    for key in ARRAY_KEYS:
        a = np.ascontiguousarray(arrays[key])
        digest.update(f"{key}|{a.dtype}|{json.dumps(a.shape)}|".encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _arrays(item: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = {key: np.ascontiguousarray(item[key]) for key in ARRAY_KEYS}
    for key in ARRAY_KEYS[:6]:
        arrays[key] = arrays[key].astype(np.uint8)
    arrays["rir_valid"] = arrays["rir_valid"].astype(np.uint8)
    arrays["src_pos"] = arrays["src_pos"].astype(np.float32)
    arrays["mic_pos"] = arrays["mic_pos"].astype(np.float32)
    arrays["rir"] = arrays["rir"].astype(np.float16)
    return arrays


def validate_audio_item(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    validate_item({k: arrays[k] for k in ARRAY_KEYS[:-1]}, metadata)  # P7's mask, pose and grid checks
    rir, valid = arrays["rir"], arrays["rir_valid"].astype(bool)
    if rir.shape != (8, 1280) or not np.isfinite(rir).all():
        raise ValueError("rir must be finite 8 x 1280")
    if np.any(rir[~valid] != 0):
        raise ValueError("invalid RIR rows must be zero")
    if not valid.any() or not np.all(valid[: int(valid.sum())]):
        raise ValueError("valid rows must be a non-empty prefix (P5 nested ranking)")


def write_freeze(directory: str | Path, splits: dict[str, Iterable[dict[str, Any]]], source: dict[str, Any]) -> Path:
    """Materialise split iterables of P6-style items. Refuses non-empty destinations."""
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise ValueError("freeze destination must be empty; create a new version")
    (root / "samples").mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    owner: dict[str, str] = {}
    ids: set[str] = set()
    for split in ("train", "val", "test"):
        count = 0
        for item in splits[split]:
            metadata = {key: item[key] for key in META_KEYS}
            metadata["room_group"] = int(metadata["room_group"])
            epoch = int(item.get("epoch", 0))
            for key in (metadata["room_id"], f"group:{metadata['room_group']}"):
                if owner.setdefault(key, split) != split:
                    raise ValueError(f"{key} crosses frozen splits")
            unique = f"{metadata['sample_id']}#{epoch}"
            if unique in ids:
                raise ValueError(f"duplicate frozen sample {unique}")
            ids.add(unique)
            arrays = _arrays(item)
            validate_audio_item(arrays, metadata)
            name = f"samples/{split}_{count:06d}.npz"
            np.savez_compressed(root / name, **arrays)
            entries.append({"split": split, "epoch": epoch, "path": name, "metadata": metadata,
                            "sha256": sample_sha(metadata, arrays)})
            count += 1
        if count == 0:
            raise ValueError(f"frozen {split} split is empty")
    manifest = {"format": FORMAT, "source": source, "samples": entries,
                "train_epochs": len({e["epoch"] for e in entries if e["split"] == "train"})}
    manifest["checksum"] = json_sha(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return root / "manifest.json"


# --- sources ---------------------------------------------------------------------

def fixed_items(eval_dir: str | Path, folds: dict[str, Any], spec: dict[str, Any], fold: int, split: str) -> Iterator[dict[str, Any]]:
    """Verified P6 fixed masks of one fold part (train, val or test), all 8 nested pairs, no augmentation."""
    from echospace.data.dataset import EchoSpaceDataset, grid_from_record, to_grid_metres
    from echospace.data.evalmasks import read_eval_sample, read_manifest

    manifest = read_manifest(eval_dir)
    if set(manifest) != {e["sample_id"] for e in spec["samples"]}:
        raise ValueError("evaluation manifest IDs differ from the committed spec")
    wanted = set(folds["folds"][fold][f"{split}_rooms"])
    for entry in spec["samples"]:
        if entry["room_id"] not in wanted:
            continue
        record = manifest[entry["sample_id"]]
        arrays = read_eval_sample(eval_dir, record, entry["sha256"]).arrays  # checksum + contract
        grid = grid_from_record(record["grid"])
        valid = arrays["rir_valid"].astype(np.uint8)
        uvh = to_grid_metres(grid, arrays["rir_positions_scene_m"].astype(np.float64).reshape(-1, 3)).reshape(-1, 2, 3)
        uvh[valid == 0] = 0
        yield EchoSpaceDataset._item(arrays, arrays["rir_waveforms"].astype(np.float32), valid, uvh, int(valid.sum()),
                                     record["coverage"], record["coverage_bin"], record["occlusion_type"],
                                     record["room_id"], record["room_group"], record["sample_id"], record["grid"], 0, False)


def _online_worker(args: tuple[str, dict[str, Any], dict[str, Any], int, int, int, list[int]]) -> list[dict[str, Any]]:
    room_cache_dir, folds, spec, fold, seed, epoch, indices = args
    from echospace.data import EchoSpaceDataset

    dataset = EchoSpaceDataset(room_cache_dir, folds, fold, "train", eval_spec=spec, seed=seed,
                               samples_per_room=folds["_samples_per_room"], k_range=(8, 8))
    dataset.set_epoch(epoch)
    items = []
    for index in indices:
        item = dataset[index]
        item["epoch"] = epoch
        items.append(item)
    return items


def online_items(room_cache_dir: str | Path, folds: dict[str, Any], spec: dict[str, Any], fold: int, seed: int,
                 epochs: int, samples_per_room: int, workers: int = 1, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """E epochs of P6 training draws (K up to 8, augmented), in a fixed order."""
    from echospace.data import EchoSpaceDataset

    probe = EchoSpaceDataset(room_cache_dir, folds, fold, "train", eval_spec=spec, seed=seed,
                             samples_per_room=samples_per_room, k_range=(8, 8))
    total = len(probe) if limit is None else min(limit, len(probe))
    shared = dict(folds, _samples_per_room=samples_per_room)
    chunks = [list(range(i, min(i + 64, total))) for i in range(0, total, 64)]
    jobs = [(str(room_cache_dir), shared, spec, fold, seed, epoch, chunk) for epoch in range(epochs) for chunk in chunks]
    if workers <= 1:
        for job in jobs:
            yield from _online_worker(job)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for items in pool.map(_online_worker, jobs):  # map keeps job order: deterministic output
            yield from items


def freeze_p8(destination: str | Path, *, eval_dir: str | Path, folds_path: str | Path, eval_spec_path: str | Path,
              fold: int = 0, training_source: str = "online", room_cache_dir: str | Path | None = None,
              seed: int = 0, epochs: int = 10, samples_per_room: int = 20, workers: int = 1,
              limit_train: int | None = None) -> Path:
    from dataclasses import asdict

    from echospace.acoustics import AugmentConfig, RirConfig
    from echospace.acoustics.rir import PROCESSOR_VERSION
    from echospace.data import load_eval_spec, load_folds

    folds, spec = load_folds(folds_path), load_eval_spec(eval_spec_path)
    if training_source == "online":
        if room_cache_dir is None or not Path(room_cache_dir).is_dir():
            raise FileNotFoundError("online training draws need the P6 room caches")
        train = online_items(room_cache_dir, folds, spec, fold, seed, epochs, samples_per_room, workers, limit_train)
    elif training_source == "fixed-masks":
        if limit_train is not None:
            raise ValueError("fixed-masks freezes the whole fold")
        train = fixed_items(eval_dir, folds, spec, fold, "train")
    else:
        raise ValueError("training_source must be online or fixed-masks")
    source = {"kind": f"P6_{training_source}", "fold": fold, "seed": seed,
              "train_epochs": epochs if training_source == "online" else 1,
              "samples_per_room": samples_per_room if training_source == "online" else None,
              "limit_train": limit_train,
              "folds_sha256": hashlib.sha256(Path(folds_path).read_bytes()).hexdigest(),
              "eval_spec_checksum": spec["checksum"], "p5_processor_version": PROCESSOR_VERSION,
              "rir_config": asdict(RirConfig()),
              "training_augmentation": asdict(AugmentConfig()) if training_source == "online" else "none",
              "val_test": "committed P6 fixed masks, 8 nested pairs, no augmentation"}
    splits = {"train": train,
              "val": fixed_items(eval_dir, folds, spec, fold, "val"),
              "test": fixed_items(eval_dir, folds, spec, fold, "test")}
    return write_freeze(destination, splits, source)


# --- dataset -------------------------------------------------------------------------

class FrozenAudioDataset(Dataset):
    """Tensors for one split. ``k``: fixed K (eval); ``random_k``: uniform K range per sample and epoch (train)."""

    def __init__(self, manifest_path: str | Path, split: str, *, k: int | None = None,
                 random_k: tuple[int, int] | None = None, seed: int = 0, verify: bool = True) -> None:
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        body = {key: value for key, value in self.manifest.items() if key != "checksum"}
        if self.manifest.get("format") != FORMAT or self.manifest.get("checksum") != json_sha(body):
            raise ValueError("p8 frozen manifest checksum/format mismatch")
        if (k is None) == (random_k is None):
            raise ValueError("give exactly one of k (fixed) or random_k (range)")
        if k is not None and not 1 <= k <= 8 or random_k is not None and not 1 <= random_k[0] <= random_k[1] <= 8:
            raise ValueError("K must be within 1..8")
        self.split, self.k, self.random_k, self.seed, self.verify, self.epoch = split, k, random_k, seed, verify, 0
        owner: dict[str, str] = {}
        for entry in self.manifest["samples"]:
            for key in (entry["metadata"]["room_id"], f"group:{entry['metadata']['room_group']}"):
                if owner.setdefault(key, entry["split"]) != entry["split"]:
                    raise ValueError(f"{key} crosses frozen splits")
        self.all_entries = [e for e in self.manifest["samples"] if e["split"] == split]
        if not self.all_entries:
            raise ValueError(f"empty frozen {split}")
        self.epochs = sorted({e["epoch"] for e in self.all_entries})
        self.set_epoch(0)

    @property
    def checksum(self) -> str:
        return self.manifest["checksum"]

    def set_epoch(self, epoch: int) -> None:
        """Training epoch e reads frozen draw e mod E (online freezes); fixed sets ignore it."""
        self.epoch = int(epoch)
        stored = self.epochs[self.epoch % len(self.epochs)]
        self.entries = [e for e in self.all_entries if e["epoch"] == stored]

    def __len__(self) -> int:
        return len(self.entries)

    def k_for(self, index: int, available: int) -> int:
        if self.k is not None:
            return min(self.k, available)
        key = f"p8-k|{self.seed}|{self.epoch}|{self.entries[index]['metadata']['sample_id']}"
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
        low, high = self.random_k
        return min(int(rng.integers(low, high + 1)), available)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.entries[index]
        path = (self.path.parent / entry["path"]).resolve()
        if not path.is_relative_to(self.path.parent.resolve()):
            raise ValueError("sample path escapes the frozen directory")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in ARRAY_KEYS}
        if self.verify and sample_sha(entry["metadata"], arrays) != entry["sha256"]:
            raise ValueError(f"frozen sample checksum mismatch: {entry['path']}")
        k = self.k_for(index, int(arrays["rir_valid"].sum()))
        for key in ("rir_valid", "rir", "src_pos", "mic_pos"):
            arrays[key][k:] = 0  # nested prefix: rows >= K never reach the model
        out = {key: torch.from_numpy(np.ascontiguousarray(arrays[key])) for key in ARRAY_KEYS}
        out["rir"] = out["rir"].float()
        out["k"] = torch.tensor(k)
        return out
