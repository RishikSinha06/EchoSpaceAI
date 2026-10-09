"""P7 immutable materialized P6 items; all three baselines share one set.

This internal batch format stores grid-frame positions, not contract-v0 scene
positions. It does not replace the P6 evaluation contract or regenerate folds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FORMAT = "p7.frozen.1"
ARRAY_KEYS = ("observed_cells", "observed_free", "observed_wall", "valid_cells",
              "target_occupancy", "target_boundary", "src_pos", "mic_pos", "rir_valid")


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sample_sha(metadata, arrays):
    digest = hashlib.sha256(json_sha(metadata).encode())
    for key in ARRAY_KEYS:
        a = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(a.dtype).encode())
        digest.update(json.dumps(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def validate_item(arrays, metadata):
    grid = metadata["grid"]
    if (grid["width"], grid["height"], grid["cell_size_m"]) != (64, 64, 0.2):
        raise ValueError("P7 freezes the P6 64x64 / 0.2 m grid")
    for key in ARRAY_KEYS[:6]:
        a = arrays[key]
        if a.shape != (64, 64) or not np.isin(a, [0, 1]).all():
            raise ValueError(f"{key}: expected binary 64x64")
    free, wall, seen = (arrays[k].astype(bool) for k in ARRAY_KEYS[1:3] + ("observed_cells",))
    if (free & wall).any() or ((free | wall) & ~seen).any():
        raise ValueError("observed masks are inconsistent")
    validity = arrays["rir_valid"]
    if validity.shape != (8,) or not np.isin(validity, [0, 1]).all():
        raise ValueError("expected eight binary position validity rows")
    for name in ("src_pos", "mic_pos"):
        positions = arrays[name]
        if positions.shape != (8, 3) or not np.isfinite(positions).all():
            raise ValueError("positions must be finite 8x3 grid-frame metres")
        if np.any(positions[validity == 0] != 0):
            raise ValueError("invalid position rows must be zero")
        cells = np.floor(positions[validity == 1, :2] / 0.2).astype(int)
        if np.any((cells < 0) | (cells >= 64)) or not free[cells[:, 1], cells[:, 0]].all():
            raise ValueError("valid source/mic poses must occupy observed free cells")
    if not (arrays["valid_cells"].astype(bool) & ~seen).any():
        raise ValueError("frozen sample has no valid unobserved cells")


def freeze_samples(directory, splits, source):
    """Materialize split iterables of P6 item dictionaries. Refuse overwrites."""
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise ValueError("freeze destination must be empty; create a new version")
    root.mkdir(parents=True, exist_ok=True)
    entries, groups, rooms, ids = [], {}, {}, set()
    for split in ("train", "val", "test"):
        number = 0
        for item in splits[split]:
            sample_id, room, group = item["sample_id"], item["room_id"], str(item["room_group"])
            if sample_id in ids:
                raise ValueError("duplicate frozen sample ID")
            if groups.get(group, split) != split or rooms.get(room, split) != split:
                raise ValueError("physical room/group crosses frozen splits")
            groups[group] = rooms[room] = split
            ids.add(sample_id)
            arrays = {key: np.ascontiguousarray(item[key]) for key in ARRAY_KEYS}
            metadata = {key: item[key] for key in ("sample_id", "room_id", "room_group", "grid", "spatial_transform")}
            validate_item(arrays, metadata)
            name = f"samples/{split}_{number:06d}.npz"
            (root / "samples").mkdir(exist_ok=True)
            np.savez_compressed(root / name, **arrays)
            entries.append({"split": split, "path": name, "metadata": metadata, "sha256": sample_sha(metadata, arrays)})
            number += 1
        if number == 0:
            raise ValueError(f"frozen {split} split is empty")
    manifest = {"format": FORMAT, "source": source, "samples": entries}
    manifest["checksum"] = json_sha(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root / "manifest.json"


class FrozenDataset(Dataset):
    def __init__(self, manifest_path, split):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        body = {k: v for k, v in self.manifest.items() if k != "checksum"}
        if self.manifest.get("format") != FORMAT or self.manifest.get("checksum") != json_sha(body):
            raise ValueError("frozen manifest checksum/format mismatch")
        if split not in ("train", "val", "test"):
            raise ValueError("unknown split")
        groups, rooms, ids = {}, {}, set()
        for entry in self.manifest["samples"]:
            meta, assigned = entry["metadata"], entry["split"]
            room, group = meta["room_id"], str(meta["room_group"])
            if assigned not in ("train", "val", "test") or meta["sample_id"] in ids:
                raise ValueError("invalid split or duplicate sample ID")
            if groups.get(group, assigned) != assigned or rooms.get(room, assigned) != assigned:
                raise ValueError("room/group crosses frozen splits")
            groups[group] = rooms[room] = assigned
            ids.add(meta["sample_id"])
        self.entries = [e for e in self.manifest["samples"] if e["split"] == split]
        if not self.entries:
            raise ValueError(f"empty frozen {split}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        path = (self.path.parent / entry["path"]).resolve()
        if not path.is_relative_to(self.path.parent.resolve()):
            raise ValueError("sample path escapes frozen directory")
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in ARRAY_KEYS}
        if sample_sha(entry["metadata"], arrays) != entry["sha256"]:
            raise ValueError("frozen sample checksum mismatch")
        validate_item(arrays, entry["metadata"])
        return {key: torch.from_numpy(value) for key, value in arrays.items()}


def freeze_p6(destination, room_cache_dir, eval_dir, folds_path, eval_spec_path, fold=0,
              seed=0, samples_per_room=16, limit_per_split=None, k_eval=8):
    """Freeze training epoch zero once and the committed fixed val/test masks.

    Optional limits are recorded and intended for integration runs, not claims
    about the complete fold. Training augmentation happens once during freezing.
    """
    from echospace.data import EchoSpaceDataset, load_eval_spec, load_folds
    from dataclasses import asdict
    from echospace.acoustics import AugmentConfig, RirConfig
    from echospace.acoustics.rir import PROCESSOR_VERSION
    if limit_per_split is not None and limit_per_split < 1:
        raise ValueError("limit_per_split must be positive")
    folds, spec = load_folds(folds_path), load_eval_spec(eval_spec_path)
    if not Path(room_cache_dir).is_dir() or not (Path(eval_dir) / "manifest.jsonl").is_file():
        raise FileNotFoundError("P6 generated artifacts are missing: build room caches and fixed eval samples first")
    datasets = {split: EchoSpaceDataset(room_cache_dir, folds, fold, split, eval_dir=eval_dir,
                                       eval_spec=spec, seed=seed, samples_per_room=samples_per_room, k_eval=k_eval)
                for split in ("train", "val", "test")}
    def items(dataset):
        n = len(dataset) if limit_per_split is None else min(len(dataset), limit_per_split)
        for i in range(n):
            yield dataset[i]
    source = {"kind": "P6", "fold": fold, "seed": seed, "train_epoch": 0,
              "samples_per_room": samples_per_room, "limit_per_split": limit_per_split, "k_eval": k_eval,
              "folds_sha256": hashlib.sha256(Path(folds_path).read_bytes()).hexdigest(),
              "eval_spec_checksum": spec["checksum"]}
    source.update(p5_processor_version=PROCESSOR_VERSION, rir_config=asdict(RirConfig()),
                  training_augmentation=asdict(AugmentConfig()))
    return freeze_samples(destination, {s: items(d) for s, d in datasets.items()}, source)


def freeze_fixed_p6(destination, eval_dir, folds_path, eval_spec_path, fold=0, k=8):
    """Use verified P6 fixed masks on all three fold-assigned room sets.

    Training sees only training-room masks. No scan redraw, waveform augmentation
    or spatial augmentation is applied. This explicit alternative needs no room
    caches and still keeps held-out rooms/groups isolated.
    """
    from echospace.data import load_eval_spec, load_folds
    from echospace.data.dataset import EchoSpaceDataset, grid_from_record, to_grid_metres
    from echospace.data.evalmasks import read_eval_sample, read_manifest
    if not 1 <= k <= 8:
        raise ValueError("K must be within 1..8")
    folds, spec = load_folds(folds_path), load_eval_spec(eval_spec_path)
    manifest = read_manifest(eval_dir)
    expected_ids = {e["sample_id"] for e in spec["samples"]}
    if set(manifest) != expected_ids:
        raise ValueError("evaluation manifest IDs differ from the committed spec")
    def items(split):
        wanted = set(folds["folds"][fold][f"{split}_rooms"])
        for entry in spec["samples"]:
            if entry["room_id"] not in wanted:
                continue
            record = manifest[entry["sample_id"]]
            sample = read_eval_sample(eval_dir, record, entry["sha256"])
            arrays = sample.arrays
            grid = grid_from_record(record["grid"])
            valid = arrays["rir_valid"].astype(np.uint8).copy()
            valid[k:] = 0
            positions = arrays["rir_positions_scene_m"].astype(np.float64)
            uvh = to_grid_metres(grid, positions.reshape(-1, 3)).reshape(-1, 2, 3)
            uvh[valid == 0] = 0
            waveforms = arrays["rir_waveforms"].astype(np.float32).copy()
            waveforms[k:] = 0
            item = EchoSpaceDataset._item(arrays, waveforms, valid, uvh, k, record["coverage"],
                                          record["coverage_bin"], record["occlusion_type"], record["room_id"],
                                          record["room_group"], record["sample_id"], record["grid"], 0, False)
            yield item
    source = {"kind": "P6_fixed_masks", "fold": fold, "k": k, "training_augmentation": "none",
              "folds_sha256": hashlib.sha256(Path(folds_path).read_bytes()).hexdigest(),
              "eval_spec_checksum": spec["checksum"],
              "note": "all fold-assigned room masks; train rooms only for optimization; no room-cache scan redraw"}
    return freeze_samples(destination, {s: items(s) for s in ("train", "val", "test")}, source)
