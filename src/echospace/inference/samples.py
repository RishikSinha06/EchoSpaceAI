"""Verified access to the fixed evaluation samples the inference service serves.

Only samples listed in the committed ``data/splits/eval_masks.json`` exist for
the service, and each one is read through P6's ``read_eval_sample``, which
checks its SHA-256 and the v0.1.0 contract before any array is used. Model
inputs are built with the same conversion that produced the P9 test scores
(``EchoSpaceDataset._item`` with grid-frame positions), so a served prediction
and a reported score come from identical tensors.

The local sample folder may hold either P6's full ``manifest.jsonl`` or the
per-room ``rooms/<room_id>.json`` files that ``make_eval_masks.py --rooms``
writes; both carry the same records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from echospace.data.dataset import EchoSpaceDataset, grid_from_record, to_grid_metres
from echospace.data.evalmasks import read_eval_sample

K_MAX = 8


class SampleError(LookupError):
    """The sample is unknown, missing locally or fails verification."""


@dataclass(frozen=True)
class ServedSample:
    sample_id: str
    sha256: str
    record: dict[str, Any]
    arrays: dict[str, np.ndarray]


class SampleStore:
    def __init__(self, eval_dir: str | Path, spec_path: str | Path, folds_path: str | Path) -> None:
        self.eval_dir = Path(eval_dir)
        self.spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        self.folds = json.loads(Path(folds_path).read_text(encoding="utf-8"))
        self._spec_by_id = {entry["sample_id"]: entry for entry in self.spec["samples"]}
        self._records: dict[str, dict[str, Any]] | None = None

    # -- catalogue ---------------------------------------------------------
    def spec_samples(self) -> list[dict[str, Any]]:
        return list(self.spec["samples"])

    def _local_records(self) -> dict[str, dict[str, Any]]:
        if self._records is None:
            records: dict[str, dict[str, Any]] = {}
            manifest = self.eval_dir / "manifest.jsonl"
            if manifest.is_file():
                for line in manifest.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        record = json.loads(line)
                        records[record["sample_id"]] = record
            for part in sorted((self.eval_dir / "rooms").glob("*.json")):
                for record in json.loads(part.read_text(encoding="utf-8"))["records"]:
                    records.setdefault(record["sample_id"], record)
            self._records = records
        return self._records

    def available(self) -> list[dict[str, Any]]:
        """Spec entries whose files exist locally (not yet hash-checked)."""
        local = self._local_records()
        return [entry for entry in self.spec["samples"] if entry["sample_id"] in local]

    def split_of(self, room_id: str, fold: int) -> str:
        """Which part of a fold a room belongs to: train, val, test or excluded."""
        part = self.folds["folds"][fold]
        for split in ("train", "val", "test"):
            if room_id in part[f"{split}_rooms"]:
                return split
        return "excluded"

    # -- loading -----------------------------------------------------------
    def load(self, sample_id: str) -> ServedSample:
        entry = self._spec_by_id.get(sample_id)
        if entry is None:
            raise SampleError(f"unknown sample_id {sample_id!r}: not in the committed evaluation spec")
        record = self._local_records().get(sample_id)
        if record is None:
            raise SampleError(f"sample {sample_id!r} is not prepared locally; run scripts/prepare_inference_samples.py")
        try:
            sample = read_eval_sample(self.eval_dir, record, entry["sha256"])
        except Exception as exc:  # checksum or contract failure: never serve it
            raise SampleError(f"sample {sample_id!r} failed verification: {exc}") from exc
        return ServedSample(sample_id, entry["sha256"], record, sample.arrays)


def model_inputs(sample: ServedSample, k: int) -> dict[str, np.ndarray]:
    """Batch-of-one model inputs with the first ``k`` nested pairs, as in P9 evaluation."""
    if not 1 <= k <= K_MAX:
        raise ValueError(f"k must be within 1..{K_MAX}")
    arrays, record = sample.arrays, sample.record
    grid = grid_from_record(record["grid"])
    valid = arrays["rir_valid"].astype(np.uint8)
    uvh = to_grid_metres(grid, arrays["rir_positions_scene_m"].astype(np.float64).reshape(-1, 3)).reshape(-1, 2, 3)
    uvh[valid == 0] = 0
    item = EchoSpaceDataset._item(
        arrays, arrays["rir_waveforms"].astype(np.float32), valid, uvh, int(valid.sum()),
        record["coverage"], record["coverage_bin"], record["occlusion_type"], record["room_id"],
        record["room_group"], record["sample_id"], record["grid"], 0, False,
    )
    available = int(item["rir_valid"].sum())
    used = min(k, available)
    for key in ("rir_valid", "rir", "src_pos", "mic_pos"):
        item[key] = np.array(item[key], copy=True)
        item[key][used:] = 0  # nested prefix: rows >= K never reach the model
    keys = ("observed_cells", "observed_free", "observed_wall", "rir", "rir_valid", "src_pos", "mic_pos")
    batch = {key: np.ascontiguousarray(item[key])[None] for key in keys}
    batch["k_used"] = np.array([used])
    return batch
