"""Run a registered model on a verified sample and assemble the result with its provenance.

Two outputs are kept apart on purpose:

* ``occupancy_probability`` / ``boundary_probability``: the network's raw
  probabilities on every cell, exactly what P9 scored (contract
  ``prediction.npz``). Metrics use only these, on valid unobserved cells.
* ``interior``: a display composition. Measured free cells are pasted as
  inside (on training rooms 100 % of observed free cells lie inside the room).
  Measured wall-hit cells are drawn as outside: their centres fall inside the
  room only about half the time, so neither value is verified, and drawing
  them outside puts the rendered wall within one cell (0.2 m) of the hit.
  Unobserved cells take the thresholded prediction.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from echospace.contract_v0 import SCHEMA_VERSION, validate_prediction
from echospace.inference.registry import ModelEntry
from echospace.inference.samples import ServedSample, model_inputs

INFERENCE_VERSION = "p10.inference.1"
DISPLAY_CONVENTION = (
    "interior = thresholded occupancy on unobserved cells; observed free cells pasted inside; "
    "observed wall-hit cells drawn outside (their true occupancy is not measured)"
)


@dataclass(frozen=True)
class Reconstruction:
    reconstruction_id: str
    model: ModelEntry
    sample: ServedSample
    k_requested: int
    k_used: int
    occupancy_probability: np.ndarray
    boundary_probability: np.ndarray
    interior: np.ndarray
    metrics: dict[str, Any] | None
    provenance: dict[str, Any]

    def prediction_arrays(self) -> dict[str, np.ndarray]:
        return {"occupancy_probability": self.occupancy_probability, "boundary_probability": self.boundary_probability}


def reconstruction_id(checkpoint_sha256: str, sample_sha256: str, k: int, threshold: float) -> str:
    """Deterministic: the same model, sample, K and threshold always give the same id."""
    key = f"{INFERENCE_VERSION}|{checkpoint_sha256}|{sample_sha256}|{k}|{threshold!r}"
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def masked_iou(probability: np.ndarray, target: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    """P9's metric: IoU at the threshold over masked cells; an empty union counts as 1."""
    predicted = probability >= threshold
    target = target.astype(bool)
    union = int(((predicted | target) & mask).sum())
    return 1.0 if union == 0 else int((predicted & target & mask).sum()) / union


def compose_interior(occupancy_probability: np.ndarray, observed_free: np.ndarray, observed_wall: np.ndarray, threshold: float) -> np.ndarray:
    interior = occupancy_probability >= threshold
    interior[observed_free.astype(bool)] = True
    interior[observed_wall.astype(bool)] = False
    return interior.astype(np.uint8)


def code_revision(repo_root: Path) -> str:
    try:
        result = subprocess.run(["git", "-c", f"safe.directory={repo_root.resolve().as_posix()}", "rev-parse", "--short", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True, timeout=5)
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def reconstruct(model: Any, entry: ModelEntry, sample: ServedSample, k: int, *, split: str, revision: str, device: str = "cpu") -> Reconstruction:
    import torch

    inputs = model_inputs(sample, k)
    k_used = int(inputs.pop("k_used")[0])
    batch = {key: torch.from_numpy(value).to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.to(device)(batch)
    occupancy = output["occupancy_logits"].float().sigmoid()[0].cpu().numpy().astype(np.float32)
    boundary = output["boundary_logits"].float().sigmoid()[0].cpu().numpy().astype(np.float32)
    arrays = sample.arrays
    validate_prediction(sample.record, {"occupancy_probability": occupancy, "boundary_probability": boundary})

    unobserved = (arrays["observed_cells"] == 0) & (arrays["valid_cells"] == 1)
    metrics = {
        "scope": "valid unobserved cells (M = 0), threshold as registered; same definition as P9",
        "occupancy_iou": masked_iou(occupancy, arrays["target_occupancy"], unobserved, entry.threshold),
        "boundary_iou": masked_iou(boundary, arrays["target_boundary"], unobserved, entry.threshold),
        "unobserved_cells": int(unobserved.sum()),
    }
    identifier = reconstruction_id(entry.checkpoint_sha256, sample.sha256, k, entry.threshold)
    record = sample.record
    provenance = {
        "format": "echospace.provenance.1",
        "reconstruction_id": identifier,
        "inference_version": INFERENCE_VERSION,
        "code_revision": revision,
        "contract_schema_version": SCHEMA_VERSION,
        "model": entry.public(),
        "sample": {
            "sample_id": sample.sample_id,
            "sha256": sample.sha256,
            "room_id": record["room_id"],
            "room_group": record["room_group"],
            "split_for_model_fold": split,
            "occlusion_type": record["occlusion_type"],
            "coverage": record["coverage"],
            "coverage_bin": record["coverage_bin"],
            "scan_seed": record["scan_seed"],
            "rir_ids": record["rir_ids"][:k_used],
            "frame": record["frame"],
            "grid": record["grid"],
        },
        "k_requested": k,
        "k_used": k_used,
        "threshold": entry.threshold,
        "display_convention": DISPLAY_CONVENTION,
        "wall_height": "assumed for display; the model predicts a 2D footprint only",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "data_attribution": "AcousticRooms (Meta Platforms, Inc.), CC BY 4.0; see DATA_LICENSE.md",
    }
    interior = compose_interior(occupancy, arrays["observed_free"], arrays["observed_wall"], entry.threshold)
    return Reconstruction(identifier, entry, sample, k, k_used, occupancy, boundary, interior, metrics, provenance)
