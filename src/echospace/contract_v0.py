"""Version 0.1 sample contracts for EchoSpace data and predictions.

The grid geometry is recorded per sample. No dataset orientation, grid shape, or
resolution is assumed by this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


SCHEMA_VERSION = "0.1.0"
SAMPLE_ARRAYS = (
    "observed_points_scene_m",
    "observed_cells",
    "observed_free",
    "observed_wall",
    "valid_cells",
    "target_occupancy",
    "target_boundary",
    "rir_waveforms",
    "rir_positions_scene_m",
    "rir_valid",
)
PREDICTION_ARRAYS = ("occupancy_probability", "boundary_probability")


class ContractError(ValueError):
    """A manifest or array artifact violates the versioned contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must contain numbers") from exc
    _require(vector.shape == (size,), f"{name} must have {size} numbers")
    _require(bool(np.isfinite(vector).all()), f"{name} must be finite")
    return vector


def _positive_number(value: Any, name: str) -> float:
    _require(not isinstance(value, bool), f"{name} must be positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be positive") from exc
    _require(np.isfinite(result) and result > 0, f"{name} must be positive")
    return result


def validate_manifest_record(record: Mapping[str, Any]) -> None:
    """Validate one portable JSONL record before loading its NPZ sample."""
    _require(record.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    for name in ("sample_id", "room_id", "dataset", "geometry_id", "sample_path"):
        _require(isinstance(record.get(name), str) and bool(record[name]), f"{name} is required")
    sample_path = Path(record["sample_path"])
    _require(not sample_path.is_absolute() and ".." not in sample_path.parts, "sample_path must be relative")
    _require(record.get("split") in ("unassigned", "train", "val", "test"), "invalid split")
    rir_ids = record.get("rir_ids")
    _require(isinstance(rir_ids, list) and all(isinstance(item, str) and item for item in rir_ids), "rir_ids must be a list of strings")
    rate = record.get("rir_sample_rate_hz")
    _require(type(rate) is int and rate > 0, "rir_sample_rate_hz must be a positive integer")

    frame = record.get("frame")
    _require(isinstance(frame, dict), "frame is required")
    _positive_number(frame.get("source_units_to_meters"), "source_units_to_meters")
    _require(isinstance(frame.get("description"), str) and bool(frame["description"]), "frame.description is required")
    try:
        transform = np.asarray(frame.get("source_to_scene"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError("source_to_scene must contain numbers") from exc
    _require(transform.shape == (4, 4), "source_to_scene must be 4x4")
    _require(bool(np.isfinite(transform).all()), "source_to_scene must be finite")
    _require(bool(np.allclose(transform[3], [0, 0, 0, 1])), "source_to_scene must be affine")
    _require(abs(float(np.linalg.det(transform[:3, :3]))) > 1e-10, "source_to_scene must be invertible")

    grid = record.get("grid")
    _require(isinstance(grid, dict), "grid is required")
    for name in ("width", "height"):
        _require(type(grid.get(name)) is int and grid[name] > 0, f"grid.{name} must be positive")
    _positive_number(grid.get("cell_size_m"), "grid.cell_size_m")
    _vector(grid.get("origin_scene_m"), 3, "grid.origin_scene_m")
    u = _vector(grid.get("axis_u_scene"), 3, "grid.axis_u_scene")
    v = _vector(grid.get("axis_v_scene"), 3, "grid.axis_v_scene")
    _require(abs(float(np.linalg.norm(u)) - 1) < 1e-5, "grid.axis_u_scene must be a unit vector")
    _require(abs(float(np.linalg.norm(v)) - 1) < 1e-5, "grid.axis_v_scene must be a unit vector")
    _require(abs(float(np.dot(u, v))) < 1e-5, "grid axes must be orthogonal")


def _arrays(artifact: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, np.ndarray]:
    missing = set(names) - set(artifact)
    _require(not missing, f"missing arrays: {', '.join(sorted(missing))}")
    return {name: np.asarray(artifact[name]) for name in names}


def _binary(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    _require(array.shape == shape, f"{name} must have shape {shape}")
    _require(array.dtype == np.bool_ or bool(np.isin(array, [0, 1]).all()), f"{name} must be binary")


def validate_sample(record: Mapping[str, Any], artifact: Mapping[str, Any]) -> None:
    """Validate sample arrays against their own manifest grid and RIR IDs."""
    validate_manifest_record(record)
    arrays = _arrays(artifact, SAMPLE_ARRAYS)
    shape = (record["grid"]["height"], record["grid"]["width"])
    for name in ("observed_cells", "observed_free", "observed_wall", "valid_cells", "target_occupancy", "target_boundary"):
        _binary(arrays[name], shape, name)
    valid = arrays["valid_cells"].astype(bool)
    observed = arrays["observed_cells"].astype(bool)
    free = arrays["observed_free"].astype(bool)
    wall = arrays["observed_wall"].astype(bool)
    _require(not bool((observed & ~valid).any()), "observed cells must be valid")
    _require(not bool(((free | wall) & ~observed).any()), "observed states require visibility")
    _require(not bool((free & wall).any()), "observed free and wall cells must be disjoint")
    _require(not bool((arrays["target_boundary"].astype(bool) & ~valid).any()), "target boundary must be valid")
    _require(not bool((arrays["target_occupancy"].astype(bool) & ~valid).any()), "target occupancy must be valid")

    points = arrays["observed_points_scene_m"]
    _require(points.ndim == 2 and points.shape[1] == 3, "observed points must be Nx3")
    _require(np.issubdtype(points.dtype, np.number) and bool(np.isfinite(points).all()), "observed points must be finite numbers")
    rir = arrays["rir_waveforms"]
    _require(rir.ndim == 2 and rir.shape[0] == len(record["rir_ids"]) and rir.shape[1] > 0, "RIR waveforms must be RxT")
    _require(np.issubdtype(rir.dtype, np.number) and bool(np.isfinite(rir).all()), "RIR waveforms must be finite numbers")
    positions = arrays["rir_positions_scene_m"]
    _require(positions.shape == (len(record["rir_ids"]), 2, 3), "RIR positions must be Rx2x3")
    _require(np.issubdtype(positions.dtype, np.number) and bool(np.isfinite(positions).all()), "RIR positions must be finite numbers")
    _binary(arrays["rir_valid"], (len(record["rir_ids"]),), "rir_valid")
    grid = record["grid"]
    origin = np.asarray(grid["origin_scene_m"], dtype=np.float64)
    axes = (np.asarray(grid["axis_u_scene"], dtype=np.float64), np.asarray(grid["axis_v_scene"], dtype=np.float64))
    for pair in positions[arrays["rir_valid"].astype(bool)]:
        for point in pair:
            relative = point - origin
            column = int(np.floor(np.dot(relative, axes[0]) / grid["cell_size_m"]))
            row = int(np.floor(np.dot(relative, axes[1]) / grid["cell_size_m"]))
            _require(0 <= row < shape[0] and 0 <= column < shape[1], "valid RIR pose must lie on the grid")
            _require(bool(free[row, column]), "valid RIR source and receiver must lie in observed free space")


def validate_prediction(record: Mapping[str, Any], artifact: Mapping[str, Any]) -> None:
    """Validate output probabilities on the exact input sample grid."""
    validate_manifest_record(record)
    arrays = _arrays(artifact, PREDICTION_ARRAYS)
    shape = (record["grid"]["height"], record["grid"]["width"])
    for name, array in arrays.items():
        _require(array.shape == shape, f"{name} must have shape {shape}")
        _require(np.issubdtype(array.dtype, np.number), f"{name} must be numeric")
        _require(bool(np.isfinite(array).all()) and bool(((array >= 0) & (array <= 1)).all()), f"{name} must contain probabilities")


def read_manifest(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read and validate JSONL while rejecting duplicate sample IDs."""
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                _require(isinstance(record, dict), "record must be an object")
                validate_manifest_record(record)
                _require(record["sample_id"] not in seen, "duplicate sample_id")
            except (json.JSONDecodeError, ContractError) as exc:
                raise ContractError(f"{path}:{line_number}: {exc}") from exc
            seen.add(record["sample_id"])
            yield record


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load numeric NPZ arrays without enabling pickle deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
