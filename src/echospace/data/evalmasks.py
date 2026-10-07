"""Fixed evaluation masks, written once as contract v0.1.0 samples (plan section 1.5).

For every room, every applicable occlusion type and every coverage bin, up to
``per_bin`` scans are kept that have at least ``k_required`` usable pairs
(both poses in observed free space), so K = 1, 2, 4 and 8 are evaluated on
the same masks through P5's nested ranking. Each kept scan is written as a
validated contract sample (manifest JSONL + NPZ) with all 8 RIRs.

The masks are generated once. ``eval_masks.json`` (committed) lists every
sample with a SHA-256 of its content; loaders refuse a sample whose content
differs, so a test mask can never be silently regenerated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from echospace.acoustics import AcousticError, RirConfig, build_bundle
from echospace.acoustics.rir import PROCESSOR_VERSION
from echospace.contract_v0 import load_npz, validate_sample
from echospace.geometry.frames import ACOUSTICROOMS_SOURCE_TO_SCENE
from echospace.geometry.raster import GeometryLabelError
from echospace.scans import COVERAGE_BINS, OCCLUSION_TYPES, FrameError, ScanNotApplicable, ScanNotFeasible, build_scan_sample

from .roomcache import RoomCache, eligible_mask

EVAL_FORMAT = "p6.eval.1"
K_REQUIRED = 8
PER_BIN = 3
SEED_BASE = 1_000_000  # disjoint from P4 gate seeds and training draws
MAX_SEEDS = 12
MAX_CONSECUTIVE_REFUSALS = 2  # stop a (type, bin) after this many bin/canvas refusals in a row
FRAME_DESCRIPTION = "AcousticRooms source (x, y, z up, metres) to scene (X right, Y up, Z = -y); grid from the scan only (P4)"


class EvalMaskError(ValueError):
    """An evaluation sample is missing or differs from its committed checksum."""


@dataclass(frozen=True)
class EvalSample:
    record: dict[str, Any]
    arrays: dict[str, np.ndarray]


def content_sha256(record: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    """Checksum of a sample's arrays and record, independent of NPZ compression."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}|{array.dtype.str}|{array.shape}|".encode())
        digest.update(array.tobytes())
    body = {k: v for k, v in record.items() if k != "sample_path"}
    digest.update(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def masks_for_room(
    cache: RoomCache,
    rir_config: RirConfig,
    per_bin: int = PER_BIN,
    k_required: int = K_REQUIRED,
    max_seeds: int = MAX_SEEDS,
) -> tuple[list[EvalSample], dict[str, dict[str, dict[str, Any]]]]:
    """Fixed samples for one room, and per (type, bin) what was possible."""
    candidates = cache.candidates()
    loader = cache.loader(rir_config)
    samples: list[EvalSample] = []
    feasibility: dict[str, dict[str, dict[str, Any]]] = {}
    for kind in OCCLUSION_TYPES:
        feasibility[kind] = {}
        for bin_name in COVERAGE_BINS:
            found, scans, short, refusals, applicable = 0, 0, 0, 0, True
            for s in range(max_seeds):
                seed = SEED_BASE + s
                try:
                    scan = build_scan_sample(cache.geometry, cache.room_id, kind, seed, bin_name, max_attempts=15)
                except ScanNotApplicable:
                    applicable = False
                    break
                except (ScanNotFeasible, GeometryLabelError, FrameError):
                    refusals += 1
                    if refusals >= MAX_CONSECUTIVE_REFUSALS:
                        break
                    continue
                refusals = 0
                scans += 1
                free = scan.observed.observed_free
                usable = np.flatnonzero(eligible_mask(cache, scan.grid, free))
                if len(usable) < k_required:
                    short += 1
                    continue
                sample_id = f"{cache.room_id}__{kind}__{bin_name}__{seed}"
                bundle = build_bundle([candidates[i] for i in usable], scan.grid, free, loader, room_id=cache.room_id,
                                      selection_key=sample_id, seed=0, k=k_required, config=rir_config)
                arrays = scan.contract_arrays()
                arrays.update(bundle.contract_arrays())
                record = {
                    "schema_version": "0.1.0",
                    "sample_id": sample_id,
                    "room_id": cache.room_id,
                    "dataset": "AcousticRooms",
                    "geometry_id": cache.meta["geometry_signature"],
                    "rir_ids": list(bundle.rir_ids),
                    "rir_sample_rate_hz": rir_config.sample_rate_hz,
                    "sample_path": f"samples/{sample_id}.npz",
                    "split": "unassigned",
                    "frame": {"description": FRAME_DESCRIPTION, "source_units_to_meters": 1.0,
                              "source_to_scene": ACOUSTICROOMS_SOURCE_TO_SCENE.tolist()},
                    "grid": scan.grid.manifest_grid(),
                    "room_group": cache.room_group,
                    "occlusion_type": kind,
                    "coverage": round(float(scan.coverage), 6),
                    "coverage_bin": bin_name,
                    "scan_seed": seed,
                    "eligible_pairs": int(len(usable)),
                    "p4_heading_rad": float(scan.grid.heading_rad),
                    "p5_processor_version": PROCESSOR_VERSION,
                    "p5_rir_config": rir_config.to_dict(),
                    "eval_format": EVAL_FORMAT,
                }
                validate_sample(record, arrays)
                samples.append(EvalSample(record, arrays))
                found += 1
                if found == per_bin:
                    break
            feasibility[kind][bin_name] = {"applicable": applicable, "scans": scans,
                                           "short_of_pairs": short, "masks": found}
            if not applicable:
                for rest in COVERAGE_BINS:
                    feasibility[kind][rest] = {"applicable": False, "scans": 0, "short_of_pairs": 0, "masks": 0}
                break
    return samples, feasibility


def write_sample(sample: EvalSample, root: str | Path) -> dict[str, Any]:
    """Write NPZ under ``root`` and return the spec entry with its checksum."""
    path = Path(root) / sample.record["sample_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **sample.arrays)
    return {
        "sample_id": sample.record["sample_id"],
        "room_id": sample.record["room_id"],
        "occlusion_type": sample.record["occlusion_type"],
        "coverage_bin": sample.record["coverage_bin"],
        "coverage": sample.record["coverage"],
        "scan_seed": sample.record["scan_seed"],
        "eligible_pairs": sample.record["eligible_pairs"],
        "sha256": content_sha256(sample.record, sample.arrays),
    }


def spec_checksum(spec: dict[str, Any]) -> str:
    body = {k: v for k, v in spec.items() if k != "checksum"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_eval_spec(path: str | Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("format") != EVAL_FORMAT or spec.get("checksum") != spec_checksum(spec):
        raise EvalMaskError(f"{path}: unknown format or edited spec")
    return spec


def read_eval_sample(root: str | Path, record: dict[str, Any], expected_sha256: str) -> EvalSample:
    """Load one fixed sample and refuse it if it is not byte-for-byte the committed one."""
    arrays = load_npz(Path(root) / record["sample_path"])
    if content_sha256(record, arrays) != expected_sha256:
        raise EvalMaskError(f"{record['sample_id']}: content differs from eval_masks.json")
    validate_sample(record, arrays)
    return EvalSample(record, arrays)


def read_manifest(root: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(root) / "manifest.jsonl"
    if not path.exists():
        raise EvalMaskError(f"{path} missing: run scripts/make_eval_masks.py")
    with path.open(encoding="utf-8") as handle:
        return {r["sample_id"]: r for r in map(json.loads, handle)}


__all__ = [
    "AcousticError", "EVAL_FORMAT", "EvalMaskError", "EvalSample", "K_REQUIRED", "content_sha256",
    "load_eval_spec", "masks_for_room", "read_eval_sample", "read_manifest", "spec_checksum", "write_sample",
]
