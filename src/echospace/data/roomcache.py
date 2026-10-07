"""Per-room cache: 1.1 m geometry plus every processed RIR (plan section 1.7).

One ``<room_id>.npz`` holds what the training loader needs, so a notebook can
simulate scans and pick RIRs without the meshes or the source archives:

- the P4 ``RoomGeometry`` (footprint, free region, slice segments, walls),
  polygons stored as WKB;
- all audited source/receiver pairs: ``rir_ids``, scene positions, and the
  P5-processed clean waveforms (emission timing, peak-normalised, 16 kHz,
  80 ms) stored as float16 as in the plan's cache format;
- provenance as JSON: D0 audit record hash, P5 processor version and config,
  each source WAV's SHA-256, cache format version.

Augmented audio is never cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import shapely

from echospace.acoustics import ProcessedRir, RirCandidate, RirConfig
from echospace.acoustics.rir import PROCESSOR_VERSION
from echospace.scans import RoomGeometry

CACHE_FORMAT = "p6.room.1"


class RoomCacheError(ValueError):
    """A room cache file is missing, corrupt or from another configuration."""


@dataclass(frozen=True)
class RoomCache:
    room_id: str
    room_group: int
    geometry: RoomGeometry
    rir_ids: tuple[str, ...]
    positions_scene_m: np.ndarray  # (R, 2, 3) float32, source then receiver
    waveforms: np.ndarray  # (R, T) float16
    meta: dict[str, Any]

    def candidates(self) -> tuple[RirCandidate, ...]:
        """P5 candidates backed by this cache (no archive references needed)."""
        audit = self.meta["audit_record_sha256"]
        return tuple(
            RirCandidate(self.room_id, _pair(rir_id), self.positions_scene_m[i], None, None,  # type: ignore[arg-type]
                         self.meta["metadata_sha256"][i], audit, verified=True)
            for i, rir_id in enumerate(self.rir_ids)
        )

    def loader(self, config: RirConfig) -> Callable[[RirCandidate], ProcessedRir]:
        """``build_bundle`` loader returning cached clean audio as float32."""
        if config.to_dict() != self.meta["rir_config"]:
            raise RoomCacheError("room cache was built with another RIR config")
        row = {rir_id: i for i, rir_id in enumerate(self.rir_ids)}

        def load(candidate: RirCandidate) -> ProcessedRir:
            i = row[candidate.rir_id]
            return ProcessedRir(
                self.waveforms[i].astype(np.float32),
                {"config": self.meta["rir_config"], "processor_version": self.meta["processor_version"],
                 "rir_id": candidate.rir_id, "source_wav_sha256": self.meta["wav_sha256"][i],
                 "storage": "float16 room cache", "cache_format": CACHE_FORMAT},
            )

        return load


def _pair(rir_id: str) -> tuple[int, int]:
    source, receiver = rir_id.rsplit("/", 1)[1].split("_")
    return int(source[1:]), int(receiver[1:])


def save_room_cache(
    path: str | Path,
    room_id: str,
    room_group: int,
    geometry: RoomGeometry,
    rir_ids: list[str],
    positions_scene_m: np.ndarray,
    waveforms: np.ndarray,
    meta: dict[str, Any],
) -> None:
    """Write atomically; ``meta`` must carry the provenance fields listed above."""
    waveforms16 = np.asarray(waveforms, dtype=np.float16)
    if not np.isfinite(waveforms16).all():
        raise RoomCacheError("waveforms do not fit float16")
    record = dict(meta)
    record.update({
        "cache_format": CACHE_FORMAT, "room_id": room_id, "room_group": int(room_group),
        "n_rirs": len(rir_ids), "scan_height_scene_m": geometry.scan_height_scene_m,
        "waveforms_sha256": hashlib.sha256(waveforms16.tobytes()).hexdigest(),
    })
    arrays = {
        "footprint_wkb": np.frombuffer(shapely.to_wkb(geometry.footprint), dtype=np.uint8),
        "free_region_wkb": np.frombuffer(shapely.to_wkb(geometry.free_region), dtype=np.uint8),
        "segments": np.asarray(geometry.segments, dtype=np.float64),
        "walls": np.asarray([np.stack(w) for w in geometry.walls], dtype=np.float64).reshape(-1, 2, 2),
        "rir_ids": np.asarray(rir_ids, dtype=np.str_),
        "positions_scene_m": np.asarray(positions_scene_m, dtype=np.float32).reshape(-1, 2, 3),
        "waveforms": waveforms16,
        "meta_utf8": np.frombuffer(json.dumps(record, sort_keys=True).encode(), dtype=np.uint8),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def load_room_cache(path: str | Path, expected_config: RirConfig | None = None) -> RoomCache:
    """Load and integrity-check one room cache."""
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["meta_utf8"].tobytes().decode("utf-8"))
            footprint = shapely.from_wkb(data["footprint_wkb"].tobytes())
            free_region = shapely.from_wkb(data["free_region_wkb"].tobytes())
            segments = data["segments"].copy()
            walls = tuple((w[0].copy(), w[1].copy()) for w in data["walls"])
            rir_ids = tuple(str(x) for x in data["rir_ids"])
            positions = data["positions_scene_m"].copy()
            waveforms = data["waveforms"].copy()
    except (OSError, KeyError, ValueError) as error:
        raise RoomCacheError(f"cannot read room cache {path}: {error}") from error
    if meta.get("cache_format") != CACHE_FORMAT:
        raise RoomCacheError(f"{path}: cache format {meta.get('cache_format')} != {CACHE_FORMAT}")
    if hashlib.sha256(waveforms.tobytes()).hexdigest() != meta["waveforms_sha256"]:
        raise RoomCacheError(f"{path}: waveform checksum mismatch")
    if len(rir_ids) != len(positions) or len(rir_ids) != len(waveforms) or len(rir_ids) != meta["n_rirs"]:
        raise RoomCacheError(f"{path}: row counts disagree")
    if expected_config is not None and meta["rir_config"] != expected_config.to_dict():
        raise RoomCacheError(f"{path}: built with another RIR config")
    if meta["processor_version"] != PROCESSOR_VERSION:
        raise RoomCacheError(f"{path}: P5 processor {meta['processor_version']} != {PROCESSOR_VERSION}")
    geometry = RoomGeometry(footprint, free_region, segments, walls, float(meta["scan_height_scene_m"]))
    return RoomCache(meta["room_id"], int(meta["room_group"]), geometry, rir_ids, positions, waveforms, meta)


def eligible_mask(cache: RoomCache, grid: Any, observed_free: np.ndarray) -> np.ndarray:
    """Which cached pairs have both poses in observed free space (vectorised P5 rule)."""
    if len(cache.rir_ids) == 0:
        return np.zeros(0, dtype=bool)
    points = cache.positions_scene_m.reshape(-1, 3).astype(np.float64)
    columns, rows = np.floor(grid.scene_to_cell(points)).astype(int).T
    inside = (rows >= 0) & (rows < grid.height) & (columns >= 0) & (columns < grid.width)
    free = np.zeros(len(points), dtype=bool)
    free[inside] = np.asarray(observed_free, dtype=bool)[rows[inside], columns[inside]]
    return free.reshape(-1, 2).all(axis=1)
