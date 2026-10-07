"""EchoSpace dataset for one fold and split (plan sections 1.4-1.7).

Framework-free: items are dicts of NumPy arrays and plain values, so a PyTorch
``DataLoader`` can use the class directly (``collate`` stacks a batch).

- ``train``: a new partial scan per item and epoch (P4), with the plan's +-1 m
  frame shift, a random coverage bin and occlusion type the room supports, a
  random K in 1..8 usable pairs chosen by P5's ranking (the scan is redrawn
  until it has K usable pairs), P5 training augmentation, then a random 90
  degree rotation and flip applied to every grid and every position together.
- ``val`` / ``test``: the fixed evaluation samples from ``make_eval_masks.py``,
  checked against ``eval_masks.json``, with the first ``k_eval`` of their 8
  nested RIRs. No augmentation.

Positions are returned in the grid frame, in metres: ``(u, v, h)`` with ``u``
along grid columns, ``v`` along rows from the grid origin, and ``h`` the scene
height. ``grid`` and ``spatial_transform`` map them back to the scene.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from echospace.acoustics import AcousticError, AugmentConfig, RirConfig, augment_bundle, build_bundle
from echospace.geometry.frames import GridFrame
from echospace.geometry.raster import GeometryLabelError
from echospace.scans import COVERAGE_BINS, OCCLUSION_TYPES, FrameError, ScanNotApplicable, ScanNotFeasible, build_scan_sample
from echospace.scans.sample import TRAINING_SHIFT_M

from .evalmasks import read_eval_sample, read_manifest
from .roomcache import RoomCache, eligible_mask, load_room_cache

K_MAX = 8
GRID_ARRAYS = ("observed_cells", "observed_free", "observed_wall", "valid_cells", "target_occupancy", "target_boundary")


class DatasetError(RuntimeError):
    """No usable sample could be produced."""


def grid_from_record(grid: dict[str, Any], heading_rad: float = 0.0) -> GridFrame:
    """Rebuild a GridFrame from a manifest grid block (origin + axes)."""
    origin = np.asarray(grid["origin_scene_m"], dtype=np.float64)
    u, v = np.asarray(grid["axis_u_scene"]), np.asarray(grid["axis_v_scene"])
    size = grid["cell_size_m"]
    anchor = origin + grid["width"] * size / 2 * u + grid["height"] * size / 2 * v
    heading = float(np.arctan2(u[2], u[0]))
    return GridFrame(tuple(float(x) for x in anchor), heading, width=grid["width"], height=grid["height"], cell_size_m=size)


def to_grid_metres(grid: GridFrame, points_scene_m: np.ndarray) -> np.ndarray:
    """Scene points (N, 3) -> grid frame (u, v, h) metres."""
    points = np.asarray(points_scene_m, dtype=np.float64).reshape(-1, 3)
    cells = grid.scene_to_cell(points)
    return np.column_stack((cells[:, 0] * grid.cell_size_m, cells[:, 1] * grid.cell_size_m, points[:, 1]))


def spatial_transform(arrays: dict[str, np.ndarray], positions_uvh: np.ndarray, quarter_turns: int, flip: bool,
                      size_m: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Rotate every grid by ``quarter_turns`` x np.rot90 and optionally flip columns; move positions with them.

    Continuous column/row coordinates ``(c, r)`` on a square ``n x n`` grid map
    as: rot90 -> ``(r, n - c)``; column flip -> ``(n - c, r)``. With metres
    ``u = c * cell``, ``v = r * cell`` the same holds with ``n * cell = size_m``.
    """
    out = dict(arrays)
    uvh = np.array(positions_uvh, dtype=np.float64, copy=True)
    for name in GRID_ARRAYS:
        grid = out[name]
        if grid.shape[0] != grid.shape[1]:
            raise ValueError("spatial augmentation needs a square grid")
        out[name] = np.ascontiguousarray(np.rot90(grid, quarter_turns % 4))
        if flip:
            out[name] = np.ascontiguousarray(out[name][:, ::-1])
    for _ in range(quarter_turns % 4):
        u, v = uvh[:, 0].copy(), uvh[:, 1].copy()
        uvh[:, 0], uvh[:, 1] = v, size_m - u
    if flip:
        uvh[:, 0] = size_m - uvh[:, 0]
    return out, uvh


class EchoSpaceDataset:
    def __init__(
        self,
        room_cache_dir: str | Path,
        folds: dict[str, Any],
        fold: int,
        split: str,
        *,
        eval_dir: str | Path | None = None,
        eval_spec: dict[str, Any] | None = None,
        samples_per_room: int = 16,
        k_range: tuple[int, int] = (1, K_MAX),
        k_eval: int = K_MAX,
        seed: int = 0,
        augment: bool = True,
        rir_config: RirConfig = RirConfig(),
        augment_config: AugmentConfig = AugmentConfig(),
        shift_m: float = TRAINING_SHIFT_M,
        wall_dropout: float = 0.0,
        max_redraws: int = 8,
        max_cached_rooms: int = 32,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        if not 1 <= k_range[0] <= k_range[1] <= K_MAX or not 1 <= k_eval <= K_MAX:
            raise ValueError("K must be within 1..8")
        self.split, self.fold, self.seed, self.epoch = split, fold, seed, 0
        self.room_cache_dir = Path(room_cache_dir)
        self.rir_config, self.augment_config = rir_config, augment_config
        self.samples_per_room, self.k_range, self.k_eval = samples_per_room, k_range, k_eval
        self.augment, self.shift_m, self.wall_dropout, self.max_redraws = augment, shift_m, wall_dropout, max_redraws
        self._rooms: dict[str, RoomCache] = {}
        self._max_cached_rooms = max_cached_rooms
        rooms = sorted(folds["folds"][fold][f"{split}_rooms"])
        self.feasible = self._feasibility(eval_spec, rooms)
        if split == "train":
            self.rooms = [r for r in rooms if (self.room_cache_dir / f"{r}.npz").exists() and self.feasible.get(r)]
            self.items: list[Any] = [(r, i) for r in self.rooms for i in range(samples_per_room)]
        else:
            if eval_dir is None or eval_spec is None:
                raise ValueError("val/test need eval_dir and eval_spec")
            self.eval_dir = Path(eval_dir)
            manifest = read_manifest(eval_dir)
            wanted = set(rooms)
            self.items = [(manifest[e["sample_id"]], e["sha256"]) for e in eval_spec["samples"] if e["room_id"] in wanted]
            self.rooms = sorted({rec["room_id"] for rec, _ in self.items})

    @staticmethod
    def _feasibility(eval_spec: dict[str, Any] | None, rooms: list[str]) -> dict[str, list[tuple[str, str]]]:
        """(type, bin) combinations each room produced a scan for, from the eval generation."""
        if eval_spec is None:
            return {r: [(k, b) for k in OCCLUSION_TYPES for b in COVERAGE_BINS] for r in rooms}
        table = eval_spec["feasibility"]
        return {r: [(k, b) for k in OCCLUSION_TYPES for b in COVERAGE_BINS if table.get(r, {}).get(k, {}).get(b, {}).get("scans", 0) > 0]
                for r in rooms}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def room(self, room_id: str) -> RoomCache:
        if room_id not in self._rooms:
            if len(self._rooms) >= self._max_cached_rooms:
                self._rooms.pop(next(iter(self._rooms)))
            self._rooms[room_id] = load_room_cache(self.room_cache_dir / f"{room_id}.npz", self.rir_config)
        return self._rooms[room_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._train_item(index) if self.split == "train" else self._eval_item(index)

    # --- evaluation -------------------------------------------------------
    def _eval_item(self, index: int) -> dict[str, Any]:
        record, sha = self.items[index]
        sample = read_eval_sample(self.eval_dir, record, sha)
        arrays = sample.arrays
        grid = grid_from_record(record["grid"])
        k = self.k_eval
        waveforms = arrays["rir_waveforms"].astype(np.float32).copy()
        valid = arrays["rir_valid"].astype(np.uint8).copy()
        positions = arrays["rir_positions_scene_m"].astype(np.float64)
        waveforms[k:] = 0
        valid[k:] = 0
        uvh = to_grid_metres(grid, positions.reshape(-1, 3)).reshape(-1, 2, 3)
        uvh[valid == 0] = 0
        item = self._item(arrays, waveforms, valid, uvh, k, record["coverage"], record["coverage_bin"],
                          record["occlusion_type"], record["room_id"], record["room_group"], record["sample_id"],
                          record["grid"], 0, False)
        item["k_requested"] = k
        return item

    # --- training ---------------------------------------------------------
    def _train_item(self, index: int) -> dict[str, Any]:
        room_id, slot = self.items[index]
        cache = self.room(room_id)
        key = f"train|seed={self.seed}|epoch={self.epoch}|room={room_id}|slot={slot}"
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
        k = k_requested = int(rng.integers(self.k_range[0], self.k_range[1] + 1))
        choices = self.feasible[room_id]
        best: tuple[Any, np.ndarray] | None = None
        for _ in range(self.max_redraws):
            kind, bin_name = choices[int(rng.integers(len(choices)))]
            scan_seed = int(rng.integers(2**31))
            try:
                scan = build_scan_sample(cache.geometry, room_id, kind, scan_seed, bin_name, shift_m=self.shift_m,
                                         wall_dropout=self.wall_dropout, max_attempts=15)
            except (ScanNotApplicable, ScanNotFeasible, GeometryLabelError, FrameError):
                continue
            usable = np.flatnonzero(eligible_mask(cache, scan.grid, scan.observed.observed_free))
            if best is None or len(usable) > len(best[1]):
                best = (scan, usable)
            if len(usable) >= k:
                break
        if best is None or len(best[1]) == 0:
            raise DatasetError(f"{room_id}: no scan with a usable pair in {self.max_redraws} draws")
        scan, usable = best
        k = min(k, len(usable))  # plan 1.4: resample the mask; if still short, use what is there
        candidates = cache.candidates()
        free = scan.observed.observed_free
        try:
            bundle = build_bundle([candidates[i] for i in usable], scan.grid, free, cache.loader(self.rir_config),
                                  room_id=room_id, selection_key=key, seed=self.seed, k=k, config=self.rir_config)
            if self.augment:
                bundle = augment_bundle(bundle, scan.grid, free, split="train", seed=self.seed, sample_key=key,
                                        config=self.augment_config, sample_rate_hz=self.rir_config.sample_rate_hz)
        except AcousticError as error:
            raise DatasetError(f"{room_id}: {error}") from error
        arrays = scan.contract_arrays()
        waveforms = np.zeros((K_MAX, self.rir_config.window_samples), dtype=np.float32)
        valid = np.zeros(K_MAX, dtype=np.uint8)
        uvh = np.zeros((K_MAX, 2, 3), dtype=np.float64)
        waveforms[:k] = bundle.waveforms
        valid[:k] = bundle.valid
        uvh[:k] = to_grid_metres(scan.grid, bundle.positions_scene_m.reshape(-1, 3)).reshape(-1, 2, 3)
        quarter_turns, flip = (int(rng.integers(4)), bool(rng.integers(2))) if self.augment else (0, False)
        if quarter_turns or flip:
            size = scan.grid.width * scan.grid.cell_size_m
            arrays, flat = spatial_transform(arrays, uvh[:k].reshape(-1, 3), quarter_turns, flip, size)
            uvh[:k] = flat.reshape(-1, 2, 3)
        item = self._item(arrays, waveforms, valid, uvh, k, scan.coverage, scan.coverage_bin, scan.occlusion_type,
                          room_id, cache.room_group, key, scan.grid.manifest_grid(), quarter_turns, flip)
        item["k_requested"] = k_requested
        return item

    @staticmethod
    def _item(arrays: dict[str, np.ndarray], waveforms: np.ndarray, valid: np.ndarray, uvh: np.ndarray, k: int,
              coverage: float, coverage_bin: str | None, kind: str, room_id: str, group: int, sample_id: str,
              grid: dict[str, Any], quarter_turns: int, flip: bool) -> dict[str, Any]:
        item: dict[str, Any] = {name: np.asarray(arrays[name], dtype=np.uint8) for name in GRID_ARRAYS}
        item.update({
            "rir": waveforms,
            "rir_valid": valid,
            "src_pos": uvh[:, 0].astype(np.float32),
            "mic_pos": uvh[:, 1].astype(np.float32),
            "k": int(k),
            "coverage": float(coverage),
            "coverage_bin": coverage_bin,
            "occlusion_type": kind,
            "room_id": room_id,
            "room_group": int(group),
            "sample_id": sample_id,
            "grid": grid,
            "spatial_transform": {"quarter_turns": int(quarter_turns), "flip_columns": bool(flip)},
        })
        return item


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack arrays into a batch; other values become lists."""
    batch: dict[str, Any] = {}
    for name in items[0]:
        values = [item[name] for item in items]
        batch[name] = np.stack(values) if isinstance(values[0], np.ndarray) else values
    return batch


def describe(item: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in item.items() if not isinstance(v, np.ndarray)}, default=str)
