"""Partial scan + grid frame + P3 labels, as contract v0.1.0 arrays (minus RIRs).

``build_scan_sample`` is the P4 handoff: it simulates one scan, places the grid
from that scan alone, rasterises the observation, and asks P3 for the target on
the same grid. RIR selection is P6's job, so the RIR arrays come back empty.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import shapely

from echospace.geometry.frames import GridFrame
from echospace.geometry.raster import GeometryLabelError, RoomLabels, rasterize_footprint

from .frame import FrameError, observed_frame
from .occlusion import OCCLUSION_TYPES, Scan, generate_scan
from .room import RoomGeometry

TRAINING_SHIFT_M = 1.0  # plan: random shift of +-1 m during training
MAX_FRAME_ATTEMPTS = 10  # redraw scan/shift when the room clips the 12.8 m canvas


@dataclass(frozen=True)
class ObservedGrid:
    observed_cells: np.ndarray  # (H, W) bool
    observed_free: np.ndarray  # (H, W) bool
    observed_wall: np.ndarray  # (H, W) bool
    observed_points_scene_m: np.ndarray  # (N, 3) float


@dataclass(frozen=True)
class ScanSample:
    room_id: str
    occlusion_type: str
    seed: int
    scan: Scan
    grid: GridFrame
    observed: ObservedGrid
    labels: RoomLabels

    @property
    def coverage(self) -> float:
        return self.scan.coverage

    @property
    def coverage_bin(self) -> str | None:
        return self.scan.coverage_bin

    def contract_arrays(self) -> dict[str, np.ndarray]:
        """Sample NPZ arrays of contract v0.1.0, with zero RIR rows for P6 to fill."""
        return {
            "observed_points_scene_m": self.observed.observed_points_scene_m.astype(np.float32),
            "observed_cells": self.observed.observed_cells.astype(np.uint8),
            "observed_free": self.observed.observed_free.astype(np.uint8),
            "observed_wall": self.observed.observed_wall.astype(np.uint8),
            "valid_cells": self.labels.valid_cells.astype(np.uint8),
            "target_occupancy": self.labels.occupancy.astype(np.uint8),
            "target_boundary": self.labels.boundary.astype(np.uint8),
            "rir_waveforms": np.zeros((0, 1), dtype=np.float32),
            "rir_positions_scene_m": np.zeros((0, 2, 3), dtype=np.float32),
            "rir_valid": np.zeros((0,), dtype=np.uint8),
        }


def observe_on_grid(
    scan: Scan,
    grid: GridFrame,
    scan_height_scene_m: float,
    rng: np.random.Generator | None = None,
    wall_dropout: float = 0.0,
) -> ObservedGrid:
    """Rasterise a scan: hit cells are wall, swept cells are free, the rest unobserved.

    A cell is free when its centre lies in the observed free space and no hit
    falls in it. ``wall_dropout`` (plan: optional sparse noise) removes that
    share of wall cells at random, so a model cannot rely on perfect edges.
    """
    x, z = grid.cell_centres_xz()
    free = shapely.contains_xy(scan.visible, x, z)
    wall = np.zeros_like(free)
    points = np.column_stack(
        (scan.hit_points[:, 0], np.full(len(scan.hit_points), scan_height_scene_m), scan.hit_points[:, 1])
    )
    if len(points):
        cells = np.floor(grid.scene_to_cell(points)).astype(int)
        on_grid = (cells[:, 0] >= 0) & (cells[:, 0] < grid.width) & (cells[:, 1] >= 0) & (cells[:, 1] < grid.height)
        wall[cells[on_grid, 1], cells[on_grid, 0]] = True
        if wall_dropout > 0:
            if rng is None:
                raise ValueError("wall dropout needs an rng")
            wall &= rng.uniform(size=wall.shape) >= wall_dropout
    free &= ~wall
    return ObservedGrid(free | wall, free, wall, points)


def sample_rng(room_id: str, occlusion_type: str, seed: int) -> np.random.Generator:
    """Deterministic stream per (room, type, seed); independent of call order."""
    return np.random.default_rng([seed, zlib.crc32(room_id.encode()), OCCLUSION_TYPES.index(occlusion_type)])


def build_scan_sample(
    room: RoomGeometry,
    room_id: str,
    occlusion_type: str,
    seed: int,
    coverage_bin_name: str | None = None,
    shift_m: float = 0.0,
    wall_dropout: float = 0.0,
) -> ScanSample:
    """Simulate one scan and return it with its grid, observation and P3 target.

    ``shift_m=0`` gives the deterministic frame for fixed test masks; training
    uses ``TRAINING_SHIFT_M``. If the room clips the 12.8 m canvas around the
    observed anchor, the scan is redrawn; after ``MAX_FRAME_ATTEMPTS`` the last
    ``GeometryLabelError`` is raised.
    """
    rng = sample_rng(room_id, occlusion_type, seed)
    last_error: Exception | None = None
    for _ in range(MAX_FRAME_ATTEMPTS):
        scan = generate_scan(room, occlusion_type, rng, coverage_bin_name)
        try:
            grid = observed_frame(scan, room.scan_height_scene_m, rng, shift_m)
            labels = rasterize_footprint(room.footprint, grid)
        except (FrameError, GeometryLabelError) as error:
            last_error = error
            continue
        observed = observe_on_grid(scan, grid, room.scan_height_scene_m, rng, wall_dropout)
        return ScanSample(room_id, occlusion_type, seed, scan, grid, observed, labels)
    assert last_error is not None
    raise last_error
