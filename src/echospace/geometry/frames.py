"""Explicit AcousticRooms source-frame and scanner-anchored grid transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# D0: AcousticRooms is metres, with Z up. Scene is X right, Y up, Z forward.
ACOUSTICROOMS_SOURCE_TO_SCENE = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def source_to_scene(points_source_m: np.ndarray) -> np.ndarray:
    """Apply the audited rigid transform to Nx3 source positions in metres."""
    points = np.asarray(points_source_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_source_m must be finite Nx3")
    return points @ ACOUSTICROOMS_SOURCE_TO_SCENE[:3, :3].T


@dataclass(frozen=True)
class GridFrame:
    """A grid fixed by a scanner pose, never by hidden mesh bounds.

    Heading is measured in the scene XZ plane from +X toward +Z. The anchor
    is the horizontal centre of the grid. Even grids place it between cells.
    """

    anchor_scene_m: tuple[float, float, float]
    heading_rad: float
    width: int = 64
    height: int = 64
    cell_size_m: float = 0.20

    def __post_init__(self) -> None:
        anchor = np.asarray(self.anchor_scene_m, dtype=np.float64)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError("anchor_scene_m must contain three finite values")
        if not math.isfinite(self.heading_rad):
            raise ValueError("heading_rad must be finite")
        if self.width <= 0 or self.height <= 0 or not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0:
            raise ValueError("grid dimensions and cell size must be positive")

    @property
    def axis_u_scene(self) -> np.ndarray:
        return np.array([math.cos(self.heading_rad), 0.0, math.sin(self.heading_rad)])

    @property
    def axis_v_scene(self) -> np.ndarray:
        return np.array([-math.sin(self.heading_rad), 0.0, math.cos(self.heading_rad)])

    @property
    def origin_scene_m(self) -> np.ndarray:
        return (np.asarray(self.anchor_scene_m) - self.width * self.cell_size_m / 2 * self.axis_u_scene
                - self.height * self.cell_size_m / 2 * self.axis_v_scene)

    def cell_centres_xz(self) -> tuple[np.ndarray, np.ndarray]:
        col, row = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        origin = self.origin_scene_m
        x = origin[0] + self.cell_size_m * (col * self.axis_u_scene[0] + row * self.axis_v_scene[0])
        z = origin[2] + self.cell_size_m * (col * self.axis_u_scene[2] + row * self.axis_v_scene[2])
        return x, z

    def scene_to_cell(self, points_scene_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_scene_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_scene_m must be finite Nx3")
        delta = points - self.origin_scene_m
        return np.column_stack((delta @ self.axis_u_scene, delta @ self.axis_v_scene)) / self.cell_size_m

    def manifest_grid(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "cell_size_m": self.cell_size_m,
            "origin_scene_m": self.origin_scene_m.tolist(),
            "axis_u_scene": self.axis_u_scene.tolist(),
            "axis_v_scene": self.axis_v_scene.tolist(),
        }
