"""2D ray casting against slice linework, the core of the simulated LiDAR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

MIN_HIT_DISTANCE_M = 1e-6
MAX_PAIRS_PER_CHUNK = 2_000_000  # ray x segment pairs held in memory at once


@dataclass(frozen=True)
class RayFan:
    """Rays from one scanner. ``hit[i]`` is False when ray ``i`` reached ``max_range_m``."""

    origin: np.ndarray  # (2,)
    angles: np.ndarray  # (A,) radians, from +X toward +Z
    distances: np.ndarray  # (A,) metres to the hit, or max range
    hit: np.ndarray  # (A,) bool
    max_range_m: float

    @property
    def endpoints(self) -> np.ndarray:
        directions = np.column_stack((np.cos(self.angles), np.sin(self.angles)))
        return self.origin + self.distances[:, None] * directions

    @property
    def hit_points(self) -> np.ndarray:
        return self.endpoints[self.hit]

    def visible_polygon(self, full_circle: bool) -> Polygon:
        """Region swept by the rays: a star polygon, closed through the scanner for a partial fan."""
        ring = self.endpoints if full_circle else np.vstack([self.origin, self.endpoints])
        polygon = Polygon(ring)
        return polygon if polygon.is_valid else polygon.buffer(0)


def cast_rays(segments: np.ndarray, origin: np.ndarray, angles: np.ndarray, max_range_m: float) -> RayFan:
    """Nearest intersection of each ray with any segment, capped at ``max_range_m``.

    ``segments`` is ``S x 2 x 2``. Solves ``origin + t d = p + s (q - p)`` for
    every ray/segment pair at once and keeps the smallest ``t > 0`` with
    ``0 <= s <= 1``. A ray that grazes an endpoint counts as a hit.
    """
    origin = np.asarray(origin, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))  # (A, 2)
    starts = segments[:, 0, :]  # (S, 2)
    edges = segments[:, 1, :] - starts  # (S, 2)

    def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    offset = starts - origin  # (S, 2)
    numerator_t = cross(offset, edges)  # (S,)
    nearest = np.full(len(angles), np.inf)
    if len(segments):
        chunk = max(1, MAX_PAIRS_PER_CHUNK // len(segments))
        for begin in range(0, len(angles), chunk):
            d = directions[begin : begin + chunk]
            denominator = cross(d[:, None, :], edges[None, :, :])  # (a, S)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = numerator_t[None, :] / denominator
                s = cross(offset[None, :, :], d[:, None, :]) / denominator
            valid = (np.abs(denominator) > 1e-12) & (t > MIN_HIT_DISTANCE_M) & (s >= 0.0) & (s <= 1.0)
            nearest[begin : begin + chunk] = np.where(valid, t, np.inf).min(axis=1)
    hit = nearest <= max_range_m
    distances = np.where(hit, nearest, max_range_m)
    return RayFan(origin, angles, distances, hit, float(max_range_m))
