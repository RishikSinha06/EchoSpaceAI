"""One room's 1.1 m slice in the scene floor plane, as the scan simulator sees it.

Scene axes follow the audited AcousticRooms transform: source ``(x, y, z)``
maps to scene ``(x, z, -y)``, so the source floor plane ``(x, y)`` becomes the
scene floor plane ``(X, Z) = (x, -y)``. Everything here works in scene X/Z
metres; the height of the slice is kept separately as scene Y.

The complete room is ground truth. Only the simulator may look at it, to decide
what a scanner would see; the grid frame is computed from that observation alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
import trimesh
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize

from echospace.io.audit_geometry import FloorSlice, floor_slice

WALL_SIMPLIFY_M = 0.05  # straighten collinear mesh pieces before listing walls
CONVEX_AREA_RATIO = 0.9  # footprint / convex hull below this counts as non-convex


class RoomGeometryError(ValueError):
    """The slice cannot support a scan simulation."""


@dataclass(frozen=True)
class RoomGeometry:
    """A single room at scanner height, in scene X/Z metres.

    ``footprint`` is the outer outline with furniture loops filled (the same
    region P3 labels). ``free_region`` is the open floor at slice height that a
    scanner can stand in: the largest face of the slice linework, which leaves
    out furniture and columns. ``segments`` is every cut segment (walls and
    furniture) as an ``S x 2 x 2`` array, what rays can hit. ``walls`` are the
    straight edges of the simplified outer outline.
    """

    footprint: Polygon
    free_region: Polygon
    segments: np.ndarray
    walls: tuple[tuple[np.ndarray, np.ndarray], ...]
    scan_height_scene_m: float

    @property
    def wall_length_m(self) -> float:
        return float(self.footprint.exterior.length)

    @property
    def is_convex(self) -> bool:
        return self.footprint.area / self.footprint.convex_hull.area >= CONVEX_AREA_RATIO

    @classmethod
    def from_slice(cls, cut: FloorSlice, floor_height_m: float = 0.0) -> RoomGeometry:
        """Build from a ``floor_slice`` of a Z-up AcousticRooms mesh."""
        if cut.footprint is None or not cut.single_closed_interior or cut.n_parts != 1:
            raise RoomGeometryError("slice is not one closed room interior")
        footprint = _to_scene(cut.footprint)
        lines = [_to_scene(LineString(line)) for line in cut.outlines if len(line) >= 2]
        faces = [face for face in polygonize(MultiLineString(lines)) if face.area > 1e-6]
        if not faces:
            raise RoomGeometryError("slice linework encloses no face")
        free_region = max(faces, key=lambda face: face.area)
        segments = np.concatenate([_segments(np.asarray(line.coords)) for line in lines])
        ring = np.asarray(footprint.exterior.simplify(WALL_SIMPLIFY_M).coords)
        walls = tuple((ring[i].copy(), ring[i + 1].copy()) for i in range(len(ring) - 1))
        return cls(footprint, free_region, segments, walls, floor_height_m + cut.height_above_floor_m)

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, slice_height_above_floor_m: float = 1.1) -> RoomGeometry:
        """Slice a Z-up AcousticRooms mesh at scanner height."""
        cut = floor_slice(mesh, up=2, height_above_floor_m=slice_height_above_floor_m)
        return cls.from_slice(cut, floor_height_m=float(mesh.bounds[0, 2]))

    def sample_free_point(self, rng: np.random.Generator, clearance_m: float, region: Polygon | None = None) -> np.ndarray:
        """A uniform random point at least ``clearance_m`` inside the free region (and ``region``)."""
        area = self.free_region.buffer(-clearance_m)
        if region is not None:
            area = area.intersection(region)
        if area.is_empty:
            raise RoomGeometryError(f"no free space with {clearance_m} m clearance")
        low_x, low_z, high_x, high_z = area.bounds
        for _ in range(10_000):
            point = rng.uniform([low_x, low_z], [high_x, high_z])
            if shapely.contains_xy(area, point[0], point[1]):
                return point
        raise RoomGeometryError("could not sample a point in the free region")


def _to_scene(geometry):  # type: ignore[no-untyped-def]
    """Source floor plane (x, y) to scene floor plane (X, Z) = (x, -y)."""
    return affinity.scale(geometry, xfact=1.0, yfact=-1.0, origin=(0, 0))


def _segments(coords: np.ndarray) -> np.ndarray:
    return np.stack([coords[:-1, :2], coords[1:, :2]], axis=1)
