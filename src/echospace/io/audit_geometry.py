"""Pure mesh checks for the D0 audit: up axis, floor slice, containment, line of sight.

Everything works in the mesh's own frame. ``up`` is an axis index (0, 1 or 2)
that the audit discovers from the data and passes in; the two remaining axes,
in increasing order, are the horizontal plane. Needs the ``audit`` extra
(trimesh, rtree, shapely).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import shapely
import trimesh
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import polygonize_full, unary_union

FLOOR_TOLERANCE_M = 1e-3
SLICE_HEIGHTS_M = (1.0, 1.1, 1.2)


def horizontal_axes(up: int) -> tuple[int, int]:
    first, second = (axis for axis in range(3) if axis != up)
    return first, second


def obj_vertex_bounds(obj_bytes: bytes) -> np.ndarray:
    """``[[min xyz], [max xyz]]`` of an OBJ's vertices without building a mesh."""
    rows = [line[2:] for line in obj_bytes.split(b"\n") if line.startswith(b"v ")]
    if not rows:
        raise ValueError("OBJ has no vertices")
    vertices = np.array(b" ".join(rows).split(), dtype=np.float64).reshape(len(rows), -1)[:, :3]
    return np.stack([vertices.min(axis=0), vertices.max(axis=0)])


def vote_up_axis(bounds_per_room: Sequence[np.ndarray]) -> dict[str, Any]:
    """Which axis is vertical, from a convention that must hold in every room.

    A box room looks the same along all three axes, so a single mesh cannot
    settle this. Across a dataset the vertical axis is the one whose minimum
    is the floor datum (zero) in every room, while horizontal minima are
    arbitrary. Returns the winner and the per-axis evidence.
    """
    bounds = np.asarray(bounds_per_room, dtype=np.float64)
    at_zero = (np.abs(bounds[:, 0, :]) <= FLOOR_TOLERANCE_M).sum(axis=0)
    extents = bounds[:, 1, :] - bounds[:, 0, :]
    up = int(np.argmax(at_zero))
    return {
        "up_axis": up,
        "rooms": int(len(bounds)),
        "rooms_with_min_at_zero_per_axis": [int(v) for v in at_zero],
        "extent_min_per_axis": [float(v) for v in extents.min(axis=0)],
        "extent_median_per_axis": [float(v) for v in np.median(extents, axis=0)],
        "extent_max_per_axis": [float(v) for v in extents.max(axis=0)],
        "unanimous": bool(at_zero[up] == len(bounds) and sorted(at_zero)[-2] < len(bounds)),
    }


def geometry_signature(vertices: np.ndarray, decimals: int = 3) -> str:
    """Hash of the vertex set, invariant to translation, vertex order and duplicates.

    Rooms that reuse one geometry with different materials share a signature
    and must be counted as a single independent room.
    """
    points = np.asarray(vertices, dtype=np.float64)
    rounded = np.round(points - points.min(axis=0), decimals) + 0.0  # +0.0 folds -0.0 into 0.0
    unique = np.unique(rounded, axis=0)
    return hashlib.sha1(unique.astype("<f8").tobytes()).hexdigest()[:16]


def mesh_summary(mesh: trimesh.Trimesh, up: int) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    merged = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    areas = np.asarray(mesh.area_faces)
    return {
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "n_degenerate_faces": int((areas <= 1e-12).sum()),
        "bounds_min": [float(v) for v in bounds[0]],
        "bounds_max": [float(v) for v in bounds[1]],
        "extent": [float(v) for v in bounds[1] - bounds[0]],
        "floor": float(bounds[0, up]),
        "ceiling": float(bounds[1, up]),
        "floor_at_zero": bool(abs(bounds[0, up]) <= FLOOR_TOLERANCE_M),
        "watertight": bool(merged.is_watertight),
        "surface_area_m2": float(areas.sum()),
    }


@dataclass(frozen=True)
class FloorSlice:
    """Planar cut through the mesh at a height above the floor."""

    height_above_floor_m: float
    footprint: Polygon | MultiPolygon | None  # outline with interior loops filled
    n_parts: int
    area_m2: float
    largest_part_fraction: float
    n_interior_loops: int
    interior_loop_area_m2: float
    open_length_fraction: float

    @property
    def single_closed_interior(self) -> bool:
        return self.footprint is not None and self.largest_part_fraction >= 0.98 and self.area_m2 >= 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "height_above_floor_m": self.height_above_floor_m,
            "n_parts": self.n_parts,
            "area_m2": self.area_m2,
            "largest_part_fraction": self.largest_part_fraction,
            "n_interior_loops": self.n_interior_loops,
            "interior_loop_area_m2": self.interior_loop_area_m2,
            "open_length_fraction": self.open_length_fraction,
            "single_closed_interior": self.single_closed_interior,
        }


def floor_slice(mesh: trimesh.Trimesh, up: int, height_above_floor_m: float) -> FloorSlice:
    """Cut the mesh horizontally and describe the loops the cut produces.

    ``n_parts`` counts disjoint closed regions (one for a single room).
    Interior loops are closed outlines inside the room at that height:
    furniture, columns or partitions. ``open_length_fraction`` is the share of
    cut length that closes no loop, i.e. how leaky the outline is.
    """
    empty = FloorSlice(height_above_floor_m, None, 0, 0.0, 0.0, 0, 0.0, 1.0)
    normal = np.zeros(3)
    normal[up] = 1.0
    origin = np.zeros(3)
    origin[up] = float(mesh.bounds[0, up]) + height_above_floor_m
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return empty
    axes = list(horizontal_axes(up))
    lines = [LineString(np.asarray(path)[:, axes]) for path in section.discrete if len(path) >= 2]
    if not lines:
        return empty
    total_length = float(sum(line.length for line in lines))
    polygons, dangles, cuts, invalid = polygonize_full(unary_union(lines))
    faces = [face for face in polygons.geoms if face.area > 1e-6]
    if not faces:
        return empty
    footprint = unary_union(faces)
    parts = list(footprint.geoms) if isinstance(footprint, MultiPolygon) else [footprint]
    parts = [Polygon(part.exterior) for part in parts]  # fill interior loops
    footprint = unary_union(parts)
    areas = sorted((part.area for part in parts), reverse=True)
    face_areas = sorted((face.area for face in faces), reverse=True)
    open_length = float(dangles.length + cuts.length + invalid.length)
    return FloorSlice(
        height_above_floor_m=height_above_floor_m,
        footprint=footprint,
        n_parts=len(parts),
        area_m2=float(sum(areas)),
        largest_part_fraction=float(areas[0] / sum(areas)),
        n_interior_loops=len(faces) - len(parts),
        interior_loop_area_m2=float(sum(face_areas[len(parts) :])),
        open_length_fraction=open_length / total_length if total_length else 1.0,
    )


def footprint_vertices(footprint: Polygon | MultiPolygon) -> np.ndarray:
    """Exterior vertices of every part, as an Nx2 array in the horizontal plane."""
    parts = footprint.geoms if isinstance(footprint, MultiPolygon) else [footprint]
    return np.vstack([np.asarray(part.exterior.coords) for part in parts])


def containment(
    mesh: trimesh.Trimesh,
    footprint: Polygon | MultiPolygon | None,
    points: np.ndarray,
    up: int,
    documented_clearance_m: float = 0.5,
) -> dict[str, Any]:
    """Do positions fall inside the room, and how far from the nearest surface?

    These meshes are not watertight, so inside/outside is tested in two
    independent parts: horizontally against the floor-slice outline and
    vertically between floor and ceiling. An axis swap or frame mismatch
    fails one of them for most points.
    """
    points = np.asarray(points, dtype=np.float64)
    axes = list(horizontal_axes(up))
    bounds = np.asarray(mesh.bounds)
    in_bounds = ((points >= bounds[0] - 1e-6) & (points <= bounds[1] + 1e-6)).all(axis=1)
    in_height = (points[:, up] > bounds[0, up]) & (points[:, up] < bounds[1, up])
    if footprint is not None:
        in_footprint = shapely.contains_xy(footprint, points[:, axes[0]], points[:, axes[1]])
    else:
        in_footprint = np.zeros(len(points), dtype=bool)
    _, distance, _ = trimesh.proximity.closest_point(mesh, points)
    return {
        "n_points": int(len(points)),
        "in_bounds_fraction": float(in_bounds.mean()),
        "in_height_fraction": float(in_height.mean()),
        "in_footprint_fraction": float(in_footprint.mean()) if footprint is not None else None,
        "inside_fraction": float((in_height & in_footprint).mean()) if footprint is not None else None,
        "surface_distance_min_m": float(distance.min()),
        "surface_distance_p05_m": float(np.quantile(distance, 0.05)),
        "surface_distance_median_m": float(np.median(distance)),
        "documented_clearance_m": documented_clearance_m,
        "meets_documented_clearance_fraction": float((distance >= documented_clearance_m - 1e-6).mean()),
    }


def line_of_sight_blocked(mesh: trimesh.Trimesh, sources: np.ndarray, receivers: np.ndarray) -> np.ndarray:
    """True where a mesh surface lies strictly between a source and its receiver.

    Uses geometry and coordinates only, never the audio, so it can explain a
    late RIR peak without being fitted to it.
    """
    sources = np.asarray(sources, dtype=np.float64).reshape(-1, 3)
    receivers = np.asarray(receivers, dtype=np.float64).reshape(-1, 3)
    delta = receivers - sources
    length = np.linalg.norm(delta, axis=1)
    blocked = np.zeros(len(sources), dtype=bool)
    valid = length > 1e-9
    if not valid.any():
        return blocked
    rows = np.flatnonzero(valid)
    locations, ray_index, _ = mesh.ray.intersects_location(sources[rows], delta[rows] / length[rows, None], multiple_hits=True)
    if len(locations) == 0:
        return blocked
    hit_distance = np.linalg.norm(locations - sources[rows][ray_index], axis=1)
    between = (hit_distance > 1e-4) & (hit_distance < length[rows][ray_index] - 1e-4)
    blocked[rows[np.unique(ray_index[between])]] = True
    return blocked
