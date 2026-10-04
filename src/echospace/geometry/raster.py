"""Slice one AcousticRooms mesh and rasterize its audited single-room outline.

The complete mesh supplies labels only. Grid placement is entirely determined
by the caller's scanner anchor and heading. No scan observations or RIRs are
fabricated here, and these labels are not training samples by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
import trimesh
from shapely import affinity
from shapely.geometry import Polygon

from echospace.io.audit_geometry import FloorSlice, floor_slice

from .frames import GridFrame


class GeometryLabelError(ValueError):
    """The mesh or chosen scan frame cannot yield a credible room label."""


@dataclass(frozen=True)
class RoomLabels:
    occupancy: np.ndarray
    boundary: np.ndarray
    valid_cells: np.ndarray
    grid: GridFrame
    diagnostics: dict[str, float | int | bool]
    footprint_scene_xz: Polygon


def _scene_polygon(cut: FloorSlice) -> Polygon:
    if not cut.single_closed_interior or cut.n_parts != 1 or cut.footprint is None:
        raise GeometryLabelError("slice is not one closed room interior")
    # Source XY plane maps to scene XZ by (x, y) -> (x, -y).
    footprint = affinity.scale(cut.footprint, xfact=1, yfact=-1, origin=(0, 0))
    if not isinstance(footprint, Polygon) or not footprint.is_valid or footprint.area <= 0:
        raise GeometryLabelError("invalid room footprint after frame transform")
    return footprint


def rasterize_footprint(footprint_xz: Polygon, grid: GridFrame, *, cut: FloorSlice | None = None) -> RoomLabels:
    """Return interior and structural-boundary masks on an explicit scan grid.

    Interior closed loops are already filled by ``floor_slice``. Wall pixels
    are interior cells whose square touches the outer footprint boundary.
    Reject partial coverage instead of silently clipping the ground truth.
    """
    if not footprint_xz.is_valid or footprint_xz.is_empty or footprint_xz.area < 1.0:
        raise GeometryLabelError("invalid or sub-1 m2 footprint")
    anchor = np.asarray(grid.anchor_scene_m)
    if not footprint_xz.covers(shapely.Point(float(anchor[0]), float(anchor[2]))):
        raise GeometryLabelError("scanner anchor is outside the room footprint")

    origin = grid.origin_scene_m
    corners = np.array([
        [origin[0], origin[2]],
        [origin[0] + grid.width * grid.cell_size_m * grid.axis_u_scene[0],
         origin[2] + grid.width * grid.cell_size_m * grid.axis_u_scene[2]],
        [origin[0] + grid.width * grid.cell_size_m * grid.axis_u_scene[0] + grid.height * grid.cell_size_m * grid.axis_v_scene[0],
         origin[2] + grid.width * grid.cell_size_m * grid.axis_u_scene[2] + grid.height * grid.cell_size_m * grid.axis_v_scene[2]],
        [origin[0] + grid.height * grid.cell_size_m * grid.axis_v_scene[0],
         origin[2] + grid.height * grid.cell_size_m * grid.axis_v_scene[2]],
    ])
    canvas = Polygon(corners)
    clipped_area = footprint_xz.difference(canvas).area
    if clipped_area > 1e-6:
        raise GeometryLabelError(f"room clips scanner grid by {clipped_area:.3f} m2")

    x, z = grid.cell_centres_xz()
    occupancy = shapely.contains_xy(footprint_xz, x, z)
    # A boundary cell is part of the interior and within half a cell diagonal
    # of the outer wall. This remains connected at diagonal edges.
    centres = shapely.points(x, z)
    distance = shapely.distance(centres, footprint_xz.exterior)
    boundary = occupancy & (distance <= grid.cell_size_m / np.sqrt(2) + 1e-9)
    valid = np.ones_like(occupancy, dtype=bool)
    if not occupancy.any() or not boundary.any():
        raise GeometryLabelError("footprint rasterized to an empty label")
    diagnostics: dict[str, float | int | bool] = {
        "footprint_area_m2": float(footprint_xz.area),
        "raster_area_m2": float(occupancy.sum() * grid.cell_size_m**2),
        "interior_cells": int(occupancy.sum()),
        "boundary_cells": int(boundary.sum()),
        "clipped_area_m2": float(clipped_area),
        "furniture_loops_filled": int(cut.n_interior_loops) if cut else 0,
        "slice_open_length_fraction": float(cut.open_length_fraction) if cut else 0.0,
    }
    return RoomLabels(occupancy, boundary, valid, grid, diagnostics, footprint_xz)


def label_acousticrooms_mesh(
    mesh: trimesh.Trimesh,
    grid: GridFrame,
    *,
    slice_height_above_floor_m: float = 1.1,
) -> RoomLabels:
    """Make labels from an audited AcousticRooms OBJ, whose vertical axis is Z."""
    if not 1.0 <= slice_height_above_floor_m <= 1.2:
        raise GeometryLabelError("slice height must be between 1.0 and 1.2 m")
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise GeometryLabelError("mesh has no valid faces or finite vertices")
    cut = floor_slice(mesh, up=2, height_above_floor_m=slice_height_above_floor_m)
    return rasterize_footprint(_scene_polygon(cut), grid, cut=cut)
