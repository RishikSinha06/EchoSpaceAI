"""Grid frame from the observation only (plan section 1.2, the no-leakage rule).

The 64 x 64 window must be placed with information available at test time.
``observed_frame`` therefore takes a ``Scan`` and nothing else from the room:

- centre: the centroid of the observed free space, moved to the nearest
  observed-free point when the centroid falls outside it (an L-shaped view);
- heading: the dominant direction of the observed walls, folded to 90 degrees
  so walls line up with the grid axes; never the mesh axes;
- optional training shift: a uniform random offset of up to ``shift_m`` per
  axis, redrawn until the anchor stays in observed free space.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import Point
from shapely.ops import nearest_points

from echospace.geometry.frames import GridFrame

from .occlusion import RAYS_PER_TURN, Scan

MAX_NEIGHBOUR_GAP_M = 0.3  # consecutive hits closer than this lie on one surface
ANCHOR_INSET_M = 0.1  # keep a snapped anchor off the edge of the observed region
MAX_SHIFT_DRAWS = 50


class FrameError(ValueError):
    """The observation is too empty to place a grid."""


def dominant_wall_heading(scan: Scan) -> float:
    """Dominant direction of observed surfaces, in radians within [-pi/4, pi/4).

    Joins consecutive hits of the same fan that are less than
    ``MAX_NEIGHBOUR_GAP_M`` apart into short surface pieces and takes the
    length-weighted circular mean of 4 x their angle, so 0 and 90 degree walls
    vote together. Returns 0 when no two hits are neighbours.
    """
    total_sin = total_cos = 0.0
    for index, scanner in enumerate(scan.scanners):
        mine = scan.hit_scanner == index
        rays, points = scan.hit_ray[mine], scan.hit_points[mine]
        if len(rays) < 2:
            continue
        order = np.argsort(rays)
        rays, points = rays[order], points[order]
        pairs = [(k, k + 1) for k in range(len(rays) - 1) if rays[k + 1] == rays[k] + 1]
        if scanner.heading_rad is None and len(rays) > 2 and rays[0] == 0 and rays[-1] == RAYS_PER_TURN - 1:
            pairs.append((len(rays) - 1, 0))  # wrap around a full turn
        for k, j in pairs:
            step = points[j] - points[k]
            length = float(np.hypot(*step))
            if 0 < length < MAX_NEIGHBOUR_GAP_M:
                angle = math.atan2(step[1], step[0])
                total_sin += length * math.sin(4 * angle)
                total_cos += length * math.cos(4 * angle)
    if total_sin == 0 and total_cos == 0:
        return 0.0
    heading = math.atan2(total_sin, total_cos) / 4
    return (heading + math.pi / 4) % (math.pi / 2) - math.pi / 4


def observed_anchor(scan: Scan) -> np.ndarray:
    """Centroid of the observed free space, snapped into it if needed."""
    visible = scan.visible
    if visible.is_empty or visible.area <= 0:
        raise FrameError("scan observed no free space")
    centroid = visible.centroid
    if visible.contains(centroid):
        return np.array([centroid.x, centroid.y])
    inner = visible.buffer(-ANCHOR_INSET_M)
    target = inner if not inner.is_empty else visible
    snapped = nearest_points(target, centroid)[0]
    return np.array([snapped.x, snapped.y])


def observed_frame(
    scan: Scan,
    scan_height_scene_m: float,
    rng: np.random.Generator | None = None,
    shift_m: float = 0.0,
    width: int = 64,
    height: int = 64,
    cell_size_m: float = 0.20,
) -> GridFrame:
    """P3 grid frame placed from the scan alone."""
    anchor = observed_anchor(scan)
    if shift_m > 0:
        if rng is None:
            raise ValueError("a shift needs an rng")
        for _ in range(MAX_SHIFT_DRAWS):
            moved = anchor + rng.uniform(-shift_m, shift_m, size=2)
            if shapely.contains_xy(scan.visible, moved[0], moved[1]):
                anchor = moved
                break
    if not scan.visible.covers(Point(anchor)):
        raise FrameError("anchor is not in observed free space")
    return GridFrame(
        (float(anchor[0]), float(scan_height_scene_m), float(anchor[1])),
        dominant_wall_heading(scan),
        width=width,
        height=height,
        cell_size_m=cell_size_m,
    )
