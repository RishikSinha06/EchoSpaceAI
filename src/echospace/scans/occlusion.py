"""Simulated partial scans: the plan's four occlusion types (plan section 1.3).

Each generator looks at the complete room, because it plays the part of the
physical world: it decides what a scanner standing there would see. The result
is a ``Scan``, which holds only what was observed. Nothing downstream of a
``Scan`` (grid frame, rasterisation) may look at the room again.

Parameters are fixed here, before any run on real rooms. Ranges marked "plan"
come from the implementation plan; the rest are this module's choices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import split, unary_union

from .raycast import RayFan, cast_rays
from .room import RoomGeometry

OCCLUSION_TYPES = ("viewpoint", "missing_wall", "doorway", "l_wing")

SCANNERS_PER_VIEWPOINT = (1, 3)  # plan: 1-3 scanner positions
MAX_RANGE_M = (5.0, 8.0)  # plan: max range 5-8 m
RAYS_PER_TURN = 360  # plan: 360 rays per scanner
SCANNER_CLEARANCE_M = 0.3  # keep the scanner off walls and furniture
MISSING_WALL_BAND_M = 1.0  # plan: hide the wall and cells within 1 m of it
DOORWAY_FOV_DEG = 120.0  # plan: 120 degree field of view
DOORWAY_SETBACK_M = 0.5  # scanner this far in from the wall it stands at
MIN_WALL_LENGTH_M = 1.0  # walls shorter than this are not chosen as a target
L_WING_AREA_FRACTION = (0.1, 0.5)  # the hidden wing's share of the footprint
COVERAGE_HIT_RADIUS_M = 0.15  # a wall point counts as seen within this of a hit

# Coverage = observed wall length / total wall length (plan). Nominal bins
# ~80 %, ~50 %, ~30 %; below the lowest edge a scan is too empty to use.
COVERAGE_BINS = {"high": (0.65, 1.0), "mid": (0.40, 0.65), "low": (0.15, 0.40)}


class ScanNotApplicable(ValueError):
    """This occlusion type cannot be generated for this room (e.g. L-wing on a convex room)."""


class ScanNotFeasible(RuntimeError):
    """No scan of this type landed in the requested coverage bin within the attempt budget."""


@dataclass(frozen=True)
class Scanner:
    origin: np.ndarray  # (2,) scene X/Z metres
    heading_rad: float | None  # None for a full turn
    fov_rad: float
    max_range_m: float


@dataclass(frozen=True)
class Scan:
    """What a partial scan observed, and nothing else.

    ``visible`` is the observed free space (rays passed through it);
    ``hit_points`` are measured surface points (walls or furniture).
    ``hidden`` records which region a missing-wall or L-wing scan removed, for
    diagnostics only; it must not be used to build model inputs.
    """

    occlusion_type: str
    scanners: tuple[Scanner, ...]
    visible: Polygon | MultiPolygon
    hit_points: np.ndarray  # (N, 2)
    hit_scanner: np.ndarray  # (N,) index into scanners, for each hit
    hit_ray: np.ndarray  # (N,) ray index within its fan, for each hit
    coverage: float
    hidden: Polygon | MultiPolygon | None = None

    @property
    def coverage_bin(self) -> str | None:
        return coverage_bin(self.coverage)


def coverage_bin(coverage: float) -> str | None:
    for name, (low, high) in COVERAGE_BINS.items():
        if low <= coverage < high or (high == 1.0 and coverage == 1.0):
            return name
    return None


def observed_wall_coverage(room: RoomGeometry, hit_points: np.ndarray) -> float:
    """Share of the outer wall length within ``COVERAGE_HIT_RADIUS_M`` of a measured hit."""
    if len(hit_points) == 0:
        return 0.0
    seen = shapely.union_all(shapely.buffer(shapely.points(hit_points), COVERAGE_HIT_RADIUS_M, quad_segs=4))
    return float(room.footprint.exterior.intersection(seen).length / room.wall_length_m)


def _fan(room: RoomGeometry, scanner: Scanner, rng: np.random.Generator) -> RayFan:
    if scanner.heading_rad is None:
        angles = (np.arange(RAYS_PER_TURN) + rng.uniform()) * (2 * math.pi / RAYS_PER_TURN)
    else:
        n = int(round(RAYS_PER_TURN * scanner.fov_rad / (2 * math.pi))) + 1
        angles = scanner.heading_rad + np.linspace(-scanner.fov_rad / 2, scanner.fov_rad / 2, n)
    return cast_rays(room.segments, scanner.origin, angles, scanner.max_range_m)


def _assemble(
    room: RoomGeometry,
    occlusion_type: str,
    scanners: list[Scanner],
    rng: np.random.Generator,
    hidden: Polygon | MultiPolygon | None = None,
) -> Scan:
    fans = [_fan(room, scanner, rng) for scanner in scanners]
    # The star polygon can cut a corner past an occluding edge; the simulator
    # knows the room, so it trims the swept area to the true free space.
    swept = unary_union([fan.visible_polygon(s.heading_rad is None) for fan, s in zip(fans, scanners)])
    visible = swept.intersection(room.free_region)
    points, owner, ray = [], [], []
    for index, fan in enumerate(fans):
        points.append(fan.hit_points)
        owner.append(np.full(int(fan.hit.sum()), index))
        ray.append(np.flatnonzero(fan.hit))
    hit_points, hit_scanner, hit_ray = np.concatenate(points), np.concatenate(owner), np.concatenate(ray)
    if hidden is not None and not hidden.is_empty:
        keep = ~shapely.contains_xy(hidden, hit_points[:, 0], hit_points[:, 1])
        keep &= shapely.distance(hidden, shapely.points(hit_points)) > 1e-9
        hit_points, hit_scanner, hit_ray = hit_points[keep], hit_scanner[keep], hit_ray[keep]
        visible = visible.difference(hidden)
    coverage = observed_wall_coverage(room, hit_points)
    return Scan(occlusion_type, tuple(scanners), visible, hit_points, hit_scanner, hit_ray, coverage, hidden)


def _full_turn_scanners(room: RoomGeometry, rng: np.random.Generator, region: Polygon | None = None) -> list[Scanner]:
    count = int(rng.integers(SCANNERS_PER_VIEWPOINT[0], SCANNERS_PER_VIEWPOINT[1] + 1))
    return [
        Scanner(room.sample_free_point(rng, SCANNER_CLEARANCE_M, region), None, 2 * math.pi, float(rng.uniform(*MAX_RANGE_M)))
        for _ in range(count)
    ]


def _pick_wall(room: RoomGeometry, rng: np.random.Generator, candidates: list[int] | None = None) -> int:
    indices = candidates if candidates is not None else list(range(len(room.walls)))
    lengths = np.array([np.linalg.norm(room.walls[i][1] - room.walls[i][0]) for i in indices])
    usable = lengths >= MIN_WALL_LENGTH_M
    if not usable.any():
        raise ScanNotApplicable("no wall long enough")
    weights = np.where(usable, lengths, 0.0)
    return indices[int(rng.choice(len(indices), p=weights / weights.sum()))]


def viewpoint_scan(room: RoomGeometry, rng: np.random.Generator) -> Scan:
    """1-3 full-turn scanners anywhere in the free region."""
    return _assemble(room, "viewpoint", _full_turn_scanners(room, rng), rng)


def missing_wall_scan(room: RoomGeometry, rng: np.random.Generator) -> Scan:
    """A viewpoint scan, then one observed wall and everything within 1 m of it hidden."""
    scanners = _full_turn_scanners(room, rng)
    fan_seed = int(rng.integers(2**63))  # same rays before and after hiding
    base = _assemble(room, "missing_wall", scanners, np.random.default_rng(fan_seed))
    seen = [
        i for i, (a, b) in enumerate(room.walls)
        if len(base.hit_points) and float(shapely.distance(LineString([a, b]), shapely.points(base.hit_points)).min()) <= COVERAGE_HIT_RADIUS_M
    ]
    if not seen:
        raise ScanNotApplicable("the scan saw no wall to remove")
    a, b = room.walls[_pick_wall(room, rng, seen)]
    hidden = LineString([a, b]).buffer(MISSING_WALL_BAND_M).intersection(room.footprint)
    return _assemble(room, "missing_wall", scanners, np.random.default_rng(fan_seed), hidden)


def doorway_scan(room: RoomGeometry, rng: np.random.Generator) -> Scan:
    """One scanner just inside a wall, looking into the room with a 120 degree field of view."""
    for _ in range(50):
        a, b = room.walls[_pick_wall(room, rng)]
        along = b - a
        normal = np.array([-along[1], along[0]]) / np.linalg.norm(along)
        foot = a + rng.uniform(0.2, 0.8) * along
        if not shapely.contains_xy(room.footprint, *(foot + 0.05 * normal)):
            normal = -normal
        origin = foot + DOORWAY_SETBACK_M * normal
        if shapely.contains_xy(room.free_region.buffer(-0.1), origin[0], origin[1]):
            scanner = Scanner(origin, math.atan2(normal[1], normal[0]), math.radians(DOORWAY_FOV_DEG), float(rng.uniform(*MAX_RANGE_M)))
            return _assemble(room, "doorway", [scanner], rng)
    raise ScanNotApplicable("no free spot just inside a wall")


def l_wing_candidates(room: RoomGeometry) -> list[Polygon]:
    """Concave wings: pieces cut off by extending an edge at a reflex corner.

    Only non-convex rooms have them. A piece qualifies when its area is within
    ``L_WING_AREA_FRACTION`` of the footprint.
    """
    if room.is_convex:
        return []
    shape = orient(Polygon(room.footprint.exterior.simplify(0.05)), sign=1.0)  # counter-clockwise
    ring = np.asarray(shape.exterior.coords)[:-1]
    reach = 4 * max(np.ptp(ring[:, 0]), np.ptp(ring[:, 1]))
    wings: list[Polygon] = []
    for i in range(len(ring)):
        before, here, after = ring[i - 1], ring[i], ring[(i + 1) % len(ring)]
        incoming, outgoing = here - before, after - here
        if incoming[0] * outgoing[1] - incoming[1] * outgoing[0] >= 0:
            continue  # convex corner
        for direction in (incoming, -outgoing):
            unit = direction / np.linalg.norm(direction)
            cutter = LineString([here - 1e-6 * unit, here + reach * unit])
            pieces = [p for p in split(room.footprint, cutter).geoms if isinstance(p, Polygon)]
            if len(pieces) < 2:
                continue
            smallest = min(pieces, key=lambda p: p.area)
            share = smallest.area / room.footprint.area
            if L_WING_AREA_FRACTION[0] <= share <= L_WING_AREA_FRACTION[1]:
                if not any(abs(w.area - smallest.area) < 1e-6 and w.equals_exact(smallest, 1e-6) for w in wings):
                    wings.append(smallest)
    return wings


def l_wing_scan(room: RoomGeometry, rng: np.random.Generator) -> Scan:
    """Scanners outside one concave wing, then that wing hidden entirely."""
    wings = l_wing_candidates(room)
    if not wings:
        raise ScanNotApplicable("room has no concave wing")
    wing = wings[int(rng.integers(len(wings)))]
    rest = room.footprint.difference(wing)
    scanners = _full_turn_scanners(room, rng, region=rest)
    return _assemble(room, "l_wing", scanners, rng, hidden=wing)


GENERATORS = {
    "viewpoint": viewpoint_scan,
    "missing_wall": missing_wall_scan,
    "doorway": doorway_scan,
    "l_wing": l_wing_scan,
}


def generate_scan(
    room: RoomGeometry,
    occlusion_type: str,
    rng: np.random.Generator,
    coverage_bin_name: str | None = None,
    max_attempts: int = 40,
) -> Scan:
    """One scan of the given type, optionally redrawn until it lands in a coverage bin.

    Raises ``ScanNotApplicable`` when the type cannot exist for this room and
    ``ScanNotFeasible`` when the bin is not reached within ``max_attempts``.
    """
    if occlusion_type not in GENERATORS:
        raise ValueError(f"unknown occlusion type {occlusion_type!r}")
    if coverage_bin_name is not None and coverage_bin_name not in COVERAGE_BINS:
        raise ValueError(f"unknown coverage bin {coverage_bin_name!r}")
    for _ in range(max_attempts):
        scan = GENERATORS[occlusion_type](room, rng)
        if coverage_bin_name is None and scan.coverage_bin is not None:
            return scan
        if scan.coverage_bin == coverage_bin_name:
            return scan
    raise ScanNotFeasible(f"{occlusion_type} did not reach coverage bin {coverage_bin_name} in {max_attempts} attempts")
