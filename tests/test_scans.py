"""Module 4 scan simulator on synthetic rooms: rays, occlusion types, frame, contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("shapely")
trimesh = pytest.importorskip("trimesh")

import shapely  # noqa: E402
from shapely import affinity  # noqa: E402
from shapely.geometry import LineString, Polygon  # noqa: E402

from echospace.contract_v0 import validate_sample  # noqa: E402
from echospace.io.audit_geometry import FloorSlice  # noqa: E402
from echospace.scans import (  # noqa: E402
    COVERAGE_BINS,
    RoomGeometry,
    ScanNotApplicable,
    ScanNotFeasible,
    build_scan_sample,
    cast_rays,
    dominant_wall_heading,
    generate_scan,
    l_wing_candidates,
    observed_frame,
)
from echospace.scans.occlusion import doorway_scan, l_wing_scan, missing_wall_scan, viewpoint_scan  # noqa: E402

SHOEBOX = Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])
L_ROOM = Polygon([(0, 0), (8, 0), (8, 3), (3, 3), (3, 7), (0, 7)])  # 3 m wide wing up the left side


def room_from_outline(outline: Polygon, loops: tuple[Polygon, ...] = ()) -> RoomGeometry:
    """A slice as floor_slice would return it, in source x/y, with furniture loops."""
    lines = [np.asarray(outline.exterior.coords)] + [np.asarray(loop.exterior.coords) for loop in loops]
    cut = FloorSlice(1.1, Polygon(outline.exterior), 1, outline.area, 1.0, len(loops), sum(l.area for l in loops), 0.0, tuple(lines))
    return RoomGeometry.from_slice(cut)


def box(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def angle_gap_deg(a: float, b: float) -> float:
    """Smallest difference between two directions taken modulo 90 degrees."""
    gap = (math.degrees(a - b) + 45) % 90 - 45
    return abs(gap)


# --- rays and room -----------------------------------------------------------

def test_rays_stop_at_the_nearest_segment_and_respect_range() -> None:
    room = room_from_outline(box(0, 0, 4, 4), (box(2.5, 1.5, 3.0, 2.5),))  # scene: z = -y
    origin = np.array([1.0, -2.0])
    fan = cast_rays(room.segments, origin, np.array([0.0, math.pi]), max_range_m=10.0)
    assert fan.hit.all()
    assert fan.distances == pytest.approx([1.5, 1.0])  # furniture at x = 2.5, wall at x = 0
    short = cast_rays(room.segments, origin, np.array([0.0]), max_range_m=1.0)
    assert not short.hit[0] and short.distances[0] == pytest.approx(1.0)


def test_room_geometry_separates_footprint_free_region_and_walls() -> None:
    room = room_from_outline(SHOEBOX, (box(2, 1, 3, 2),))
    assert room.footprint.area == pytest.approx(24.0)
    assert room.free_region.area == pytest.approx(23.0)  # the table is not standing room
    assert len(room.walls) == 4 and room.wall_length_m == pytest.approx(20.0)
    assert room.is_convex and not room_from_outline(L_ROOM).is_convex
    assert room.footprint.bounds[1] < 0  # scene Z = -source y


def test_from_mesh_matches_the_outline_route() -> None:
    mesh = trimesh.creation.box(extents=(6.0, 4.0, 2.5))
    mesh.apply_translation([3.0, 2.0, 1.25])
    room = RoomGeometry.from_mesh(mesh)
    assert room.footprint.area == pytest.approx(24.0)
    assert room.scan_height_scene_m == pytest.approx(1.1)


# --- occlusion types ---------------------------------------------------------

def test_viewpoint_scan_of_a_small_empty_room_sees_every_wall() -> None:
    room = room_from_outline(box(0, 0, 4, 3))  # farthest corner is 5 m away, within range
    scan = viewpoint_scan(room, np.random.default_rng(0))
    assert scan.coverage > 0.95 and scan.coverage_bin == "high"
    assert room.footprint.buffer(1e-6).contains(shapely.multipoints(scan.hit_points))
    assert scan.visible.within(room.free_region.buffer(1e-6))


def test_furniture_blocks_rays_and_is_recorded_as_a_measured_surface() -> None:
    blocker = box(2.0, 0.5, 2.4, 3.5)  # a tall cabinet across most of the room
    room = room_from_outline(box(0, 0, 6, 4), (blocker,))
    left = np.array([1.0, -2.0])
    fan = cast_rays(room.segments, left, np.linspace(-0.3, 0.3, 31), 8.0)
    assert np.allclose(fan.distances, 1.0 / np.cos(np.linspace(-0.3, 0.3, 31)), atol=1e-9)  # all stop at x = 2
    scan = generate_scan(room, "viewpoint", np.random.default_rng(3))
    on_blocker = shapely.distance(affinity.scale(blocker, 1, -1, origin=(0, 0)).exterior, shapely.points(scan.hit_points)) < 1e-6
    assert on_blocker.any() or scan.coverage < 1.0


def test_missing_wall_scan_hides_a_wall_band_completely() -> None:
    room = room_from_outline(SHOEBOX)
    scan = missing_wall_scan(room, np.random.default_rng(1))
    assert scan.hidden is not None and scan.hidden.area > 0
    assert not shapely.contains_xy(scan.hidden, scan.hit_points[:, 0], scan.hit_points[:, 1]).any()
    assert scan.visible.intersection(scan.hidden).area < 1e-9
    assert scan.coverage < 0.95


def test_doorway_scan_looks_into_the_room_within_its_field_of_view() -> None:
    room = room_from_outline(SHOEBOX)
    scan = doorway_scan(room, np.random.default_rng(2))
    (scanner,) = scan.scanners
    assert scanner.heading_rad is not None and math.degrees(scanner.fov_rad) == pytest.approx(120.0)
    bearings = np.arctan2(*(scan.hit_points - scanner.origin).T[::-1])
    offsets = (bearings - scanner.heading_rad + math.pi) % (2 * math.pi) - math.pi
    assert np.abs(offsets).max() <= math.radians(60) + 1e-6
    assert room.free_region.contains(shapely.Point(scanner.origin))


def test_l_wing_scan_hides_the_concave_wing_and_needs_a_concave_room() -> None:
    room = room_from_outline(L_ROOM)
    wings = l_wing_candidates(room)
    assert wings and all(0.1 <= w.area / room.footprint.area <= 0.5 for w in wings)
    scan = l_wing_scan(room, np.random.default_rng(4))
    assert scan.visible.intersection(scan.hidden).area < 1e-9
    assert not shapely.contains_xy(scan.hidden, scan.hit_points[:, 0], scan.hit_points[:, 1]).any()
    with pytest.raises(ScanNotApplicable):
        l_wing_scan(room_from_outline(SHOEBOX), np.random.default_rng(0))


def test_coverage_bins_are_honoured_or_refused() -> None:
    room = room_from_outline(L_ROOM)
    scan = generate_scan(room, "l_wing", np.random.default_rng(5), "mid")
    low, high = COVERAGE_BINS["mid"]
    assert low <= scan.coverage < high
    with pytest.raises(ScanNotFeasible):  # a 4 x 3 m room is always seen almost whole
        generate_scan(room_from_outline(box(0, 0, 4, 3)), "viewpoint", np.random.default_rng(0), "low", max_attempts=5)


def test_generate_scan_never_returns_a_scan_below_the_lowest_bin(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: with no bin requested, None == None used to accept a scan with
    # 0.11 coverage (Apartments_idx_37 doorway).
    from dataclasses import replace

    from echospace.scans import occlusion

    room = room_from_outline(SHOEBOX)
    real = viewpoint_scan(room, np.random.default_rng(0))
    coverages = iter([0.05, 0.10, 0.5])
    monkeypatch.setitem(occlusion.GENERATORS, "viewpoint", lambda r, g: replace(real, coverage=next(coverages)))
    assert generate_scan(room, "viewpoint", np.random.default_rng(0)).coverage == 0.5
    monkeypatch.setitem(occlusion.GENERATORS, "viewpoint", lambda r, g: replace(real, coverage=0.05))
    with pytest.raises(ScanNotFeasible):
        generate_scan(room, "viewpoint", np.random.default_rng(0), max_attempts=3)


# --- grid frame (no leakage) -----------------------------------------------------

def test_frame_heading_follows_observed_walls_not_mesh_axes() -> None:
    for degrees in (0.0, 17.0, 30.0, -40.0):
        turned = affinity.rotate(SHOEBOX, degrees, origin=(0, 0))
        scan = viewpoint_scan(room_from_outline(turned), np.random.default_rng(6))
        # source y flips to scene z, so a source rotation of +a is -a in the scene
        assert angle_gap_deg(dominant_wall_heading(scan), math.radians(-degrees)) < 1.0


def test_frame_moves_with_the_room_and_stays_in_observed_free_space() -> None:
    here = room_from_outline(L_ROOM)
    there = room_from_outline(affinity.translate(L_ROOM, 12.0, -5.0))
    scan_here = viewpoint_scan(here, np.random.default_rng(7))
    scan_there = viewpoint_scan(there, np.random.default_rng(7))
    frame_here = observed_frame(scan_here, 1.1)
    frame_there = observed_frame(scan_there, 1.1)
    assert np.allclose(np.subtract(frame_there.anchor_scene_m, frame_here.anchor_scene_m), [12.0, 0.0, 5.0], atol=1e-6)
    assert frame_here.heading_rad == pytest.approx(frame_there.heading_rad)
    anchor = shapely.Point(frame_here.anchor_scene_m[0], frame_here.anchor_scene_m[2])
    assert scan_here.visible.covers(anchor)
    shifted = observed_frame(scan_here, 1.1, np.random.default_rng(0), shift_m=1.0)
    assert scan_here.visible.covers(shapely.Point(shifted.anchor_scene_m[0], shifted.anchor_scene_m[2]))


def test_frame_uses_only_the_scan() -> None:
    # Two rooms that differ only beyond a wall the scanner cannot see through
    # give the same scan and therefore the same frame.
    closed = room_from_outline(SHOEBOX)
    with_annex = room_from_outline(SHOEBOX, (box(7, 0, 9, 4),))  # a separate loop outside the room
    a = viewpoint_scan(closed, np.random.default_rng(8))
    b = viewpoint_scan(with_annex, np.random.default_rng(8))
    assert np.allclose(a.hit_points, b.hit_points)
    fa, fb = observed_frame(a, 1.1), observed_frame(b, 1.1)
    assert fa.anchor_scene_m == fb.anchor_scene_m and fa.heading_rad == fb.heading_rad


# --- sample and contract -------------------------------------------------------

def _record(sample) -> dict:  # type: ignore[no-untyped-def]
    return {
        "schema_version": "0.1.0",
        "sample_id": f"{sample.room_id}-{sample.occlusion_type}-{sample.seed}",
        "room_id": sample.room_id,
        "dataset": "synthetic",
        "geometry_id": sample.room_id,
        "rir_ids": [],
        "rir_sample_rate_hz": 22050,
        "sample_path": "sample.npz",
        "split": "unassigned",
        "frame": {"description": "test", "source_units_to_meters": 1.0,
                  "source_to_scene": [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]},
        "grid": sample.grid.manifest_grid(),
    }


@pytest.mark.parametrize("occlusion_type", ["viewpoint", "missing_wall", "doorway", "l_wing"])
def test_built_samples_pass_the_contract_and_never_observe_outside_the_room(occlusion_type: str) -> None:
    room = room_from_outline(L_ROOM, (box(5.5, 0.8, 6.5, 1.8),))
    sample = build_scan_sample(room, "synthetic_L", occlusion_type, seed=11)
    arrays = sample.contract_arrays()
    validate_sample(_record(sample), arrays)
    free, target = arrays["observed_free"].astype(bool), arrays["target_occupancy"].astype(bool)
    assert free.any() and arrays["observed_wall"].any()
    assert not (free & ~target).any()  # observed free space is always inside the room label
    if occlusion_type in ("missing_wall", "l_wing"):
        x, z = sample.grid.cell_centres_xz()
        hidden_cells = shapely.contains_xy(sample.scan.hidden, x, z)
        assert not (free & hidden_cells).any()


def test_samples_are_deterministic_per_seed() -> None:
    room = room_from_outline(L_ROOM)
    a = build_scan_sample(room, "r", "viewpoint", seed=3).contract_arrays()
    b = build_scan_sample(room, "r", "viewpoint", seed=3).contract_arrays()
    c = build_scan_sample(room, "r", "viewpoint", seed=4).contract_arrays()
    assert all(np.array_equal(a[k], b[k]) for k in a)
    # an empty room can be observed whole from anywhere, so compare the rays, not the cells
    assert a["observed_points_scene_m"].shape != c["observed_points_scene_m"].shape or not np.allclose(
        a["observed_points_scene_m"], c["observed_points_scene_m"]
    )


def test_training_shift_and_wall_dropout_keep_the_contract() -> None:
    room = room_from_outline(SHOEBOX)
    sample = build_scan_sample(room, "r", "viewpoint", seed=0, shift_m=1.0, wall_dropout=0.2)
    validate_sample(_record(sample), sample.contract_arrays())
    clean = build_scan_sample(room, "r", "viewpoint", seed=0, shift_m=1.0)
    assert sample.observed.observed_wall.sum() < clean.observed.observed_wall.sum()


def test_wall_cells_lie_on_measured_hits_only() -> None:
    room = room_from_outline(SHOEBOX)
    sample = build_scan_sample(room, "r", "doorway", seed=1)
    cells = np.floor(sample.grid.scene_to_cell(sample.observed.observed_points_scene_m)).astype(int)
    marked = np.zeros_like(sample.observed.observed_wall)
    marked[cells[:, 1], cells[:, 0]] = True
    assert np.array_equal(marked, sample.observed.observed_wall)
    assert LineString(sample.labels.footprint_scene_xz.exterior.coords).length == pytest.approx(20.0)
