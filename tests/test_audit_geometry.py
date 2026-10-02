"""Mesh checks on synthetic shoeboxes: up axis, slice, containment, line of sight."""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("shapely")
pytest.importorskip("rtree")

from echospace.io import audit_geometry as ag  # noqa: E402
from echospace.io.audit import CANVAS_12_8_M, anchored_coverage, fits_bbox  # noqa: E402

UP = 2
SIZE = np.array([6.0, 4.0, 2.5])


def _room(size: np.ndarray = SIZE, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> "trimesh.Trimesh":
    """Shoebox with its floor at z = 0 (after ``offset``), Z up."""
    box = trimesh.creation.box(extents=size)
    box.apply_translation(size / 2.0 + np.asarray(offset))
    return box


def _with_table(room: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Add a floor-to-1.5 m block between x = 2.5 and x = 3.5, across y = 1..3."""
    block = trimesh.creation.box(extents=[1.0, 2.0, 1.5])
    block.apply_translation([3.0, 2.0, 0.75])
    return trimesh.util.concatenate([room, block])


def test_obj_vertex_bounds_matches_mesh() -> None:
    room = _room(offset=(-3.0, 2.0, 0.0))
    bounds = ag.obj_vertex_bounds(room.export(file_type="obj").encode())
    assert np.allclose(bounds, room.bounds)
    with pytest.raises(ValueError):
        ag.obj_vertex_bounds(b"# empty\n")


def test_up_axis_is_voted_from_the_shared_floor_datum() -> None:
    rooms = [_room(offset=(dx, dy, 0.0)) for dx, dy in [(-3.0, 1.0), (2.0, -4.0), (0.0, 5.0), (7.0, 0.0)]]
    vote = ag.vote_up_axis([room.bounds for room in rooms])
    assert vote["up_axis"] == UP and vote["unanimous"]
    assert vote["rooms_with_min_at_zero_per_axis"] == [1, 1, 4]
    y_up = [room.bounds[:, [0, 2, 1]] for room in rooms]
    assert ag.vote_up_axis(y_up)["up_axis"] == 1


def test_geometry_signature_ignores_translation_and_order_only() -> None:
    room = _room()
    moved = _room(offset=(4.0, -2.0, 0.0))
    shuffled = room.vertices[::-1]
    assert ag.geometry_signature(room.vertices) == ag.geometry_signature(moved.vertices)
    assert ag.geometry_signature(room.vertices) == ag.geometry_signature(shuffled)
    assert ag.geometry_signature(room.vertices) != ag.geometry_signature(_room(SIZE + [0.5, 0, 0]).vertices)


def test_slice_of_empty_room_is_one_closed_interior() -> None:
    room = _room()
    for height in ag.SLICE_HEIGHTS_M:
        cut = ag.floor_slice(room, UP, height)
        assert cut.single_closed_interior and cut.n_parts == 1
        assert cut.area_m2 == pytest.approx(24.0)
        assert cut.n_interior_loops == 0 and cut.open_length_fraction == pytest.approx(0.0)
    assert ag.mesh_summary(room, UP)["watertight"] and ag.mesh_summary(room, UP)["floor_at_zero"]


def test_slice_reports_furniture_as_interior_loop_without_shrinking_footprint() -> None:
    cut = ag.floor_slice(_with_table(_room()), UP, 1.1)
    assert cut.single_closed_interior
    assert cut.n_interior_loops == 1
    assert cut.interior_loop_area_m2 == pytest.approx(2.0)
    assert cut.area_m2 == pytest.approx(24.0)
    assert ag.floor_slice(_with_table(_room()), UP, 2.0).n_interior_loops == 0  # above the table


def test_slice_detects_two_disjoint_rooms_and_open_outline() -> None:
    two = trimesh.util.concatenate([_room(), _room(offset=(10.0, 0.0, 0.0))])
    cut = ag.floor_slice(two, UP, 1.1)
    assert cut.n_parts == 2 and not cut.single_closed_interior
    open_room = _room()
    wall = np.abs(open_room.face_normals[:, 0] - 1.0) < 1e-6  # drop the +x wall
    open_room.update_faces(~wall)
    cut = ag.floor_slice(open_room, UP, 1.1)
    assert not cut.single_closed_interior and cut.open_length_fraction == pytest.approx(1.0)


def test_containment_passes_inside_and_fails_on_axis_swap() -> None:
    room = _room()
    footprint = ag.floor_slice(room, UP, 1.1).footprint
    rng = np.random.default_rng(0)
    points = rng.uniform(0.5, SIZE - 0.5, size=(50, 3))
    good = ag.containment(room, footprint, points, UP)
    assert good["inside_fraction"] == 1.0
    assert good["surface_distance_min_m"] >= 0.5 - 1e-9
    assert good["meets_documented_clearance_fraction"] == 1.0
    swapped = ag.containment(room, footprint, points[:, [2, 1, 0]], UP)  # x and z exchanged
    assert swapped["inside_fraction"] < 0.6


def test_line_of_sight_blocked_only_through_the_table() -> None:
    room = _with_table(_room())
    sources = np.array([[1.0, 2.0, 1.0], [1.0, 2.0, 2.0], [1.0, 0.5, 1.0]])
    receivers = np.array([[5.0, 2.0, 1.0], [5.0, 2.0, 2.0], [5.0, 0.5, 1.0]])
    assert ag.line_of_sight_blocked(room, sources, receivers).tolist() == [True, False, False]


def test_footprint_vertices_feed_both_eligibility_rules() -> None:
    footprint = ag.floor_slice(_room(np.array([10.0, 6.0, 2.5])), UP, 1.1).footprint
    vertices = ag.footprint_vertices(footprint)
    assert fits_bbox(vertices, CANVAS_12_8_M)
    assert anchored_coverage(vertices, [5.0, 3.0], CANVAS_12_8_M)
    assert not anchored_coverage(vertices, [1.0, 3.0], CANVAS_12_8_M)
