"""P3 scanner-frame and geometry labels, using only synthetic meshes."""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("shapely")

from echospace.geometry.frames import GridFrame, source_to_scene  # noqa: E402
from echospace.geometry.raster import GeometryLabelError, label_acousticrooms_mesh  # noqa: E402


def room(width: float = 6.0, depth: float = 4.0) -> "trimesh.Trimesh":
    box = trimesh.creation.box(extents=(width, depth, 2.5))
    box.apply_translation([width / 2, depth / 2, 1.25])
    return box


def grid(anchor_source: tuple[float, float, float] = (3.0, 2.0, 1.1), heading: float = 0.0) -> GridFrame:
    anchor = source_to_scene(np.array([anchor_source]))[0]
    return GridFrame(tuple(anchor), heading)


def test_source_to_scene_and_grid_round_trip() -> None:
    points = source_to_scene(np.array([[3.0, 2.0, 1.1], [4.0, 1.0, 1.2]]))
    assert np.allclose(points, [[3, 1.1, -2], [4, 1.2, -1]])
    for angle in (0.0, np.pi / 4, np.pi / 2):
        frame = grid(heading=angle)
        cell = frame.scene_to_cell(np.array([frame.anchor_scene_m]))[0]
        assert np.allclose(cell, [32, 32])
        assert np.allclose(frame.origin_scene_m + 32 * frame.cell_size_m *
                           (frame.axis_u_scene + frame.axis_v_scene), frame.anchor_scene_m)
    with pytest.raises(ValueError, match="anchor"):
        GridFrame((float("nan"), 0, 0), 0)


def test_room_labels_follow_explicit_scan_grid() -> None:
    labels = label_acousticrooms_mesh(room(), grid())
    assert labels.occupancy.shape == (64, 64)
    assert labels.boundary.shape == labels.valid_cells.shape == (64, 64)
    assert labels.valid_cells.all()
    assert (labels.boundary <= labels.occupancy).all()
    assert labels.diagnostics["footprint_area_m2"] == pytest.approx(24.0)
    assert labels.diagnostics["raster_area_m2"] == pytest.approx(24.0, abs=1.0)
    assert labels.occupancy[32, 32]
    turned = label_acousticrooms_mesh(room(), grid(heading=np.pi / 4))
    assert turned.occupancy.sum() == pytest.approx(labels.occupancy.sum(), abs=30)


def test_closed_furniture_loop_is_filled_and_not_a_wall_label() -> None:
    base = label_acousticrooms_mesh(room(), grid())
    table = trimesh.creation.box(extents=(1.0, 1.0, 1.5))
    table.apply_translation([3.0, 2.0, 0.75])
    furnished = label_acousticrooms_mesh(trimesh.util.concatenate([room(), table]), grid())
    assert furnished.diagnostics["furniture_loops_filled"] == 1
    assert np.array_equal(furnished.occupancy, base.occupancy)
    assert np.array_equal(furnished.boundary, base.boundary)


def test_rejects_bad_anchor_clipping_and_multiple_rooms() -> None:
    with pytest.raises(GeometryLabelError, match="outside"):
        label_acousticrooms_mesh(room(), grid((10.0, 10.0, 1.1)))
    with pytest.raises(GeometryLabelError, match="clips"):
        label_acousticrooms_mesh(room(width=14.0), grid((7.0, 2.0, 1.1)))
    second = room()
    second.apply_translation([10, 0, 0])
    with pytest.raises(GeometryLabelError, match="one closed"):
        label_acousticrooms_mesh(trimesh.util.concatenate([room(), second]), grid())


def test_slice_height_is_explicitly_bounded() -> None:
    with pytest.raises(GeometryLabelError, match="slice height"):
        label_acousticrooms_mesh(room(), grid(), slice_height_above_floor_m=1.4)
