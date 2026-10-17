"""Exports of one reconstruction: ``echospace.grid.v2`` JSON, GLB and the shared wall rule.

``echospace.grid.v2`` extends the viewer's v1 grid with the real metric frame
(grid origin and axes in scene metres, so rotated scan frames render in place),
the measured wall-hit cells, raw probabilities and, when known, the ground
truth. v1 files still load in the viewer unchanged.

Wall rule, shared with ``apps/web/src/grid.js`` and checked against it in CI:
a wall face stands on every edge between an interior cell and a non-interior
cell (or the canvas border). It is *observed* when the outside cell is a
measured wall hit, otherwise *inferred*. Scene frame: X right, Y up, Z
forward; floor at Y = 0; the wall height is a display assumption.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from echospace.inference.predict import Reconstruction

GRID_V2 = "echospace.grid.v2"
DEFAULT_WALL_HEIGHT_M = 2.7
# (d_row, d_col) of the neighbour across each edge, and that edge's corners as (u, v) grid offsets
_EDGES = (
    (-1, 0, (0, 0), (1, 0)),
    (1, 0, (1, 1), (0, 1)),
    (0, -1, (0, 1), (0, 0)),
    (0, 1, (1, 0), (1, 1)),
)


def _rows(array: np.ndarray, decimals: int | None = None) -> list[list[Any]]:
    if decimals is None:
        return np.asarray(array).astype(int).tolist()
    return np.round(np.asarray(array, dtype=np.float64), decimals).tolist()


def grid_document(reconstruction: Reconstruction, wall_height_m: float = DEFAULT_WALL_HEIGHT_M, include_truth: bool = True) -> dict[str, Any]:
    arrays = reconstruction.sample.arrays
    record = reconstruction.sample.record
    grid = record["grid"]
    k = reconstruction.k_used
    positions = np.asarray(arrays["rir_positions_scene_m"], dtype=np.float64)[:k]
    document: dict[str, Any] = {
        "format": GRID_V2,
        "name": f"{record['room_id']} / {reconstruction.model.model_id} / K={k}",
        "reconstruction_id": reconstruction.reconstruction_id,
        "cell_size_m": float(grid["cell_size_m"]),
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "origin_scene_m": [float(v) for v in grid["origin_scene_m"]],
        "axis_u_scene": [float(v) for v in grid["axis_u_scene"]],
        "axis_v_scene": [float(v) for v in grid["axis_v_scene"]],
        "wall_height_m": wall_height_m,
        "wall_height_assumed": True,
        "interior": _rows(reconstruction.interior),
        "observed": _rows(arrays["observed_cells"]),
        "observed_wall": _rows(arrays["observed_wall"]),
        "occupancy_probability": _rows(reconstruction.occupancy_probability, 3),
        "boundary_probability": _rows(reconstruction.boundary_probability, 3),
        "measurements": [{"source_scene_m": p[0].tolist(), "receiver_scene_m": p[1].tolist()} for p in positions],
        "display_convention": reconstruction.provenance["display_convention"],
        "truth": None,
    }
    if include_truth:
        document["truth"] = {"interior": _rows(arrays["target_occupancy"]), "boundary": _rows(arrays["target_boundary"])}
    return document


def validate_grid_v2(document: dict[str, Any]) -> None:
    if document.get("format") != GRID_V2:
        raise ValueError(f"expected format {GRID_V2}")
    height, width = int(document["height"]), int(document["width"])
    for key in ("interior", "observed", "observed_wall"):
        array = np.asarray(document[key])
        if array.shape != (height, width) or not np.isin(array, (0, 1)).all():
            raise ValueError(f"{key} must be a {height}x{width} binary grid")
    if not np.asarray(document["interior"]).any():
        raise ValueError("grid contains no interior cells")
    u, v = np.asarray(document["axis_u_scene"], float), np.asarray(document["axis_v_scene"], float)
    if abs(np.linalg.norm(u) - 1) > 1e-5 or abs(np.linalg.norm(v) - 1) > 1e-5 or abs(u @ v) > 1e-5:
        raise ValueError("grid axes must be orthonormal")


def wall_edges(interior: np.ndarray, observed_wall: np.ndarray) -> list[tuple[int, int, int, bool]]:
    """``(row, col, edge_index, observed)`` for every wall face of the interior."""
    inside = np.asarray(interior).astype(bool)
    hits = np.asarray(observed_wall).astype(bool)
    rows, cols = inside.shape
    edges = []
    for r, c in zip(*np.nonzero(inside)):
        for index, (dr, dc, _, _) in enumerate(_EDGES):
            nr, nc = r + dr, c + dc
            on_canvas = 0 <= nr < rows and 0 <= nc < cols
            if on_canvas and inside[nr, nc]:
                continue
            edges.append((int(r), int(c), index, bool(on_canvas and hits[nr, nc])))
    return edges


def scene_geometry(document: dict[str, Any]) -> dict[str, np.ndarray]:
    """Triangle soup (N x 3 x 3, scene metres) for floor, observed and inferred walls."""
    validate_grid_v2(document)
    origin = np.asarray(document["origin_scene_m"], dtype=np.float64)
    u, v = np.asarray(document["axis_u_scene"], float), np.asarray(document["axis_v_scene"], float)
    cell, height = float(document["cell_size_m"]), float(document["wall_height_m"])
    up = np.array([0.0, height, 0.0])

    def corner(col: float, row: float) -> np.ndarray:
        point = origin + cell * (col * u + row * v)
        point[1] = 0.0  # floor plane; the grid's own height is the scan height, not the floor
        return point

    def quad(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> list[np.ndarray]:
        return [np.stack([a, b, c]), np.stack([a, c, d])]

    interior = np.asarray(document["interior"])
    floor: list[np.ndarray] = []
    for r, c in zip(*np.nonzero(interior)):
        floor += quad(corner(c, r), corner(c, r + 1), corner(c + 1, r + 1), corner(c + 1, r))
    walls: dict[bool, list[np.ndarray]] = {True: [], False: []}
    for r, c, index, observed in wall_edges(interior, np.asarray(document["observed_wall"])):
        _, _, (ua, va), (ub, vb) = _EDGES[index]
        a, b = corner(c + ua, r + va), corner(c + ub, r + vb)
        walls[observed] += quad(a, b, b + up, a + up)
    empty = np.zeros((0, 3, 3))
    return {
        "floor": np.stack(floor) if floor else empty,
        "observed_walls": np.stack(walls[True]) if walls[True] else empty,
        "inferred_walls": np.stack(walls[False]) if walls[False] else empty,
    }


def geometry_stats(document: dict[str, Any]) -> dict[str, Any]:
    """Numbers the browser renderer must reproduce exactly for the same document."""
    parts = scene_geometry(document)
    cell = float(document["cell_size_m"])
    every = np.concatenate([p.reshape(-1, 3) for p in parts.values() if len(p)])
    return {
        "floor_triangles": int(len(parts["floor"])),
        "observed_wall_triangles": int(len(parts["observed_walls"])),
        "inferred_wall_triangles": int(len(parts["inferred_walls"])),
        "area_m2": round(float(np.asarray(document["interior"]).sum()) * cell * cell, 6),
        "observed_wall_m": round(len(parts["observed_walls"]) / 2 * cell, 6),
        "inferred_wall_m": round(len(parts["inferred_walls"]) / 2 * cell, 6),
        "bounds_min": [round(float(x), 5) for x in every.min(axis=0)],
        "bounds_max": [round(float(x), 5) for x in every.max(axis=0)],
    }


def glb_bytes(document: dict[str, Any], provenance: dict[str, Any] | None = None) -> bytes:
    """Binary glTF with three named meshes in scene metres (Y up), as the viewer draws them."""
    import trimesh

    colours = {"floor": (36, 55, 70, 200), "observed_walls": (32, 200, 193, 225), "inferred_walls": (255, 186, 102, 185)}
    scene = trimesh.Scene()
    for name, triangles in scene_geometry(document).items():
        if not len(triangles):
            continue
        vertices = triangles.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=np.tile(colours[name], (len(faces), 1)))
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.metadata["echospace"] = {
        "format": document["format"],
        "reconstruction_id": document.get("reconstruction_id"),
        "units": "metres",
        "frame": "scene: X right, Y up, Z forward",
        "wall_height_m": document["wall_height_m"],
        "wall_height_assumed": True,
        "provenance_sidecar": "provenance.json",
        **({"model_id": provenance["model"]["model_id"], "sample_id": provenance["sample"]["sample_id"]} if provenance else {}),
    }
    buffer = io.BytesIO()
    scene.export(buffer, file_type="glb")
    return buffer.getvalue()
