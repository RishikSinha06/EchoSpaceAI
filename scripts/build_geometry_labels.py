"""Build P3 labels for one mesh with an explicitly supplied scan pose.

This writes geometry targets only. P4 supplies actual partial observations;
P6 assembles verified training samples and their RIRs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from echospace.geometry.frames import source_to_scene
from echospace.geometry.raster import GridFrame, RoomLabels, label_acousticrooms_mesh


def save_overlay(path: Path, labels: RoomLabels) -> None:
    """Draw the continuous outline over the raster for human review."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = labels.grid
    ring = np.asarray(labels.footprint_scene_xz.exterior.coords)
    scene_points = np.column_stack((ring[:, 0], np.full(len(ring), grid.anchor_scene_m[1]), ring[:, 1]))
    cells = grid.scene_to_cell(scene_points)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(labels.occupancy, origin="upper", extent=(0, grid.width, grid.height, 0), cmap="Greys", vmin=0, vmax=1)
    rows, cols = np.nonzero(labels.boundary)
    ax.scatter(cols + 0.5, rows + 0.5, marker="s", s=42, facecolors="none",
               edgecolors="#e07118", linewidths=0.8, label="boundary cells")
    ax.plot(cells[:, 0], cells[:, 1], color="#1874ba", linewidth=1.2, label="mesh outline")
    ax.plot(grid.width / 2, grid.height / 2, marker="+", color="red", markersize=10, label="supplied anchor")
    ax.set(xlim=(0, grid.width), ylim=(grid.height, 0), xlabel="grid column", ylabel="grid row",
           title="Geometry-only target diagnostic")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path, help="Audited AcousticRooms OBJ")
    parser.add_argument("--anchor-source-xyz", required=True, type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--heading-deg-scene", required=True, type=float,
                        help="Scanner heading in scene XZ, from +X toward +Z")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--slice-height-m", type=float, default=1.1)
    args = parser.parse_args()

    anchor = source_to_scene(np.asarray([args.anchor_source_xyz], dtype=float))[0]
    grid = GridFrame(tuple(float(x) for x in anchor), np.deg2rad(args.heading_deg_scene))
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    labels = label_acousticrooms_mesh(mesh, grid, slice_height_above_floor_m=args.slice_height_m)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "geometry_labels.npz",
        target_occupancy=labels.occupancy.astype(np.uint8),
        target_boundary=labels.boundary.astype(np.uint8),
        valid_cells=labels.valid_cells.astype(np.uint8),
    )
    details = {
        "source_mesh": str(args.mesh),
        "anchor_source_m": args.anchor_source_xyz,
        "heading_deg_scene": args.heading_deg_scene,
        "slice_height_above_floor_m": args.slice_height_m,
        "grid": grid.manifest_grid(),
        "diagnostics": labels.diagnostics,
        "status": "geometry_only_not_training_sample",
    }
    (args.out_dir / "geometry_labels.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    save_overlay(args.out_dir / "geometry_overlay.png", labels)
    print(json.dumps(details["diagnostics"], sort_keys=True))


if __name__ == "__main__":
    main()
