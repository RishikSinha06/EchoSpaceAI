"""P7 foundation gate on frozen synthetic rooms, never a dataset benchmark."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from echospace.geometry.frames import GridFrame
from echospace.models import SpatialConfig
from echospace.training.frozen import freeze_samples
from echospace.training.runner import TrainConfig, run_baseline


def synthetic_item(room, index, seed):
    """Closed shoebox label, visible lower wall and partial interior strip.

    Different sizes define distinct synthetic rooms. Pose rows sit only in
    observed free space. No measured acoustic signal is fabricated or consumed.
    """
    rng = np.random.default_rng(seed)
    lo, hi = 12 + seed % 4, 46 + seed % 5
    target = np.zeros((64, 64), np.uint8)
    target[lo:hi+1, lo:hi+1] = 1
    boundary = np.zeros_like(target)
    boundary[lo, lo:hi+1] = boundary[hi, lo:hi+1] = 1
    boundary[lo:hi+1, lo] = boundary[lo:hi+1, hi] = 1
    free = np.zeros_like(target)
    free[lo+1:lo+8+index, lo+1:hi] = 1
    wall = np.zeros_like(target)
    wall[lo, lo+1:hi] = 1
    seen = free | wall
    poses = np.zeros((8, 2, 3), np.float32)
    cells = np.argwhere(free)
    for pair in range(2):
        for member in range(2):
            row, col = cells[rng.integers(len(cells))]
            poses[pair, member] = [(col+0.5)*0.2, (row+0.5)*0.2, 1.2]
    return {"sample_id": f"synthetic_room_{room}_mask_{index}", "room_id": f"synthetic_room_{room}", "room_group": room,
            "grid": GridFrame((0, 1.1, 0), 0).manifest_grid(), "spatial_transform": {"quarter_turns": 0, "flip_columns": False},
            "observed_cells": seen, "observed_free": free, "observed_wall": wall,
            "target_occupancy": target, "target_boundary": boundary, "valid_cells": np.ones_like(target),
            "src_pos": poses[:, 0], "mic_pos": poses[:, 1], "rir_valid": np.array([1, 1, 0, 0, 0, 0, 0, 0], np.uint8)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts/p7_check")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--plots", action="store_true", help="draw held-out probabilities; requires existing audit extras")
    args = parser.parse_args()
    samples = {s: [synthetic_item(room, i, 17+room) for room in rooms for i in range(2)]
               for s, rooms in {"train": range(4), "val": range(4, 5), "test": range(5, 6)}.items()}
    frozen = freeze_samples(args.out_dir / "frozen", samples,
                            {"kind": "synthetic_p7_gate", "seed": 0, "note": "six disjoint artificial rooms; no acoustic data"})
    results = {}
    for variant in ("B0", "A", "P"):
        results[variant] = run_baseline(variant, frozen, args.out_dir / variant,
                                        TrainConfig(epochs=3, batch_size=4, patience=2, device=args.device),
                                        SpatialConfig(base_channels=8))
    checksums = {r["frozen_checksum"] for r in results.values()}
    ids = [r["split_sample_ids"] for r in results.values()]
    if len(checksums) != 1 or not all(i == ids[0] for i in ids):
        raise RuntimeError("baselines did not use identical frozen samples")
    report = {"foundation_gate": "PASS", "full_p6_held_out_gate": "PENDING: generated P6 artifacts not present here",
              "scope": "synthetic foundation only; not AcousticRooms accuracy or fusion readiness",
              "frozen_checksum": next(iter(checksums)), "device": args.device,
              "variants": {v: {"parameters": r["parameter_count"], "test": r["test_metrics"],
                                  "best_epoch": r.get("best_epoch"), "amp_enabled": r["amp_enabled"]} for v, r in results.items()}}
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for index in range(2):
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            heldout = samples["test"][index]
            for row, head in enumerate(("occupancy", "boundary")):
                axes[row, 0].imshow(heldout[f"target_{head}"], origin="lower", vmin=0, vmax=1)
                axes[row, 0].set_title(f"Synthetic {head} target")
                for column, variant in enumerate(("B0", "A", "P"), start=1):
                    with np.load(args.out_dir / variant / "predictions_test" / f"{index:06d}.npz") as prediction:
                        axes[row, column].imshow(prediction[f"{head}_probability"], origin="lower", vmin=0, vmax=1)
                    axes[row, column].contour(heldout["observed_cells"], levels=[0.5], colors="red", linewidths=0.7)
                    axes[row, column].set_title(f"{variant}: {head} probability")
                for ax in axes[row]:
                    ax.set_xlabel("column")
                    ax.set_ylabel("row")
            fig.suptitle("P7 synthetic held-out smoke only; red outline = observed cells")
            fig.tight_layout()
            fig.savefig(args.out_dir / f"heldout_{index}.png", dpi=120)
            plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
