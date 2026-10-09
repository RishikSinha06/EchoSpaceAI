"""Plot a seeded sample of completed held-out B0/A/P predictions for review."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from echospace.data.evalmasks import read_manifest
from echospace.training.frozen import FrozenDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=REPO / "data/cache/p7_fold0_fixed/manifest.json")
    parser.add_argument("--runs", type=Path, default=REPO / "artifacts/p7_fold0_fixed")
    parser.add_argument("--eval-dir", type=Path, default=REPO / "data/processed/eval_samples")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    dataset = FrozenDataset(args.frozen, "test")
    records = read_manifest(args.eval_dir)
    tables = {}
    for variant in ("B0", "A", "P"):
        export = json.loads((args.runs / variant / "predictions_test/manifest.json").read_text(encoding="utf-8"))
        tables[variant] = {p["sample_id"]: p for p in export["predictions"]}
    chosen = np.random.default_rng(0).choice(len(dataset), min(args.count, len(dataset)), replace=False).tolist()
    out = args.runs / "review"
    out.mkdir(exist_ok=True)
    for page in range(0, len(chosen), 4):
        indices = chosen[page:page+4]
        for head in ("occupancy", "boundary"):
            fig, axes = plt.subplots(len(indices), 4, figsize=(12, 3*len(indices)), squeeze=False)
            for row, index in enumerate(indices):
                entry, arrays = dataset.entries[index], dataset[index]
                sid = entry["metadata"]["sample_id"]
                record = records[sid]
                axes[row, 0].imshow(arrays[f"target_{head}"], origin="lower", vmin=0, vmax=1)
                axes[row, 0].set_title(f"{record['room_id']}\n{record['occlusion_type']} / {record['coverage_bin']}", fontsize=8)
                for column, variant in enumerate(("B0", "A", "P"), start=1):
                    pred = tables[variant][sid]
                    with np.load(args.runs / variant / "predictions_test" / pred["prediction_path"], allow_pickle=False) as npz:
                        axes[row, column].imshow(npz[f"{head}_probability"], origin="lower", vmin=0, vmax=1)
                    axes[row, column].set_title(f"{variant} {head} probability")
                for ax in axes[row]:
                    ax.contour(arrays["observed_cells"], levels=[0.5], colors="red", linewidths=0.5)
                    ax.set_xticks([0, 32, 63])
                    ax.set_yticks([0, 32, 63])
            fig.suptitle(f"P7 fold {dataset.manifest['source']['fold']} held-out; red = observation outline")
            fig.tight_layout()
            fig.savefig(out / f"{head}_{page//4+1}.png", dpi=120)
            plt.close(fig)
    selection = {"seed": 0, "selection": "uniform without replacement, independent of metrics",
                 "frozen_checksum": dataset.manifest["checksum"],
                 "sample_ids": [dataset.entries[i]["metadata"]["sample_id"] for i in chosen]}
    (out / "selection.json").write_text(json.dumps(selection, indent=2)+"\n", encoding="utf-8")
    print(f"Prepared occupancy/boundary overlays for {len(chosen)} held-out samples in {out}")


if __name__ == "__main__":
    main()
