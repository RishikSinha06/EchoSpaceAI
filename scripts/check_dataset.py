"""P6 gate: splits, fixed evaluation masks and the dataset on real rooms.

    python scripts/check_dataset.py

Plan Stage 1 gate: a visual check of 20 random samples (label, G, M and
sensor positions overlaid), all unit tests passing, and a histogram over all
coverage bins and occlusion types. This script also re-validates every fixed
evaluation sample against its committed checksum and contract v0.1.0,
re-checks that no D0 duplicate group crosses a split, and measures how often
a training draw gets the K it asked for.

Writes reports/p6_dataset_check.md, reports/p6_dataset_summary.json and
reports/figures/p6/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from audit_data import versions  # noqa: E402
from echospace.data import EchoSpaceDataset, load_eval_spec, load_folds, read_eval_sample  # noqa: E402
from echospace.data.evalmasks import read_manifest  # noqa: E402
from echospace.scans import COVERAGE_BINS, OCCLUSION_TYPES  # noqa: E402

FIGURE_SAMPLES = 20
K_PROBE = 200
MAX_FIGURE_BYTES = 300_000


def draw(items: list[dict[str, Any]], path: Path, title: str) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colours = ListedColormap(["white", "0.85", "#8fc1e8", "#d1495b"])
    fig, axes = plt.subplots(1, len(items), figsize=(3.2 * len(items), 3.6), dpi=70)
    for ax, item in zip(np.atleast_1d(axes), items):
        image = item["target_occupancy"].astype(int)
        image[item["observed_free"].astype(bool)] = 2
        image[item["observed_wall"].astype(bool)] = 3
        ax.imshow(image, cmap=colours, vmin=0, vmax=3, interpolation="nearest")
        boundary = np.ma.masked_where(item["target_boundary"] == 0, item["target_boundary"])
        ax.imshow(boundary, cmap=ListedColormap(["0.45"]), alpha=0.35, interpolation="nearest")
        valid = item["rir_valid"].astype(bool)
        src, mic = item["src_pos"][valid] / 0.2 - 0.5, item["mic_pos"][valid] / 0.2 - 0.5
        ax.plot(src[:, 0], src[:, 1], "k^", markersize=5)
        ax.plot(mic[:, 0], mic[:, 1], "o", color="#2a7f3f", markersize=4, markerfacecolor="none")
        turn = item["spatial_transform"]
        ax.set_title(f"{item['room_id']}\n{item['occlusion_type']} {item['coverage_bin']} cov {item['coverage']:.2f}\n"
                     f"K {item['k']}/{item['k_requested']} rot {turn['quarter_turns']} flip {int(turn['flip_columns'])}", fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(title + "  (grey: room label, dark: boundary label, blue: observed free, red: observed wall, ^ source, o mic)", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path.stat().st_size


def histogram(counts: dict[str, Counter], path: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = list(COVERAGE_BINS)
    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=80)
    width = 0.8 / len(OCCLUSION_TYPES)
    for k, kind in enumerate(OCCLUSION_TYPES):
        bars = ax.bar(np.arange(len(bins)) + (k - 1.5) * width, [counts[kind][b] for b in bins], width, label=kind)
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(np.arange(len(bins)), bins)
    ax.set_ylabel("fixed evaluation masks")
    ax.set_title("Fixed evaluation masks by coverage bin and occlusion type", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rooms-dir", default="data/processed/rooms")
    parser.add_argument("--eval-dir", default="data/processed/eval_samples")
    parser.add_argument("--spec", default="data/splits/eval_masks.json")
    parser.add_argument("--folds", default="data/splits/folds.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=".", help="folder that receives reports/ (default: repository root)")
    args = parser.parse_args()
    out = REPO / args.out

    started = time.time()
    spec = load_eval_spec(REPO / args.spec)
    folds = load_folds(REPO / args.folds)
    if folds["eval_spec_checksum"] != spec["checksum"]:
        raise SystemExit("folds were made from another eval_masks.json")
    figures = out / "reports" / "figures" / "p6"

    # 1. every fixed sample: checksum + contract
    manifest = read_manifest(REPO / args.eval_dir)
    for entry in spec["samples"]:
        read_eval_sample(REPO / args.eval_dir, manifest[entry["sample_id"]], entry["sha256"])
    counts: dict[str, Counter] = {kind: Counter() for kind in OCCLUSION_TYPES}
    for entry in spec["samples"]:
        counts[entry["occlusion_type"]][entry["coverage_bin"]] += 1
    pairs = np.array([e["eligible_pairs"] for e in spec["samples"]])
    short = sum(v["short_of_pairs"] for room in spec["feasibility"].values() for t in room.values() for v in t.values())
    scans = sum(v["scans"] for room in spec["feasibility"].values() for t in room.values() for v in t.values())
    combos = Counter()
    for room in spec["feasibility"].values():
        for kind, by_bin in room.items():
            for bin_name, v in by_bin.items():
                combos["applicable" if v["applicable"] else "not_applicable"] += 1
                if v["applicable"]:
                    combos["with_masks" if v["masks"] else ("scans_but_short_of_pairs" if v["scans"] else "no_scan")] += 1

    # 2. splits: D0 duplicate groups never cross parts (independent re-check)
    duplicate_group = json.loads((REPO / "reports" / "d0_duplicates.json").read_text(encoding="utf-8"))["room_group"]
    crossings = 0
    for fold in folds["folds"]:
        part_of = {room: part for part in ("test", "val", "train") for room in fold[f"{part}_rooms"]}
        parts_by_group: dict[int, set[str]] = defaultdict(set)
        for room, part in part_of.items():
            parts_by_group[duplicate_group[room]].add(part)
        crossings += sum(len(p) > 1 for p in parts_by_group.values())
    fold_table = [{"fold": f["fold"], **{f"{p}_rooms": len(f[f"{p}_rooms"]) for p in ("test", "val", "train")},
                   **{f"{p}_groups": len(f[f"{p}_groups"]) for p in ("test", "val", "train")},
                   "test_nonconvex_rooms": sum(not folds["rooms"][r]["convex"] for r in f["test_rooms"]),
                   "test_masks": sum(1 for e in spec["samples"] if e["room_id"] in set(f["test_rooms"]))}
                  for f in folds["folds"]]

    # 3. training draws on one fold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        train = EchoSpaceDataset(REPO / args.rooms_dir, folds, args.fold, "train", eval_spec=spec, seed=args.seed)
        rng = np.random.default_rng(args.seed)
        probe = rng.choice(len(train), size=min(K_PROBE, len(train)), replace=False)
        tick = time.time()
        items = []
        for i in probe:
            item = train[int(i)]
            items.append({k: item[k] for k in ("k", "k_requested", "occlusion_type", "coverage_bin", "room_id")})
            if len(items) <= FIGURE_SAMPLES:
                items[-1]["full"] = item
        per_item = (time.time() - tick) / len(probe)
        test = EchoSpaceDataset(REPO / args.rooms_dir, folds, args.fold, "test", eval_dir=REPO / args.eval_dir, eval_spec=spec)
        test_items = [test[int(i)] for i in np.random.default_rng(args.seed + 1).choice(len(test), size=4, replace=False)]
    shown = [it.pop("full") for it in items if "full" in it]
    sizes = {}
    for n in range(0, FIGURE_SAMPLES, 4):
        name = f"p6_train_samples_{n // 4 + 1}.png"
        sizes[name] = draw(shown[n:n + 4], figures / name, f"fold {args.fold} training samples {n + 1}-{n + 4}")
    sizes["p6_test_samples.png"] = draw(test_items, figures / "p6_test_samples.png", f"fold {args.fold} fixed test samples (K = 8)")
    sizes["p6_eval_mask_histogram.png"] = histogram(counts, figures / "p6_eval_mask_histogram.png")
    reduced = [it for it in items if it["k"] < it["k_requested"]]

    summary = {
        "eval_samples": len(spec["samples"]),
        "eval_samples_checked": len(spec["samples"]),
        "rooms_with_masks": spec["rooms_with_masks"],
        "rooms_cached": spec["rooms"],
        "masks_by_type_and_bin": {k: dict(v) for k, v in counts.items()},
        "eligible_pairs_per_mask": {"min": int(pairs.min()), "median": float(np.median(pairs)), "max": int(pairs.max())},
        "eval_scans_short_of_8_pairs": {"short": short, "scans": scans},
        "type_bin_combinations": dict(combos),
        "folds": fold_table,
        "excluded_rooms_without_masks": folds["excluded_rooms_without_masks"],
        "d0_group_crossings": crossings,
        "train_probe": {
            "fold": args.fold, "items": len(items), "train_rooms": len(train.rooms), "train_len": len(train),
            "k_reduced": len(reduced), "k_reduced_examples": reduced[:5],
            "k_delivered": dict(Counter(it["k"] for it in items)),
            "types": dict(Counter(it["occlusion_type"] for it in items)),
            "bins": dict(Counter(it["coverage_bin"] for it in items)),
            "seconds_per_item": round(per_item, 3),
        },
        "figures": sizes,
        "figures_over_limit": {k: v for k, v in sizes.items() if v > MAX_FIGURE_BYTES},
        "versions": versions(),
        "seconds": round(time.time() - started),
    }
    (out / "reports" / "p6_dataset_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    (out / "reports" / "p6_dataset_check.md").write_text(render(summary, spec), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("folds",)}, indent=1, default=str))
    return 0


def render(s: dict[str, Any], spec: dict[str, Any]) -> str:
    bins = list(COVERAGE_BINS)
    t = s["train_probe"]
    lines = [
        "# P6 dataset check",
        "",
        f"Generated by `scripts/check_dataset.py` at code revision `{s['versions']['code_revision']}`.",
        "",
        "## Fixed evaluation masks",
        "",
        f"- {s['eval_samples']} samples over {s['rooms_with_masks']} of {s['rooms_cached']} cached rooms; every one re-read, "
        "matched its SHA-256 in `data/splits/eval_masks.json` and passed `contract_v0.validate_sample`.",
        f"- Each has {spec['k_required']} nested RIRs (K = 1/2/4/8 on the same mask); usable pairs per mask: "
        f"min {s['eligible_pairs_per_mask']['min']}, median {s['eligible_pairs_per_mask']['median']:.0f}, max {s['eligible_pairs_per_mask']['max']}.",
        f"- Scans dropped for having fewer than {spec['k_required']} usable pairs: {s['eval_scans_short_of_8_pairs']['short']} "
        f"of {s['eval_scans_short_of_8_pairs']['scans']}.",
        f"- (room, type, bin) combinations: {s['type_bin_combinations']}.",
        "",
        "| Type | " + " | ".join(bins) + " |",
        "| --- | " + " | ".join("---:" for _ in bins) + " |",
    ]
    for kind, row in s["masks_by_type_and_bin"].items():
        lines.append(f"| {kind} | " + " | ".join(str(row.get(b, 0)) for b in bins) + " |")
    lines += [
        "",
        "![histogram](figures/p6/p6_eval_mask_histogram.png)",
        "",
        "## Folds",
        "",
        "| Fold | test rooms (groups) | val rooms (groups) | train rooms (groups) | non-convex test rooms | test masks |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for f in s["folds"]:
        lines.append(f"| {f['fold']} | {f['test_rooms']} ({f['test_groups']}) | {f['val_rooms']} ({f['val_groups']}) | "
                     f"{f['train_rooms']} ({f['train_groups']}) | {f['test_nonconvex_rooms']} | {f['test_masks']} |")
    lines += [
        "",
        f"- D0 duplicate groups crossing a split, re-checked from `reports/d0_duplicates.json`: **{s['d0_group_crossings']}**.",
        f"- Rooms without any evaluation mask, left out of the folds: {', '.join(s['excluded_rooms_without_masks']) or 'none'}.",
        "",
        "## Training draws (fold {})".format(t["fold"]),
        "",
        f"- {t['train_rooms']} training rooms, {t['train_len']} items per epoch; probed {t['items']} random items at "
        f"{t['seconds_per_item']} s each (single process).",
        f"- K delivered: {dict(sorted(t['k_delivered'].items()))}; draws that got fewer usable pairs than the K requested: "
        f"{t['k_reduced']} of {t['items']}.",
        f"- Types {t['types']}; bins {t['bins']}.",
        "- Overlays of the first 20 probed items: `figures/p6/p6_train_samples_1..5.png`; four fixed test samples: "
        "`figures/p6/p6_test_samples.png`.",
        "",
        f"Versions: {', '.join(f'{k} {v}' for k, v in s['versions'].items() if k != 'code_revision')}.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
