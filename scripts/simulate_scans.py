"""Module 4 gate: simulate partial scans on the accepted AcousticRooms rooms.

For every accepted room in the D0 manifest it builds, per occlusion type,
``--seeds`` free-running scans (their coverage bins fill the histogram) and one
scan aimed at each coverage bin (whether that bin is reachable at all, which
P6 needs for its fixed test masks). Every built sample is checked against
contract v0.1.0. It also draws an overlay per occlusion type for 20 distinct
rooms (one per D0 duplicate group) that yield a sample, spread across categories: the plan's Stage 1 gate asks for a visual check of
20 samples and a histogram over all coverage bins and occlusion types.

    python scripts/simulate_scans.py                # all accepted rooms
    python scripts/simulate_scans.py --limit 12     # quick look, into data/cache/scan_trial

Reads the manifest and the archives (read only). Writes
``reports/p4_scan_check.md``, ``reports/p4_scans_summary.json`` and
``reports/figures/p4/``. Same dataset root rules as audit_data.py.
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

from audit_data import natural_key, versions  # noqa: E402
from echospace.contract_v0 import validate_sample  # noqa: E402
from echospace.geometry.frames import ACOUSTICROOMS_SOURCE_TO_SCENE  # noqa: E402
from echospace.geometry.raster import GeometryLabelError  # noqa: E402
from echospace.io.adapters import acousticrooms as ar  # noqa: E402
from echospace.scans import (  # noqa: E402
    COVERAGE_BINS,
    OCCLUSION_TYPES,
    FrameError,
    RoomGeometry,
    RoomGeometryError,
    ScanNotApplicable,
    ScanNotFeasible,
    ScanSample,
    build_scan_sample,
)

FIGURE_ROOMS = 20
BIN_ATTEMPTS = 15  # redraws when aiming at one coverage bin
MAX_FIGURE_BYTES = 300_000


def record_for(sample: ScanSample) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "sample_id": f"{sample.room_id}__{sample.occlusion_type}__{sample.seed}",
        "room_id": sample.room_id,
        "dataset": "AcousticRooms",
        "geometry_id": sample.room_id,
        "rir_ids": [],
        "rir_sample_rate_hz": 22050,
        "sample_path": "unwritten.npz",
        "split": "unassigned",
        "frame": {
            "description": "AcousticRooms source (x, y, z up, metres) to scene (X right, Y up, Z = -y)",
            "source_units_to_meters": 1.0,
            "source_to_scene": ACOUSTICROOMS_SOURCE_TO_SCENE.tolist(),
        },
        "grid": sample.grid.manifest_grid(),
    }


def attempt(room: RoomGeometry, room_id: str, kind: str, seed: int, bin_name: str | None) -> tuple[str, ScanSample | None]:
    try:
        sample = build_scan_sample(room, room_id, kind, seed, bin_name, max_attempts=BIN_ATTEMPTS if bin_name else 40)
    except ScanNotApplicable:
        return "not_applicable", None
    except ScanNotFeasible:
        return "bin_not_reached", None
    except GeometryLabelError as error:
        return ("clips_canvas" if "clips" in str(error) else "label_error"), None
    except FrameError:
        return "frame_error", None
    validate_sample(record_for(sample), sample.contract_arrays())
    return "ok", sample


def figure_rooms(room_ids: list[str], count: int) -> list[str]:
    """Round-robin across categories in natural order, like audit_data.select_rooms."""
    by_category: dict[str, list[str]] = defaultdict(list)
    for room_id in sorted(room_ids, key=natural_key):
        by_category[room_id.rsplit("_idx", 1)[0]].append(room_id)
    picked: list[str] = []
    depth = 0
    while len(picked) < min(count, len(room_ids)):
        for category in sorted(by_category):
            if depth < len(by_category[category]) and len(picked) < count:
                picked.append(by_category[category][depth])
        depth += 1
    return picked


def draw_room(room_id: str, samples: dict[str, ScanSample | None], outcomes: dict[str, str], path: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colours = ListedColormap(["white", "0.85", "#8fc1e8", "#d1495b"])  # outside, unobserved room, free, wall
    fig, axes = plt.subplots(1, len(OCCLUSION_TYPES), figsize=(12, 3.5), dpi=70)
    for ax, kind in zip(axes, OCCLUSION_TYPES):
        sample = samples.get(kind)
        ax.set_xticks([]), ax.set_yticks([])
        if sample is None:
            ax.text(0.5, 0.5, outcomes.get(kind, "n/a").replace("_", " "), ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(kind, fontsize=9)
            continue
        image = sample.labels.occupancy.astype(int)
        image[sample.observed.observed_free] = 2
        image[sample.observed.observed_wall] = 3
        ax.imshow(image, cmap=colours, vmin=0, vmax=3, origin="upper", interpolation="nearest")
        ring = np.asarray(sample.labels.footprint_scene_xz.exterior.coords)
        cells = sample.grid.scene_to_cell(np.column_stack((ring[:, 0], np.zeros(len(ring)), ring[:, 1])))
        ax.plot(cells[:, 0] - 0.5, cells[:, 1] - 0.5, color="0.3", linewidth=0.7)
        if sample.scan.hidden is not None:
            for part in getattr(sample.scan.hidden, "geoms", [sample.scan.hidden]):
                hidden = np.asarray(part.exterior.coords)
                h = sample.grid.scene_to_cell(np.column_stack((hidden[:, 0], np.zeros(len(hidden)), hidden[:, 1])))
                ax.plot(h[:, 0] - 0.5, h[:, 1] - 0.5, color="#7a3e9d", linewidth=1.0, linestyle="--")
        origins = np.array([[s.origin[0], 0.0, s.origin[1]] for s in sample.scan.scanners])
        o = sample.grid.scene_to_cell(origins)
        ax.plot(o[:, 0] - 0.5, o[:, 1] - 0.5, "k^", markersize=5)
        ax.plot(sample.grid.width / 2 - 0.5, sample.grid.height / 2 - 0.5, "+", color="#2a7f3f", markersize=9, markeredgewidth=1.5)
        ax.set_title(f"{kind}: cov {sample.coverage:.2f} ({sample.coverage_bin})", fontsize=9)
    fig.suptitle(f"{room_id}  (grey: room label, blue: observed free, red: observed wall, ^ scanner, + grid anchor, dashed: hidden)", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path.stat().st_size


def draw_histogram(counts: dict[str, Counter], path: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = list(COVERAGE_BINS)
    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=80)
    width = 0.8 / len(OCCLUSION_TYPES)
    for k, kind in enumerate(OCCLUSION_TYPES):
        values = [counts[kind][b] for b in bins]
        bars = ax.bar(np.arange(len(bins)) + (k - 1.5) * width, values, width, label=kind)
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(np.arange(len(bins)), [f"{b} [{lo:.2f}, {hi:.2f})" for b, (lo, hi) in COVERAGE_BINS.items()])
    ax.set_ylabel("free-running scans")
    ax.set_title("Coverage bin by occlusion type (all accepted rooms)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help=f"dataset root (default: ${ar.ROOT_ENV_VAR} or data/raw)")
    parser.add_argument("--manifest", default="data/manifests/acousticrooms_index.jsonl")
    parser.add_argument("--index-cache", default="data/cache/acousticrooms_index.json")
    parser.add_argument("--limit", type=int, help="first N accepted rooms only, into a scratch folder")
    parser.add_argument("--seeds", type=int, default=3, help="free-running scans per room and type")
    args = parser.parse_args()

    started = time.time()
    root = ar.resolve_root(args.root)
    with (REPO / args.manifest).open(encoding="utf-8") as handle:
        accepted = [row["room_id"] for row in map(json.loads, handle) if row["accepted"]]
    room_ids = figure_rooms(accepted, args.limit) if args.limit else sorted(accepted, key=natural_key)
    out = REPO / "data" / "cache" / "scan_trial" if args.limit else REPO
    # Overlay candidates: round-robin across categories, one room per D0 duplicate
    # group; the first FIGURE_ROOMS of them that yield a sample are drawn.
    groups_path = REPO / "reports" / "d0_duplicates.json"
    room_group = json.loads(groups_path.read_text(encoding="utf-8"))["room_group"] if groups_path.exists() else {}
    distinct, seen_groups = [], set()
    for room_id in sorted(room_ids, key=natural_key):
        group = room_group.get(room_id, room_id)
        if group not in seen_groups:
            seen_groups.add(group)
            distinct.append(room_id)
    candidates = figure_rooms(distinct, 3 * FIGURE_ROOMS)
    kept: dict[str, tuple[dict[str, ScanSample | None], dict[str, str]]] = {}
    index = ar.load_cached_index(root, REPO / args.index_cache)
    if index is None:
        raise SystemExit("index cache missing or stale; run scripts/audit_data.py first")

    free_runs: dict[str, Counter] = {kind: Counter() for kind in OCCLUSION_TYPES}
    aimed: dict[str, Counter] = {kind: Counter() for kind in OCCLUSION_TYPES}
    rooms: dict[str, Any] = {}
    sizes: dict[str, int] = {}
    contract_checked = 0
    with ar.ArchiveReader(root) as reader, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for position, room_id in enumerate(room_ids, 1):
            tick = time.time()
            try:
                room = RoomGeometry.from_mesh(ar.load_mesh(reader, index.rooms[room_id].mesh))
            except RoomGeometryError as error:
                rooms[room_id] = {"error": str(error)}
                continue
            entry: dict[str, Any] = {"convex": room.is_convex, "footprint_m2": round(room.footprint.area, 3), "types": {}}
            shown: dict[str, ScanSample | None] = {}
            outcomes: dict[str, str] = {}
            for kind in OCCLUSION_TYPES:
                runs = []
                for seed in range(args.seeds):
                    outcome, sample = attempt(room, room_id, kind, seed, None)
                    contract_checked += sample is not None
                    runs.append({"outcome": outcome, "coverage": round(sample.coverage, 4) if sample else None,
                                 "bin": sample.coverage_bin if sample else None})
                    free_runs[kind][sample.coverage_bin if sample else outcome] += 1
                    if seed == 0:
                        shown[kind], outcomes[kind] = sample, outcome
                reachable = {}
                for bin_name in COVERAGE_BINS:
                    outcome, sample = attempt(room, room_id, kind, 1000, bin_name)
                    contract_checked += sample is not None
                    reachable[bin_name] = outcome
                    aimed[kind][f"{bin_name}:{outcome}"] += 1
                entry["types"][kind] = {"free_runs": runs, "aimed_bins": reachable}
            rooms[room_id] = entry
            if room_id in candidates and any(sample is not None for sample in shown.values()):
                kept[room_id] = (shown, outcomes)
            summary = " ".join(f"{k}={'/'.join(str(r['bin'] or r['outcome'])[:4] for r in entry['types'][k]['free_runs'])}" for k in OCCLUSION_TYPES)
            print(f"[{position}/{len(room_ids)}] {room_id}: {summary} [{time.time() - tick:.1f}s]", flush=True)

    for room_id in [r for r in candidates if r in kept][:FIGURE_ROOMS]:
        shown, outcomes = kept[room_id]
        sizes[room_id] = draw_room(room_id, shown, outcomes, out / "reports" / "figures" / "p4" / f"p4_{room_id}.png")
    histogram_bytes = draw_histogram(free_runs, out / "reports" / "figures" / "p4" / "p4_coverage_histogram.png")
    feasible = {
        kind: {b: sum(1 for r in rooms.values() if "types" in r and r["types"][kind]["aimed_bins"][b] == "ok") for b in COVERAGE_BINS}
        for kind in OCCLUSION_TYPES
    }
    summary = {
        "rooms": len(room_ids),
        "contract_checked_samples": contract_checked,
        "free_running_scans_by_type": {k: dict(v) for k, v in free_runs.items()},
        "rooms_reaching_bin_by_type": feasible,
        "aimed_outcomes_by_type": {k: dict(v) for k, v in aimed.items()},
        "rooms_with_any_sample": sum(
            1 for r in rooms.values() if "types" in r and any(run["outcome"] == "ok" for t in r["types"].values() for run in t["free_runs"])
        ),
        "non_convex_rooms": sum(1 for r in rooms.values() if r.get("convex") is False),
        "figures": sorted(sizes, key=natural_key),
        "figures_over_limit": {k: v for k, v in sizes.items() if v > MAX_FIGURE_BYTES},
        "histogram_bytes": histogram_bytes,
    }
    result = {"summary": summary, "rooms": rooms, "seeds_per_room_and_type": args.seeds, "bin_attempts": BIN_ATTEMPTS, "versions": versions()}
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports" / "p4_scans_summary.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "reports" / "p4_scan_check.md").write_text(render(summary, args.seeds, result["versions"]), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))
    print(f"done in {time.time() - started:.0f}s; outputs under {out.relative_to(REPO) if out != REPO else '.'}")
    return 0


def render(s: dict[str, Any], seeds: int, v: dict[str, str]) -> str:
    bins = list(COVERAGE_BINS)
    lines = [
        "# P4 scan simulator check",
        "",
        f"Generated by `scripts/simulate_scans.py` at code revision `{v['code_revision']}` over {s['rooms']} accepted D0 rooms; "
        f"{seeds} free-running scans per room and occlusion type (seeds 0-{seeds - 1}) and one scan aimed at each coverage bin (seed 1000, "
        f"{BIN_ATTEMPTS} redraws). Every built sample ({s['contract_checked_samples']}) passed `contract_v0.validate_sample`.",
        "",
        "## Free-running scans: coverage bin by occlusion type",
        "",
        "| Type | " + " | ".join(bins) + " | other outcomes |",
        "| --- | " + " | ".join("---:" for _ in bins) + " | --- |",
    ]
    for kind, counts in s["free_running_scans_by_type"].items():
        other = {k: n for k, n in counts.items() if k not in bins}
        lines.append(f"| {kind} | " + " | ".join(str(counts.get(b, 0)) for b in bins) + f" | {other or 'none'} |")
    lines += [
        "",
        "![coverage histogram](figures/p4/p4_coverage_histogram.png)",
        "",
        "## Rooms where a scan aimed at each bin succeeded",
        "",
        "| Type | " + " | ".join(bins) + " |",
        "| --- | " + " | ".join("---:" for _ in bins) + " |",
    ]
    for kind, row in s["rooms_reaching_bin_by_type"].items():
        lines.append(f"| {kind} | " + " | ".join(f"{row[b]} / {s['rooms']}" for b in bins) + " |")
    lines += [
        "",
        f"- Rooms with at least one usable free-running sample: {s['rooms_with_any_sample']} of {s['rooms']}.",
        f"- Non-convex rooms (L-wing applicable in principle): {s['non_convex_rooms']}.",
        f"- Overlays for {len(s['figures'])} rooms in `reports/figures/p4/`"
        + (f"; over {MAX_FIGURE_BYTES} bytes: {s['figures_over_limit']}" if s["figures_over_limit"] else "") + ".",
        "",
        "## Method",
        "",
        "- Rays (360 per full turn, 5-8 m range) stop at every segment of the 1.1 m slice, furniture included; hits are observed wall cells, "
        "swept free space is observed free cells. Missing wall hides one observed wall and 1 m around it; doorway is one scanner 0.5 m inside "
        "a wall with a 120 degree view; L-wing hides a concave wing (10-50 % of the area) of a non-convex room.",
        "- Coverage = observed outer-wall length / total outer-wall length; bins high [0.65, 1], mid [0.40, 0.65), low [0.15, 0.40). "
        "Thresholds were fixed in code before this run.",
        "- The grid frame comes from the scan only: anchor at the centroid of observed free space, heading from observed surfaces. "
        "`clips_canvas` means the room does not fit the 12.8 m canvas around that anchor; such samples are refused, not clipped.",
        f"- Versions: {', '.join(f'{k} {val}' for k, val in v.items() if k != 'code_revision')}.",
        "- Data: AcousticRooms, CC BY 4.0 (see `DATA_LICENSE.md`). Figures are derived from it.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
