"""P10 gate: does the inference service reproduce the recorded held-out scores?

Replays every fixed test mask of the model's fold through exactly the path the
API uses (registry -> verified sample -> reconstruct -> exports), then
compares room-weighted IoU at K = 8 with the value recorded at training time.

Pass criteria, fixed before the first run:
- every sample verifies and every prediction satisfies the v0.1.0 contract;
- every grid document validates and every sampled GLB reproduces the grid geometry;
- |room-weighted IoU - recorded| <= 0.001 for both heads at K = 8 (training
  scored on CUDA with mixed precision; this replay runs in float32 on CPU).

    python scripts/check_inference.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from echospace.inference import export  # noqa: E402
from echospace.inference.predict import code_revision, reconstruct  # noqa: E402
from echospace.inference.registry import ModelRegistry  # noqa: E402
from echospace.inference.samples import SampleStore  # noqa: E402

TOLERANCE = 1e-3
GLB_EVERY = 25


def room_weighted(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average each room's masks, then weight rooms equally (as P9's room_metrics)."""
    rooms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rooms[row["room_id"]].append(row)
    result = {"rooms": len(rooms), "samples": len(rows)}
    for key in ("occupancy_iou", "boundary_iou"):
        result[key] = float(np.mean([np.mean([r[key] for r in rs]) for rs in rooms.values()]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="echofusion-d-fold0-seed0")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default="reports/p10_inference_check.md")
    parser.add_argument("--summary", default="reports/p10_inference_summary.json")
    args = parser.parse_args()

    store = SampleStore(REPO / "data/processed/eval_samples", REPO / "data/splits/eval_masks.json", REPO / "data/splits/folds.json")
    registry = ModelRegistry(REPO / "configs/models.json", REPO, REPO / "data/splits/folds.json", REPO / "data/splits/eval_masks.json")
    entry = registry.get(args.model)
    started = time.time()
    model = registry.load(args.model)
    load_s = time.time() - started
    revision = code_revision(REPO)
    rooms = set(store.folds["folds"][entry.fold][f"{args.split}_rooms"])
    wanted = [s for s in store.spec_samples() if s["room_id"] in rooms]
    rows: dict[int, list[dict[str, Any]]] = {k: [] for k in args.k}
    latencies, glb_checked, failures = [], 0, []
    for index, spec_entry in enumerate(wanted):
        try:
            sample = store.load(spec_entry["sample_id"])
        except Exception as exc:
            failures.append(f"{spec_entry['sample_id']}: {exc}")
            continue
        for k in args.k:
            tick = time.perf_counter()
            result = reconstruct(model, entry, sample, k, split=args.split, revision=revision, device=args.device)
            latencies.append(time.perf_counter() - tick)
            rows[k].append({"sample_id": sample.sample_id, "room_id": sample.record["room_id"], "k_used": result.k_used,
                            "occupancy_iou": result.metrics["occupancy_iou"], "boundary_iou": result.metrics["boundary_iou"]})
            if k == max(args.k):
                document = export.grid_document(result)
                export.validate_grid_v2(document)
                if index % GLB_EVERY == 0:
                    import trimesh

                    stats = export.geometry_stats(document)
                    scene = trimesh.load(trimesh.util.wrap_as_stream(export.glb_bytes(document, result.provenance)), file_type="glb")
                    if not np.allclose(scene.bounds, [stats["bounds_min"], stats["bounds_max"]], atol=1e-4):
                        failures.append(f"{sample.sample_id}: GLB bounds differ from the grid geometry")
                    glb_checked += 1
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(wanted)} samples [{time.time() - started:.0f}s]", flush=True)

    scores = {str(k): room_weighted(rows[k]) for k in args.k}
    recorded = entry.raw.get("recorded_test_k8_room_weighted", {})
    k8 = scores.get("8")
    deltas = {head: abs(k8[head] - recorded[head]) for head in ("occupancy_iou", "boundary_iou")} if k8 and recorded and args.split == "test" else {}
    checks = {
        "all_samples_verified_and_scored": not failures and all(len(rows[k]) == len(wanted) for k in args.k),
        "k8_matches_recorded_within_tolerance": bool(deltas) and all(d <= TOLERANCE for d in deltas.values()),
        "glb_matches_grid_geometry": glb_checked > 0 and not any("GLB" in f for f in failures),
    }
    passed = all(checks.values())
    summary = {
        "gate": "PASS" if passed else "FAIL",
        "model_id": entry.model_id,
        "checkpoint_sha256": entry.checkpoint_sha256,
        "split": args.split,
        "fold": entry.fold,
        "samples": len(wanted),
        "rooms": len(rooms),
        "device": args.device,
        "code_revision": revision,
        "checks": checks,
        "tolerance": TOLERANCE,
        "recorded_k8": recorded,
        "replayed": scores,
        "abs_delta_k8": deltas,
        "glb_checked": glb_checked,
        "model_load_s": round(load_s, 2),
        "latency_s": {"median": round(float(np.median(latencies)), 4), "p95": round(float(np.quantile(latencies, 0.95)), 4)} if latencies else None,
        "failures": failures[:50],
    }
    (REPO / args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")

    lines = [
        "# P10 inference gate",
        "",
        f"**Gate: {summary['gate']}.** The inference service replays all {len(wanted)} fixed {args.split} masks "
        f"({len(rooms)} rooms, fold {entry.fold}) for `{entry.model_id}` through the same path the API uses, "
        f"on {args.device}, at code revision `{revision}`.",
        "",
        "| Check | Result |",
        "| --- | --- |",
        *[f"| {name.replace('_', ' ')} | {'pass' if ok else 'FAIL'} |" for name, ok in checks.items()],
        "",
        "## Reproduction of the recorded score",
        "",
        "| K | rooms | masks | occupancy IoU | boundary IoU |",
        "| ---: | ---: | ---: | ---: | ---: |",
        *[f"| {k} | {s['rooms']} | {s['samples']} | {s['occupancy_iou']:.6f} | {s['boundary_iou']:.6f} |" for k, s in scores.items()],
    ]
    if deltas:
        lines += [
            f"| recorded K = 8 | {recorded['rooms']} | {recorded['samples']} | {recorded['occupancy_iou']:.6f} | {recorded['boundary_iou']:.6f} |",
            "",
            f"Absolute difference at K = 8: occupancy {deltas['occupancy_iou']:.2e}, boundary {deltas['boundary_iou']:.2e} "
            f"(tolerance {TOLERANCE}).",
        ]
    lines += [
        "",
        "Room-weighted: each room's masks are averaged, then rooms count equally. IoU is on valid unobserved cells "
        "(M = 0) at the registered threshold, from the raw network output, exactly as P9 scored it. Only K = 8 was "
        "recorded at training time; the other K values are reported, not gated.",
        "",
        "## Exports and runtime",
        "",
        f"- Every K = {max(args.k)} grid document validated as `echospace.grid.v2`; {glb_checked} GLBs (every {GLB_EVERY}th sample) "
        "were parsed back and their bounds matched the grid geometry.",
        f"- Model load {summary['model_load_s']} s; per reconstruction median {summary['latency_s']['median']} s, "
        f"p95 {summary['latency_s']['p95']} s on {args.device}." if latencies else "- No reconstructions ran.",
        f"- Failures: {len(failures)}." + ("" if not failures else " First: " + failures[0]),
        "",
        "## Scope",
        "",
        f"- {entry.raw.get('scope', '')}.",
        "- These are test-pool numbers reproduced by the service, not new evidence of acoustic benefit; D vs P remains unresolved (see `docs/architecture_research_2026-10-17.md`).",
        "- The service only accepts committed evaluation samples; arbitrary scans or recordings are out of scope for H1.",
        "",
    ]
    (REPO / args.report).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({k: summary[k] for k in ("gate", "checks", "replayed", "abs_delta_k8", "latency_s")}, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
