"""D0 follow-up: is there a peak at the predicted first reflection?

The plan's Stage 0 gate asks for direct-path agreement (checked by
audit_data.py) and a visible peak at the predicted first reflection for at
least one wall. For each accepted room this script mirrors the source across
two large flat surfaces, the floor and the longest straight wall of the 1.1 m
outline, predicts the reflection delay and asks whether the RIR has a peak
there that beats the same window at control times (see
``echospace.io.audit.reflection_hit``; chance is about 10 %).

It reuses the seeded line-of-sight pairs that audit_data.py stored in the
manifest, so it samples nothing new.

    python scripts/audit_reflections.py

Reads the manifest and the archives (read only), writes
``reports/d0_reflections.json``. Same dataset root rules as audit_data.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import shapely

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from audit_data import SLICE_HEIGHT_M, natural_key, versions  # noqa: E402
from echospace.io import audit_geometry as ag  # noqa: E402
from echospace.io.adapters import acousticrooms as ar  # noqa: E402
from echospace.io.audit import (  # noqa: E402
    REFLECTION_CONTROL_QUANTILE,
    REFLECTION_CONTROL_SHIFTS,
    REFLECTION_MIN_GAP_SAMPLES,
    REFLECTION_WINDOW_SAMPLES,
    direct_path_samples,
    mirror_across_plane,
    reflection_hit,
    wall_specular_point,
)

# Fixed before the run: a surface is visible in a room when at least half of
# at least 8 judgeable pairs show a peak at the predicted time.
MIN_JUDGED_PAIRS = 8
MIN_HIT_RATE = 0.5


def judge(hits: list[bool]) -> dict[str, Any]:
    rate = sum(hits) / len(hits) if hits else None
    return {
        "n_judged": len(hits),
        "n_hits": int(sum(hits)),
        "hit_rate": rate,
        "visible": bool(len(hits) >= MIN_JUDGED_PAIRS and rate is not None and rate >= MIN_HIT_RATE),
    }


def audit_room(reader: ar.ArchiveReader, room: ar.RoomEntry, record: dict[str, Any]) -> dict[str, Any]:
    up = record["up_axis"]
    horizontal = list(ag.horizontal_axes(up))
    floor, ceiling = record["mesh"]["floor"], record["mesh"]["ceiling"]
    mesh = ar.load_mesh(reader, room.mesh)
    footprint = ag.floor_slice(mesh, up, SLICE_HEIGHT_M).footprint
    wall_start, wall_end = ag.longest_wall(footprint)
    floor_point = np.zeros(3)
    floor_point[up] = floor
    floor_normal = np.zeros(3)
    floor_normal[up] = 1.0

    hits: dict[str, list[bool]] = {"floor": [], "wall": []}
    for source_id, receiver_id, *_rest, blocked in record["pair_offsets"]:
        if blocked:
            continue
        meta = ar.load_pair_metadata(reader, room.metadata[(source_id, receiver_id)])
        source, receiver = np.asarray(meta.source_xyz, float), np.asarray(meta.receiver_xyz, float)
        rir = ar.load_rir(reader, room.rirs[(source_id, receiver_id)])
        fs = rir.sample_rate_hz
        direct = direct_path_samples(source, receiver, fs)

        image = mirror_across_plane(source, floor_point, floor_normal)
        u = (source[up] - floor) / ((source[up] - floor) + (receiver[up] - floor))
        bounce = source[horizontal] + u * (receiver[horizontal] - source[horizontal])
        if shapely.contains_xy(footprint, bounce[0], bounce[1]):
            verdict = reflection_hit(rir.waveform, direct_path_samples(image, receiver, fs), direct)
            if verdict is not None:
                hits["floor"].append(verdict)

        spot = wall_specular_point(source, receiver, wall_start, wall_end, horizontal)
        if spot is not None and floor < spot[1] < ceiling:
            wall_point = np.zeros(3)
            wall_point[horizontal] = wall_start
            wall_normal = np.zeros(3)
            direction = wall_end - wall_start
            wall_normal[horizontal] = [-direction[1], direction[0]]
            image = mirror_across_plane(source, wall_point, wall_normal)
            verdict = reflection_hit(rir.waveform, direct_path_samples(image, receiver, fs), direct)
            if verdict is not None:
                hits["wall"].append(verdict)

    result = {surface: judge(values) for surface, values in hits.items()}
    result["wall_length_m"] = float(np.linalg.norm(wall_end - wall_start))
    result["any_visible"] = result["floor"]["visible"] or result["wall"]["visible"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help=f"dataset root (default: ${ar.ROOT_ENV_VAR} or data/raw)")
    parser.add_argument("--manifest", default="data/manifests/acousticrooms_index.jsonl")
    parser.add_argument("--index-cache", default="data/cache/acousticrooms_index.json")
    parser.add_argument("--out", default="reports/d0_reflections.json")
    args = parser.parse_args()

    root = ar.resolve_root(args.root)
    with (REPO / args.manifest).open(encoding="utf-8") as handle:
        records = {row["room_id"]: row for row in map(json.loads, handle) if row["accepted"]}
    index = ar.load_cached_index(root, REPO / args.index_cache)
    if index is None:
        raise SystemExit("index cache missing or stale; run scripts/audit_data.py first")

    rooms: dict[str, Any] = {}
    with ar.ArchiveReader(root) as reader, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # trimesh on degenerate triangles
        for position, room_id in enumerate(sorted(records, key=natural_key), 1):
            rooms[room_id] = audit_room(reader, index.rooms[room_id], records[room_id])
            r = rooms[room_id]
            print(f"[{position}/{len(records)}] {room_id}: floor {r['floor']['n_hits']}/{r['floor']['n_judged']}, "
                  f"wall {r['wall']['n_hits']}/{r['wall']['n_judged']} ({r['wall_length_m']:.1f} m)", flush=True)

    pooled = {s: [sum(r[s]["n_hits"] for r in rooms.values()), sum(r[s]["n_judged"] for r in rooms.values())] for s in ("floor", "wall")}
    summary = {
        "rooms": len(rooms),
        "rooms_floor_visible": sum(r["floor"]["visible"] for r in rooms.values()),
        "rooms_wall_visible": sum(r["wall"]["visible"] for r in rooms.values()),
        "rooms_any_visible": sum(r["any_visible"] for r in rooms.values()),
        "rooms_judgeable_any": sum(r["floor"]["n_judged"] >= MIN_JUDGED_PAIRS or r["wall"]["n_judged"] >= MIN_JUDGED_PAIRS for r in rooms.values()),
        "pooled_hit_rate": {s: (h / n if n else None) for s, (h, n) in pooled.items()},
        "pooled_pairs_judged": {s: n for s, (_h, n) in pooled.items()},
        "chance_hit_rate_approx": round(1.0 - REFLECTION_CONTROL_QUANTILE, 2),
        "rooms_not_visible": sorted((k for k, r in rooms.items() if not r["any_visible"]), key=natural_key),
        "gate_passed": None,
    }
    summary["gate_passed"] = summary["rooms_any_visible"] >= 1
    result = {
        "method": {
            "surfaces": ["floor (up = floor height)", f"longest straight edge of the {SLICE_HEIGHT_M} m outline, as a vertical wall"],
            "window_samples": REFLECTION_WINDOW_SAMPLES,
            "min_gap_from_direct_samples": REFLECTION_MIN_GAP_SAMPLES,
            "control_shifts_samples": list(REFLECTION_CONTROL_SHIFTS),
            "control_quantile": REFLECTION_CONTROL_QUANTILE,
            "room_visible_rule": f">= {MIN_HIT_RATE:.0%} hits on >= {MIN_JUDGED_PAIRS} judged pairs",
            "pairs": "line-of-sight pairs from the manifest's pair_offsets (seeded, 64 per room)",
        },
        "summary": summary,
        "rooms": rooms,
        "versions": versions(),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
