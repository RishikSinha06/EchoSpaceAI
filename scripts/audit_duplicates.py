"""D0 follow-up: which accepted AcousticRooms ids are the same physical room?

``audit_data.py`` counts independent rooms by ``geometry_signature``, an exact
hash of the mesh vertices. That catches material variants of one mesh but not
the same room shell re-exported with different furniture or tessellation. This
script compares the floor-slice footprints of all accepted rooms (translation,
90-degree turns and mirrors allowed) and groups rooms that share a signature
or a footprint. Room-level splits (module 6) must keep each group together.

    python scripts/audit_duplicates.py

Reads ``data/manifests/acousticrooms_index.jsonl`` and the archives (read only),
writes ``reports/d0_duplicates.json``. Same dataset root rules as audit_data.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from audit_data import CANVASES, SLICE_HEIGHT_M, natural_key, versions  # noqa: E402
from echospace.io import audit_geometry as ag  # noqa: E402
from echospace.io.adapters import acousticrooms as ar  # noqa: E402

# Footprints this close are treated as the same room. Chosen after seeing the
# data (the same-shell twins score >= 0.9999); the report gives the counts at
# looser thresholds too, so the choice is visible rather than hidden.
SAME_FOOTPRINT_IOU = 0.999
SENSITIVITY_IOU = (0.99, 0.95)
MAX_AREA_RATIO_GAP = 0.1  # skip pairs whose areas differ by more than 10 %


class Groups:
    """Union-find over room ids."""

    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)

    def clusters(self) -> list[list[str]]:
        found: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            found[self.find(item)].append(item)
        return sorted((sorted(group, key=natural_key) for group in found.values()), key=lambda g: natural_key(g[0]))


def group_rooms(records: dict[str, dict[str, Any]], pairs: list[tuple[str, str, float]], threshold: float) -> Groups:
    groups = Groups(sorted(records, key=natural_key))
    by_signature: dict[str, list[str]] = defaultdict(list)
    for room_id, record in records.items():
        by_signature[record["geometry_signature"]].append(room_id)
    for members in by_signature.values():
        for other in members[1:]:
            groups.join(members[0], other)
    for a, b, iou in pairs:
        if iou >= threshold:
            groups.join(a, b)
    return groups


def independent_counts(records: dict[str, dict[str, Any]], groups: Groups) -> dict[str, int]:
    counts = {"accepted": len({groups.find(r) for r in records})}
    for label in CANVASES:
        for rule in ("bbox", "anchor"):
            counts[f"{rule}_{label}"] = len({groups.find(r) for r, rec in records.items() if rec["eligibility"][label][rule]})
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help=f"dataset root (default: ${ar.ROOT_ENV_VAR} or data/raw)")
    parser.add_argument("--manifest", default="data/manifests/acousticrooms_index.jsonl")
    parser.add_argument("--index-cache", default="data/cache/acousticrooms_index.json")
    parser.add_argument("--out", default="reports/d0_duplicates.json")
    args = parser.parse_args()

    root = ar.resolve_root(args.root)
    with (REPO / args.manifest).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    records = {row["room_id"]: row for row in rows if row["accepted"]}
    index = ar.load_cached_index(root, REPO / args.index_cache)
    if index is None:
        raise SystemExit("index cache missing or stale; run scripts/audit_data.py first")

    footprints = {}
    with ar.ArchiveReader(root) as reader, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # trimesh on degenerate triangles
        for room_id in sorted(records, key=natural_key):
            mesh = ar.load_mesh(reader, index.rooms[room_id].mesh)
            footprints[room_id] = ag.floor_slice(mesh, records[room_id]["up_axis"], SLICE_HEIGHT_M).footprint

    ids = sorted(footprints, key=natural_key)
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            area_a, area_b = footprints[a].area, footprints[b].area
            if abs(area_a - area_b) > MAX_AREA_RATIO_GAP * max(area_a, area_b):
                continue
            iou = ag.footprint_iou(footprints[a], footprints[b])
            if iou >= min(SENSITIVITY_IOU):
                pairs.append((a, b, round(iou, 5)))

    by_signature = group_rooms(records, [], 1.1)
    same_room = group_rooms(records, pairs, SAME_FOOTPRINT_IOU)
    cluster_of = {room_id: next(i for i, g in enumerate(same_room.clusters()) if room_id in g) for room_id in ids}
    result = {
        "slice_height_m": SLICE_HEIGHT_M,
        "same_footprint_iou": SAME_FOOTPRINT_IOU,
        "accepted_rooms": len(records),
        "independent": {
            "by_signature": independent_counts(records, by_signature),
            f"by_signature_or_footprint_iou_{SAME_FOOTPRINT_IOU}": independent_counts(records, same_room),
            **{f"sensitivity_iou_{t}": independent_counts(records, group_rooms(records, pairs, t)) for t in SENSITIVITY_IOU},
        },
        "groups": [group for group in same_room.clusters() if len(group) > 1],
        "signature_groups": [group for group in by_signature.clusters() if len(group) > 1],
        "footprint_pairs_not_sharing_signature": [
            [a, b, iou] for a, b, iou in pairs if iou >= SAME_FOOTPRINT_IOU and by_signature.find(a) != by_signature.find(b)
        ],
        "room_group": {room_id: cluster_of[room_id] for room_id in ids},
        "versions": versions(),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")

    for name, counts in result["independent"].items():
        print(f"{name}: {counts}")
    print(f"groups of 2+ rooms: {len(result['groups'])} (signature alone: {len(result['signature_groups'])}); "
          f"footprint-only links: {len(result['footprint_pairs_not_sharing_signature'])}")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
