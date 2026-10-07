"""P6 step 1: one cache file per accepted D0 room (geometry + every processed RIR).

    python scripts/build_room_cache.py              # all accepted rooms, skips rooms already cached
    python scripts/build_room_cache.py --rooms Bedrooms_idx_3 --rebuild

Reads the D0 manifest, reports/d0_duplicates.json and the archives (read
only). Writes data/processed/rooms/<room_id>.npz and index.json (ignored by
Git). Same dataset root rules as audit_data.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from audit_data import natural_key, versions  # noqa: E402
from echospace.acoustics import AcousticError, RirConfig, load_audited_candidates, process_candidate  # noqa: E402
from echospace.acoustics.rir import PROCESSOR_VERSION  # noqa: E402
from echospace.data import RoomCacheError, load_room_cache, save_room_cache  # noqa: E402
from echospace.io.adapters import acousticrooms as ar  # noqa: E402
from echospace.scans import RoomGeometry, RoomGeometryError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help=f"dataset root (default: ${ar.ROOT_ENV_VAR} or data/raw)")
    parser.add_argument("--manifest", default="data/manifests/acousticrooms_index.jsonl")
    parser.add_argument("--duplicates", default="reports/d0_duplicates.json")
    parser.add_argument("--index-cache", default="data/cache/acousticrooms_index.json")
    parser.add_argument("--out", default="data/processed/rooms")
    parser.add_argument("--rooms", nargs="+")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config = RirConfig()
    root = ar.resolve_root(args.root)
    with (REPO / args.manifest).open(encoding="utf-8") as handle:
        audits = {row["room_id"]: row for row in map(json.loads, handle) if row["accepted"]}
    room_group = json.loads((REPO / args.duplicates).read_text(encoding="utf-8"))["room_group"]
    index = ar.load_cached_index(root, REPO / args.index_cache)
    if index is None:
        raise SystemExit("index cache missing or stale; run scripts/audit_data.py first")
    out = REPO / args.out
    room_ids = args.rooms or sorted(audits, key=natural_key)
    status: dict[str, dict] = {}
    started = time.time()
    with ar.ArchiveReader(root) as reader, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for position, room_id in enumerate(room_ids, 1):
            tick = time.time()
            target = out / f"{room_id}.npz"
            if target.exists() and not args.rebuild:
                try:
                    cache = load_room_cache(target, config)
                    status[room_id] = {"status": "cached", "n_rirs": len(cache.rir_ids)}
                    continue
                except RoomCacheError:
                    pass  # rebuild a stale or corrupt file
            room = index.rooms[room_id]
            try:
                geometry = RoomGeometry.from_mesh(ar.load_mesh(reader, room.mesh))
                candidates = load_audited_candidates(reader, room, audits[room_id])
            except (RoomGeometryError, AcousticError) as error:
                status[room_id] = {"status": "refused", "reason": str(error)}
                print(f"[{position}/{len(room_ids)}] {room_id}: REFUSED {error}", flush=True)
                continue
            ids, positions, waveforms, wav_sha, meta_sha, failed = [], [], [], [], [], []
            for candidate in candidates:
                try:
                    processed = process_candidate(reader, candidate, config)
                except AcousticError as error:
                    failed.append({"rir_id": candidate.rir_id, "reason": str(error)})
                    continue
                ids.append(candidate.rir_id)
                positions.append(candidate.positions_scene_m)
                waveforms.append(processed.waveform)
                wav_sha.append(processed.metadata["source_wav_sha256"])
                meta_sha.append(candidate.metadata_sha256)
            meta = {
                "audit_record_sha256": candidates[0].audit_sha256 if candidates else None,
                "geometry_signature": audits[room_id]["geometry_signature"],
                "rir_config": config.to_dict(),
                "processor_version": PROCESSOR_VERSION,
                "metadata_sha256": meta_sha,
                "wav_sha256": wav_sha,
                "failed_rirs": failed,
                "footprint_m2": round(geometry.footprint.area, 4),
                "convex": geometry.is_convex,
                "category": room.category,
            }
            save_room_cache(target, room_id, int(room_group[room_id]), geometry, ids,
                            np.asarray(positions).reshape(-1, 2, 3), np.asarray(waveforms).reshape(-1, config.window_samples), meta)
            status[room_id] = {"status": "built", "n_rirs": len(ids), "failed": len(failed)}
            print(f"[{position}/{len(room_ids)}] {room_id}: {len(ids)} RIRs ({len(failed)} failed) "
                  f"[{time.time() - tick:.1f}s]", flush=True)
    summary = {
        "rooms": len(room_ids),
        "built_or_cached": sum(s["status"] != "refused" for s in status.values()),
        "refused": {k: v["reason"] for k, v in status.items() if v["status"] == "refused"},
        "rirs": sum(s.get("n_rirs", 0) for s in status.values()),
        "failed_rirs": sum(s.get("failed", 0) for s in status.values()),
        "rir_config": config.to_dict(),
        "processor_version": PROCESSOR_VERSION,
        "versions": versions(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps({"summary": summary, "rooms": status}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
