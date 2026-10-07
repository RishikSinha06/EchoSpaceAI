"""P6 step 2: fixed evaluation masks for every cached room, written once.

    python scripts/make_eval_masks.py               # resumes: rooms already done are kept
    python scripts/make_eval_masks.py --rooms Bedrooms_idx_3

Writes contract samples to data/processed/eval_samples/ (manifest.jsonl,
samples/*.npz; ignored by Git) and the committed list with checksums and
per-room feasibility to data/splits/eval_masks.json. Refuses to overwrite an
existing data/splits/eval_masks.json unless --replace is given: test masks
must not be regenerated silently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from audit_data import natural_key, versions  # noqa: E402
from echospace.acoustics import RirConfig  # noqa: E402
from echospace.acoustics.rir import PROCESSOR_VERSION  # noqa: E402
from echospace.data import load_room_cache, masks_for_room  # noqa: E402
from echospace.data.evalmasks import EVAL_FORMAT, K_REQUIRED, MAX_SEEDS, PER_BIN, SEED_BASE, spec_checksum, write_sample  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rooms-dir", default="data/processed/rooms")
    parser.add_argument("--out", default="data/processed/eval_samples")
    parser.add_argument("--spec", default="data/splits/eval_masks.json")
    parser.add_argument("--rooms", nargs="+")
    parser.add_argument("--replace", action="store_true", help="allow replacing an existing committed spec")
    args = parser.parse_args()

    spec_path = REPO / args.spec
    if spec_path.exists() and not args.replace and not args.rooms:
        raise SystemExit(f"{args.spec} exists; fixed test masks are not regenerated (use --replace deliberately)")
    config = RirConfig()
    rooms_dir, out = REPO / args.rooms_dir, REPO / args.out
    partial = out / "rooms"
    partial.mkdir(parents=True, exist_ok=True)
    room_ids = args.rooms or sorted((p.stem for p in rooms_dir.glob("*.npz")), key=natural_key)
    started = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for position, room_id in enumerate(room_ids, 1):
            done = partial / f"{room_id}.json"
            if done.exists() and not args.rooms:
                continue
            tick = time.time()
            cache = load_room_cache(rooms_dir / f"{room_id}.npz", config)
            samples, feasibility = masks_for_room(cache, config)
            entries = [write_sample(sample, out) for sample in samples]
            done.write_text(json.dumps({"room_id": room_id, "entries": entries, "feasibility": feasibility,
                                        "records": [s.record for s in samples]}) + "\n", encoding="utf-8")
            per_type = Counter(e["occlusion_type"] for e in entries)
            print(f"[{position}/{len(room_ids)}] {room_id}: {len(entries)} masks {dict(per_type)} [{time.time() - tick:.1f}s]", flush=True)

    if args.rooms:
        print("partial run (--rooms): spec not written")
        return 0
    entries, feasibility, records = [], {}, []
    for room_id in sorted((p.stem for p in partial.glob("*.json")), key=natural_key):
        part = json.loads((partial / f"{room_id}.json").read_text(encoding="utf-8"))
        entries += part["entries"]
        records += part["records"]
        feasibility[room_id] = part["feasibility"]
    with (out / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    spec = {
        "format": EVAL_FORMAT,
        "k_required": K_REQUIRED,
        "per_bin": PER_BIN,
        "seed_base": SEED_BASE,
        "max_seeds": MAX_SEEDS,
        "p5_processor_version": PROCESSOR_VERSION,
        "p5_rir_config": config.to_dict(),
        "rooms": len(feasibility),
        "rooms_with_masks": len({e["room_id"] for e in entries}),
        "samples": sorted(entries, key=lambda e: e["sample_id"]),
        "feasibility": feasibility,
        "versions": versions(),
    }
    spec["checksum"] = spec_checksum(spec)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(entries)} masks over {spec['rooms_with_masks']} of {spec['rooms']} rooms; spec {args.spec}; "
          f"{time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
