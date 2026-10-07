"""P6 step 3: room-group folds over the rooms that have fixed evaluation masks.

    python scripts/make_splits.py

Only rooms with at least one evaluation mask take part: every group must be
tested exactly once, and a room without masks cannot be tested. Reads
data/splits/eval_masks.json and each room cache's metadata; writes the
committed data/splits/folds.json. Refuses to overwrite it without --replace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from echospace.data import RoomInfo, load_eval_spec, make_folds, save_folds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rooms-dir", default="data/processed/rooms")
    parser.add_argument("--spec", default="data/splits/eval_masks.json")
    parser.add_argument("--out", default="data/splits/folds.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    out = REPO / args.out
    if out.exists() and not args.replace:
        raise SystemExit(f"{args.out} exists; folds are not regenerated (use --replace deliberately)")
    spec = load_eval_spec(REPO / args.spec)
    with_masks = sorted({e["room_id"] for e in spec["samples"]})
    rooms = []
    for room_id in with_masks:
        with np.load(REPO / args.rooms_dir / f"{room_id}.npz", allow_pickle=False) as data:
            meta = json.loads(data["meta_utf8"].tobytes().decode("utf-8"))
        rooms.append(RoomInfo(room_id, int(meta["room_group"]), float(meta["footprint_m2"]), bool(meta["convex"])))
    folds = make_folds(rooms, n_folds=args.folds, inner_val_fraction=args.inner_val, seed=args.seed)
    folds["eval_spec_checksum"] = spec["checksum"]
    folds["excluded_rooms_without_masks"] = sorted(set(spec["feasibility"]) - set(with_masks))
    from echospace.data.splits import folds_checksum

    folds["checksum"] = folds_checksum(folds)
    save_folds(folds, out)
    for fold in folds["folds"]:
        print(f"fold {fold['fold']}: test {len(fold['test_rooms'])} rooms / {len(fold['test_groups'])} groups, "
              f"val {len(fold['val_rooms'])}/{len(fold['val_groups'])}, train {len(fold['train_rooms'])}/{len(fold['train_groups'])}")
    print(f"{len(rooms)} rooms, {len({r.room_group for r in rooms})} groups; excluded without masks: "
          f"{folds['excluded_rooms_without_masks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
