"""Rebuild the fixed evaluation samples that the inference service may serve.

Runs P6's ``build_room_cache.py`` and ``make_eval_masks.py`` for the rooms of
one fold partition, then checks every rebuilt sample against the SHA-256 in
the committed ``data/splits/eval_masks.json``. The committed spec and folds
are only read, never written; a sample that does not match byte for byte is
reported and the script exits non-zero.

    python scripts/prepare_inference_samples.py --fold 0 --split test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from echospace.inference.samples import SampleStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--rooms-dir", default="data/processed/rooms")
    parser.add_argument("--eval-dir", default="data/processed/eval_samples")
    parser.add_argument("--spec", default="data/splits/eval_masks.json")
    parser.add_argument("--folds", default="data/splits/folds.json")
    parser.add_argument("--root", help="AcousticRooms root, passed to build_room_cache.py")
    args = parser.parse_args()

    folds = json.loads((REPO / args.folds).read_text(encoding="utf-8"))
    rooms = folds["folds"][args.fold][f"{args.split}_rooms"]
    started = time.time()
    cache_cmd = [sys.executable, str(REPO / "scripts" / "build_room_cache.py"), "--out", args.rooms_dir, "--rooms", *rooms]
    if args.root:
        cache_cmd[2:2] = ["--root", args.root]
    mask_cmd = [sys.executable, str(REPO / "scripts" / "make_eval_masks.py"), "--rooms-dir", args.rooms_dir,
                "--out", args.eval_dir, "--spec", args.spec, "--rooms", *rooms]
    for command in (cache_cmd, mask_cmd):
        print("running", Path(command[1]).name, f"for {len(rooms)} rooms ...", flush=True)
        subprocess.run(command, cwd=REPO, check=True)

    store = SampleStore(REPO / args.eval_dir, REPO / args.spec, REPO / args.folds)
    expected = [s for s in store.spec_samples() if s["room_id"] in set(rooms)]
    bad = []
    for entry in expected:
        try:
            store.load(entry["sample_id"])
        except Exception as exc:  # report every failure, not just the first
            bad.append(f"{entry['sample_id']}: {exc}")
    print(f"fold {args.fold} {args.split}: {len(rooms)} rooms, {len(expected) - len(bad)}/{len(expected)} samples "
          f"match the committed SHA-256 [{time.time() - started:.0f}s]")
    for line in bad[:20]:
        print("  MISMATCH", line)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
