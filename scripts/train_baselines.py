"""Freeze P6 once, then train/run B0/A/P on identical samples."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from echospace.models import SpatialConfig, WallConfig
from echospace.training.frozen import freeze_p6
from echospace.training.runner import TrainConfig, run_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="materialize P6 items once; refuses overwrites")
    freeze.add_argument("--out-dir", type=Path, required=True)
    freeze.add_argument("--room-cache-dir", type=Path, default=REPO / "data/processed/rooms")
    freeze.add_argument("--eval-dir", type=Path, default=REPO / "data/processed/eval_samples")
    freeze.add_argument("--folds", type=Path, default=REPO / "data/splits/folds.json")
    freeze.add_argument("--eval-spec", type=Path, default=REPO / "data/splits/eval_masks.json")
    freeze.add_argument("--fold", type=int, default=0)
    freeze.add_argument("--seed", type=int, default=0)
    freeze.add_argument("--samples-per-room", type=int, default=16)
    freeze.add_argument("--limit-per-split", type=int)
    freeze.add_argument("--k-eval", type=int, default=8)
    train = sub.add_parser("run", help="run all selected baselines against an existing freeze")
    train.add_argument("--frozen", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--config", type=Path, default=REPO / "configs/p7_baselines.json")
    train.add_argument("--device", choices=("cpu", "cuda"))
    train.add_argument("--epochs", type=int)
    train.add_argument("--variants", nargs="+", choices=("B0", "A", "P"))
    args = parser.parse_args()
    if args.command == "freeze":
        try:
            path = freeze_p6(args.out_dir, args.room_cache_dir, args.eval_dir, args.folds, args.eval_spec,
                             args.fold, args.seed, args.samples_per_room, args.limit_per_split, args.k_eval)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        print(path)
    else:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        if settings.get("format") != "p7.config.1":
            parser.error("unknown P7 configuration format")
        config = settings["training"]
        for key in ("device", "epochs"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        for variant in args.variants or settings["variants"]:
            result = run_baseline(variant, args.frozen, args.out_dir / variant, TrainConfig(**config),
                                  SpatialConfig(**settings["spatial"]), WallConfig(**settings["wall"]))
            print(json.dumps({"variant": variant, "status": result["status"], "test": result["test_metrics"]}))


if __name__ == "__main__":
    main()
