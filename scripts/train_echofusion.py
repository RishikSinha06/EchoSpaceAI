"""P8: freeze P6 samples with audio once, then train EchoFusion variants on them.

    # 1. freeze fold 0 (room caches: 10 epochs of online draws; or --training-source fixed-masks)
    python scripts/train_echofusion.py freeze --out-dir data/cache/p8_fold0 --fold 0 --workers 6

    # 2. train variants on the same freeze (GPU recommended)
    python scripts/train_echofusion.py run --frozen data/cache/p8_fold0/manifest.json \
        --out-dir artifacts/p8_fold0_seed0 --device cuda --seed 0 --variants A S C D

Settings come from configs/p8_echofusion.json; command-line overrides are
recorded in each run manifest. Destinations must be empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from echospace.training.audio_frozen import freeze_p8  # noqa: E402
from echospace.training.fusion_runner import FusionTrainConfig, run_fusion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "p8_echofusion.json")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="materialise P6 samples with waveforms once")
    freeze.add_argument("--out-dir", type=Path, required=True)
    freeze.add_argument("--fold", type=int, default=0)
    freeze.add_argument("--seed", type=int, default=0)
    freeze.add_argument("--training-source", choices=("online", "fixed-masks"))
    freeze.add_argument("--draw-epochs", type=int)
    freeze.add_argument("--samples-per-room", type=int)
    freeze.add_argument("--workers", type=int, default=1)
    freeze.add_argument("--limit-train", type=int, help="smoke freezes only; recorded")
    freeze.add_argument("--room-cache-dir", type=Path, default=REPO / "data" / "processed" / "rooms")
    freeze.add_argument("--eval-dir", type=Path, default=REPO / "data" / "processed" / "eval_samples")
    freeze.add_argument("--folds", type=Path, default=REPO / "data" / "splits" / "folds.json")
    freeze.add_argument("--eval-spec", type=Path, default=REPO / "data" / "splits" / "eval_masks.json")
    run = sub.add_parser("run", help="train and evaluate variants on an existing freeze")
    run.add_argument("--frozen", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--variants", nargs="+", choices=("A", "P", "S", "C", "D"))
    run.add_argument("--device", choices=("cpu", "cuda"))
    run.add_argument("--seed", type=int)
    run.add_argument("--epochs", type=int)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--workers", type=int)
    run.add_argument("--max-train-batches", type=int, help="smoke runs only; recorded")
    args = parser.parse_args()

    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if settings.get("format") != "p8.config.1":
        parser.error("unknown P8 configuration format")
    if args.command == "freeze":
        frozen = settings["freeze"]
        path = freeze_p8(args.out_dir, eval_dir=args.eval_dir, folds_path=args.folds, eval_spec_path=args.eval_spec,
                         fold=args.fold, training_source=args.training_source or frozen["training_source"],
                         room_cache_dir=args.room_cache_dir, seed=args.seed,
                         epochs=args.draw_epochs or frozen["draw_epochs"],
                         samples_per_room=args.samples_per_room or frozen["samples_per_room"],
                         workers=args.workers, limit_train=args.limit_train)
        print(path)
        return 0
    training = dict(settings["training"])
    for key, value in (("device", args.device), ("seed", args.seed), ("epochs", args.epochs),
                       ("batch_size", args.batch_size), ("workers", args.workers), ("max_train_batches", args.max_train_batches)):
        if value is not None:
            training[key] = value
    training["k_range"], training["eval_k"] = tuple(training["k_range"]), tuple(training["eval_k"])
    config = FusionTrainConfig(**training)
    for variant in args.variants or settings["variants"]:
        result = run_fusion(variant, args.frozen, args.out_dir / variant, config, settings["model"],
                            progress=lambda row, v=variant: print(json.dumps({"variant": v, "epoch": row["epoch"],
                                                                               "train_loss": round(row["train_loss"], 5),
                                                                               "val_iou": round(row["validation"]["occupancy_iou"], 5)}), flush=True))
        print(json.dumps({"variant": variant, "status": result["status"], "best_epoch": result["best_epoch"],
                          "test": {k: round(m["occupancy_iou"], 4) for k, m in result["test_metrics"].items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
