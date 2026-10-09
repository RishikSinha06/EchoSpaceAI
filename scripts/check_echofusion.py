"""P8 synthetic gate: does the acoustic path of EchoFusion carry information?

Synthetic rooms hide the far wall from the scan and put its echo in every
RIR (``echospace.training.synthetic_audio``). Variants A, P, S, C and D train
on identical frozen samples with a small configuration (CPU-friendly), then
are scored on held-out synthetic rooms at K = 1 and 8.

Sizing: 200 training rooms x 4 masks, 40 epochs. A first version with 48
rooms and 20 epochs failed for every variant; a diagnostic run showed that
audio-only S sits on a plateau for ~25 epochs and then learns the echo delay
(test IoU 0.94 vs 0.69 for A), so that version stopped too early.

Pass criteria, fixed before the first run and unchanged since:
- D beats A by >= 0.05 occupancy IoU at K = 8 (fusion uses the audio);
- S beats the mean-room-shape predictor by >= 0.05 (plan's learning gate);
- P does not beat A by more than 0.02 (positions alone add no hidden geometry here).
Not an AcousticRooms result.

    python scripts/check_echofusion.py [--out-dir artifacts/p8_check] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from echospace.training.audio_frozen import FrozenAudioDataset, write_freeze  # noqa: E402
from echospace.training.fusion_runner import FusionTrainConfig, mean_shape_baseline, run_fusion  # noqa: E402
from echospace.training.synthetic_audio import synthetic_splits  # noqa: E402

SMALL = {"width": 64, "channels": (16, 32, 64, 64), "audio_channels": (16, 32, 64, 64), "fusion_layers": 2,
         "heads": 4, "feedforward": 128, "dropout": 0.0}
MARGIN_FUSION, MARGIN_LEARNING, LEAK_TOLERANCE = 0.05, 0.05, 0.02


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts" / "p8_check")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    frozen = write_freeze(args.out_dir / "frozen", synthetic_splits(train_rooms=200, val_rooms=12, test_rooms=16),
                          {"kind": "synthetic_p8_gate", "note": "hidden far wall, echo in every RIR; not AcousticRooms"})
    config = FusionTrainConfig(epochs=args.epochs, batch_size=16, learning_rate=1e-3, warmup_epochs=2, patience=args.epochs,
                               eval_k=(1, 8), device=args.device, threads=8)
    results = {}
    for variant in ("A", "P", "S", "C", "D"):
        run = run_fusion(variant, frozen, args.out_dir / variant, config, SMALL,
                         progress=lambda row, v=variant: print(f"{v} epoch {row['epoch']}: val IoU {row['validation']['occupancy_iou']:.3f}", flush=True))
        results[variant] = {"status": run["status"], "parameters": run["parameter_count"], "best_epoch": run["best_epoch"],
                            **{k: round(m["occupancy_iou"], 4) for k, m in run["test_metrics"].items()}}
    train = FrozenAudioDataset(frozen, "train", k=8)
    test = FrozenAudioDataset(frozen, "test", k=8)
    trivial = round(mean_shape_baseline(train, test)["occupancy_iou"], 4)
    checks = {
        "fusion_uses_audio": results["D"]["K=8"] - results["A"]["K=8"] >= MARGIN_FUSION,
        "S_beats_mean_shape": results["S"]["K=8"] - trivial >= MARGIN_LEARNING,
        "P_does_not_beat_A": results["P"]["K=8"] - results["A"]["K=8"] <= LEAK_TOLERANCE,
        "all_complete": all(r["status"] == "complete" for r in results.values()),
    }
    summary = {"results": results, "mean_shape_occupancy_iou": trivial, "checks": checks, "passed": all(checks.values()),
               "margins": {"fusion": MARGIN_FUSION, "learning": MARGIN_LEARNING, "leak_tolerance": LEAK_TOLERANCE},
               "epochs": args.epochs, "model": SMALL}
    (args.out_dir / "gate_summary.json").write_text(json.dumps(summary, indent=1, default=list) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, default=list))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
