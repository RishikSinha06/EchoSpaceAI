"""Verify completed P7 runs and all held-out predictions against P6 identities."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from echospace.contract_v0 import validate_prediction
from echospace.data import load_eval_spec, load_folds
from echospace.data.evalmasks import read_manifest
from echospace.training.frozen import FrozenDataset
from echospace.training.runner import load_checkpoint


def check(frozen, runs, eval_dir, folds_path, eval_spec_path):
    datasets = {split: FrozenDataset(frozen, split) for split in ("train", "val", "test")}
    manifest = datasets["test"].manifest
    source = manifest["source"]
    if source["kind"] != "P6_fixed_masks":
        raise ValueError("this full fixed-mask gate requires P6_fixed_masks source")
    folds, spec = load_folds(folds_path), load_eval_spec(eval_spec_path)
    if (source["eval_spec_checksum"] != spec["checksum"]
            or source["folds_sha256"] != hashlib.sha256(Path(folds_path).read_bytes()).hexdigest()):
        raise ValueError("frozen source differs from the committed P6 metadata")
    fold = folds["folds"][source["fold"]]
    records = read_manifest(eval_dir)
    expected = {s: {e["sample_id"] for e in spec["samples"] if e["room_id"] in fold[f"{s}_rooms"]}
                for s in datasets}
    for split, dataset in datasets.items():
        if {e["metadata"]["sample_id"] for e in dataset.entries} != expected[split]:
            raise ValueError("full fold samples are missing or assigned to the wrong split")
        for index in range(len(dataset)):
            _ = dataset[index]  # verify all frozen NPZs, not only replay examples
    results = {}
    common_config = None
    for variant in ("B0", "A", "P"):
        root = runs / variant
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if run["status"] != "complete" or run["frozen_checksum"] != manifest["checksum"]:
            raise ValueError(f"{variant}: incomplete run or different freeze")
        if common_config is None:
            common_config = run["config"]
        elif run["config"] != common_config:
            raise ValueError("baseline training settings/seeds differ")
        for relative, expected_sha in run["source_files_sha256"].items():
            if hashlib.sha256((REPO / "src/echospace" / relative).read_bytes()).hexdigest() != expected_sha:
                raise ValueError("model/training source changed since the recorded run")
        for split in datasets:
            if (set(run["split_sample_ids"][split]) != expected[split]
                    or len(run["split_sample_ids"][split]) != len(expected[split])):
                raise ValueError("baselines did not consume the common full fold")
        checkpoint_sha = None
        if variant != "B0":
            checkpoint_sha = hashlib.sha256((root / "best.pt").read_bytes()).hexdigest()
            if checkpoint_sha != run["checkpoint_sha256"]:
                raise ValueError("selected checkpoint checksum differs")
            model, payload = load_checkpoint(root / "best.pt", manifest["checksum"])
            model.eval()
            if payload["epoch"] != run["best_epoch"]:
                raise ValueError("selected checkpoint epoch differs")
        counts = {}
        replay_error = 0.0
        for split in ("val", "test"):
            directory = root / f"predictions_{split}"
            export = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            entries = {e["metadata"]["sample_id"]: e for e in datasets[split].entries}
            predictions = export["predictions"]
            if len(predictions) != len(entries) or {e["sample_id"] for e in predictions} != set(entries):
                raise ValueError("held-out predictions are missing or duplicated")
            if export["model"] != run["model"] or export["frozen_checksum"] != manifest["checksum"]:
                raise ValueError("prediction model/freeze binding differs")
            for i, prediction in enumerate(predictions):
                entry = entries[prediction["sample_id"]]
                path = (directory / prediction["prediction_path"]).resolve()
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError("prediction path escapes run directory")
                if hashlib.sha256(path.read_bytes()).hexdigest() != prediction["prediction_file_sha256"]:
                    raise ValueError("prediction file checksum differs")
                if prediction["sample_sha256"] != entry["sha256"] or prediction["checkpoint_sha256"] != checkpoint_sha:
                    raise ValueError("prediction sample/checkpoint binding differs")
                if any(prediction[key] != entry["metadata"][key] for key in ("room_id", "room_group", "spatial_transform")):
                    raise ValueError("prediction identity/transform differs from its sample")
                if prediction["grid"] != records[prediction["sample_id"]]["grid"]:
                    raise ValueError("prediction grid differs from the P6 sample")
                with np.load(path, allow_pickle=False) as data:
                    arrays = {key: data[key].copy() for key in data.files}
                validate_prediction(records[prediction["sample_id"]], arrays)
                if variant != "B0" and split == "test" and i in (0, len(predictions)//2, len(predictions)-1):
                    sample = datasets[split][i]
                    with torch.inference_mode():
                        output = model({k: v[None] for k, v in sample.items()})
                    for head in ("occupancy", "boundary"):
                        replay = output[f"{head}_logits"][0].sigmoid().numpy()
                        error = float(np.max(np.abs(replay - arrays[f"{head}_probability"])))
                        replay_error = max(replay_error, error)
                        if error > 1e-4:
                            raise ValueError("held-out prediction does not replay from the selected checkpoint")
            counts[split] = len(predictions)
        results[variant] = {"parameter_count": run["parameter_count"], "best_epoch": run.get("best_epoch"),
                            "epochs_completed": len(run["history"]), "stop_reason": run["stop_reason"],
                            "checkpoint_sha256": checkpoint_sha, "prediction_counts": counts,
                            "validation": run["val_metrics"], "test": run["test_metrics"],
                            "maximum_replay_error": replay_error}
    return {"gate": "PASS", "scope": f"complete P6 fold-{source['fold']} fixed-mask baseline foundation; one seed, K={source['k']}, no augmentation",
            "fold": source["fold"], "frozen_checksum": manifest["checksum"], "source": source, "training_config": common_config,
            "split_counts": {s: len(d) for s, d in datasets.items()}, "variants": results,
            "notes": ["not five-fold research results; metrics are sample-averaged on valid unobserved cells",
                      "test targets did not select checkpoints; source and prediction bindings verified",
                      "CPU replay tolerance 1e-4 permits CUDA/CPU float32 numerical differences"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=REPO / "data/cache/p7_fold0_fixed/manifest.json")
    parser.add_argument("--runs", type=Path, default=REPO / "artifacts/p7_fold0_fixed")
    parser.add_argument("--eval-dir", type=Path, default=REPO / "data/processed/eval_samples")
    parser.add_argument("--folds", type=Path, default=REPO / "data/splits/folds.json")
    parser.add_argument("--eval-spec", type=Path, default=REPO / "data/splits/eval_masks.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = check(args.frozen, args.runs, args.eval_dir, args.folds, args.eval_spec)
    (args.runs / "gate_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
