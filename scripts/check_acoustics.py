"""P5 real integration gate on the tiny attributed AcousticRooms sample.

Uses P2's accepted audit records and P4 scans. All caches, arrays and plots go
under ignored local output paths. This subset does not estimate full-dataset K
availability or assign training splits.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from echospace.acoustics import (  # noqa: E402
    AugmentConfig, RirCache, RirConfig, augment_bundle, build_bundle,
    load_audited_candidates, process_candidate,
)
from echospace.contract_v0 import validate_sample  # noqa: E402
from echospace.acoustics.rir import PROCESSOR_VERSION  # noqa: E402
from echospace.geometry.frames import ACOUSTICROOMS_SOURCE_TO_SCENE  # noqa: E402
from echospace.io.adapters import acousticrooms as ar  # noqa: E402
from echospace.scans import RoomGeometry, ScanNotApplicable, ScanNotFeasible, build_scan_sample  # noqa: E402


def figure(path: Path, sample, bundle) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(sample.observed.observed_free, origin="lower", extent=(0, 64, 0, 64), cmap="Greys")
    for i in np.flatnonzero(bundle.valid):
        cells = sample.grid.scene_to_cell(bundle.positions_scene_m[i])
        axes[0].plot(cells[:, 0], cells[:, 1], ".-", label=bundle.rir_ids[i].split("/")[-1])
        time_ms = np.arange(bundle.waveforms.shape[1]) / 16
        axes[1].plot(time_ms, bundle.waveforms[i], linewidth=0.7)
    axes[0].set(title="Observed free cells and valid source/mic pairs", xlabel="column", ylabel="row")
    if bundle.valid.any():
        axes[0].legend(fontsize=6)
    axes[1].set(title="Emission-time RIR windows", xlabel="time (ms)", ylabel="PCM-scaled amplitude")
    fig.suptitle(f"{sample.room_id} / {sample.occlusion_type} / diagnostic only")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(samples_root: Path, audit_path: Path, output: Path, cache_path: Path, settings: dict) -> dict:
    if settings.get("processor_version") != PROCESSOR_VERSION:
        raise ValueError("configuration processor_version differs from the installed processor")
    config = RirConfig(**settings["preprocessing"])
    augmentation = AugmentConfig(**settings["augmentation"])
    audits = {entry["room_id"]: entry for entry in
              (json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    cache = RirCache(cache_path)
    output.mkdir(parents=True, exist_ok=True)
    outcomes = []
    loaded_pairs = []
    with ar.ArchiveReader(samples_root) as reader:
        for entry in json.loads((samples_root / "sample_index.json").read_text(encoding="utf-8")):
            room_id = entry["room_id"]
            directory = f"{entry['category']}/{room_id}"
            room = ar.RoomEntry(room_id, entry["category"], ar.MemberRef((f"{directory}/{room_id}.obj",)))
            for source, receiver in entry["pairs"]:
                stem = f"{directory}/S{source:03d}_R{receiver:03d}"
                room.metadata[(source, receiver)] = ar.MemberRef((f"{stem}.json",))
                room.rirs[(source, receiver)] = ar.MemberRef((f"{stem}_hybrid_IR.wav",))
            candidates = load_audited_candidates(reader, room, audits[room_id])

            def loader(candidate):
                return process_candidate(reader, candidate, config, cache)

            # Exercise every supplied WAV and cache hit, even when a scan hides a pose.
            for candidate in candidates:
                clean = loader(candidate)
                cached = loader(candidate)
                if not np.array_equal(clean.waveform, cached.waveform):
                    raise RuntimeError("cache round trip differs")
                loaded_pairs.append({"rir_id": candidate.rir_id, "metadata": clean.metadata})
            geometry = RoomGeometry.from_mesh(ar.load_mesh(reader, room.mesh))
            for kind in ("viewpoint", "missing_wall", "doorway", "l_wing"):
                try:
                    sample = build_scan_sample(geometry, room_id, kind, seed=0)
                except (ScanNotApplicable, ScanNotFeasible) as exc:
                    outcomes.append({"room_id": room_id, "type": kind, "status": "scan_unavailable", "reason": str(exc)})
                    continue
                key = f"{room_id}__{kind}__0"
                bundle = build_bundle(candidates, sample.grid, sample.observed.observed_free, loader,
                                      room_id=room_id, selection_key=key, k=8, config=config, pad_missing=True)
                arrays = sample.contract_arrays()
                arrays.update(bundle.contract_arrays())
                record = {
                    "schema_version": "0.1.0", "sample_id": key, "room_id": room_id, "dataset": "AcousticRooms",
                    "geometry_id": audits[room_id]["geometry_signature"], "rir_ids": list(bundle.rir_ids),
                    "rir_sample_rate_hz": config.sample_rate_hz, "sample_path": f"{key}.npz", "split": "unassigned",
                    "frame": {"description": "D0 source (x,y,z) -> scene (x,z,-y), metres",
                              "source_units_to_meters": 1, "source_to_scene": ACOUSTICROOMS_SOURCE_TO_SCENE.tolist()},
                    "grid": sample.grid.manifest_grid(),
                }
                validate_sample(record, arrays)
                for k in settings["bundle_sizes"]:
                    smaller = build_bundle(candidates, sample.grid, sample.observed.observed_free, loader,
                                           room_id=room_id, selection_key=key, k=k, config=config, pad_missing=True)
                    if smaller.rir_ids != bundle.rir_ids[:k] or not np.array_equal(smaller.waveforms, bundle.waveforms[:k]):
                        raise RuntimeError("K nesting differs")
                train = augment_bundle(bundle, sample.grid, sample.observed.observed_free, split="train", seed=0,
                                       sample_key=key, config=augmentation, sample_rate_hz=config.sample_rate_hz)
                train_arrays = sample.contract_arrays()
                train_arrays.update(train.contract_arrays())
                validate_sample({**record, "split": "train"}, train_arrays)
                for split in ("val", "test"):
                    evaluation = augment_bundle(bundle, sample.grid, sample.observed.observed_free, split=split,
                                                seed=0, sample_key=key, config=augmentation)
                    if not np.array_equal(evaluation.waveforms, bundle.waveforms):
                        raise RuntimeError("evaluation waveform was augmented")
                np.savez_compressed(output / record["sample_path"], **arrays)
                metadata = {"record": record, "selection": bundle.selection, "provenance": bundle.provenance}
                (output / f"{key}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                figure(output / f"{key}.png", sample, bundle)
                outcomes.append({"room_id": room_id, "type": kind, "status": "contract_pass",
                                 "valid_pairs": int(bundle.valid.sum()), "padded_rows": int((bundle.valid == 0).sum()),
                                 "selection": bundle.selection})
    passed = [entry for entry in outcomes if entry["status"] == "contract_pass"]
    gate = bool(passed) and all(any(x["room_id"] == entry["room_id"] and x["valid_pairs"] >= 2 for x in passed)
                               for entry in json.loads((samples_root / "sample_index.json").read_text()))
    report = {"gate": "PASS" if gate else "FAIL", "scope": "two attributed rooms; integration, not full dataset",
              "config": settings, "versions": {"python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__}, "processed_pairs": loaded_pairs, "scan_outcomes": outcomes,
              "notes": ["clean caches only; no assigned splits or training eligibility",
                        "D0 timing was sampled at room level; no direct-path verification claimed per loaded WAV",
                        "IR_norm is retained uninterpreted; full-dataset K-in-observed-free availability belongs to P6"]}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, default=REPO / "data/samples/acousticrooms")
    parser.add_argument("--audit-manifest", type=Path, default=REPO / "data/manifests/acousticrooms_index.jsonl")
    parser.add_argument("--config", type=Path, default=REPO / "configs/p5_acoustics.json")
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts/p5_check")
    parser.add_argument("--cache-dir", type=Path, default=REPO / "data/cache/p5_waveforms")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if settings["preprocessing"]["sample_rate_hz"] != 16000:
        parser.error("this integration gate's plots expect the initial 16 kHz configuration")
    result = run(args.samples_root, args.audit_manifest, args.out_dir, args.cache_dir, settings)
    print(json.dumps({"gate": result["gate"], "processed_pairs": len(result["processed_pairs"]),
                      "scan_outcomes": [{key: item[key] for key in ("room_id", "type", "status", "valid_pairs") if key in item}
                                        for item in result["scan_outcomes"]]}, indent=2))
    if result["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
