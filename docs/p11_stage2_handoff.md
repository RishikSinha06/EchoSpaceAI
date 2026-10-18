# P11 — Conditional Stage II

## Status and scope

P11 adds separate evaluation modules and continuation commands. It does not
modify the active dual-study implementation or its source/config fingerprints.
It does not declare the unfinished study or the project production ready.

The original P9 fold-0 A/P/D selection is already locked. The new CPU
`evaluate_stage2.py` diagnoses those checkpoints on their identical verified
fixed samples. Its results are retrospective and exploratory: this population
has already influenced project development. The ongoing five-fold dual study
has a separate selection/results lock and is not silently substituted here.

## Delivered

- `evaluation/controls.py`: correct audio, zero waveform with poses/validity
  retained, deterministic different-room-group swaps matched by actual K;
  post-preprocessing +/-6 dB gain, 20 dB noise, +/-3-sample timing shifts,
  40/60 ms truncation and bounded +/-5 cm pose stress restricted to observed free cells.
- `evaluation/engine.py`: raw missing-region occupancy/boundary IoU, hidden-wall
  F1 at 0.20/0.40 m and symmetric mean nearest boundary distance; raw and
  validation-temperature-calibrated Brier scores. No observed-cell scoring.
- `evaluation/stats.py`: equal-room aggregation, then paired physical-room-group
  bootstrap effects. Masks are not independent replicates; seeds stay separate.
- `evaluation/protocol.py`: selected checkpoint byte hashes, immutable frozen
  identity and validation-only temperature selection; completed-study coverage gate.
- `evaluate_stage2.py`: original P9 K=1/2/4/8 comparisons, matched D controls at
  K=2, per-coverage/occlusion tables, predetermined worst occupancy cases,
  primary-plan slice, calibration records and report/plot output.
- `evaluate_locked_study.py`: resumable CPU equivalent for all five completed
  core models across the final dual-study folds/seeds; D-A, D-P, WallFusion-P,
  architecture and intervention effects; refuses an unfinished parent study.
- `train_stage2_controls.py`: separate C/S full-fold training continuation using
  the parent backbone/optimizer recipe. All 30 fits (5 folds x 3 seeds x 2 models)
  lock before C/S test scoring. Refuses an unfinished or incompatible parent.

Inputs stay read only. Results are under new `artifacts/p11_*` directories.
No new dependency or dataset download is required. Existing model/audit/test
extras suffice. No API or viewer changes are required for this offline stage.

## Run the frozen original P9 diagnostics

From the P11 checkout:

```powershell
python scripts/evaluate_stage2.py --source-root "PATH/TO/TRAINING_CHECKOUT" --output artifacts/p11_original_fold0
```

Uses the original `artifacts/p9_full_fold0/{A,P,D}/best.pt`, selection lock,
`data/cache/p9_full_fold0/manifest.json` and its packed sidecar. Verifies
checkpoint hashes and source/frozen identity before scoring. Defaults to CPU,
two threads, batch 16; the current training GPU remains available.
Output must be empty; preserve previous evaluations rather than overwrite them.
A complete run writes `plan.json`, per-condition row/donor records,
validation-only calibration records, `summary.json`, `report.md`, and
`conditions.png`. Raw arrays and model binaries are not committed.

Temperature grid is fixed to 0.5/0.75/1/1.5/2/3, fitted at K=2 by equal-room
validation missing-cell Brier separately for each head, then reused unchanged
for all K/conditions. The occupancy/boundary threshold remains 0.5. No test
threshold or model tuning is performed. Calibration can fail to transfer;
co-report raw and calibrated Brier instead of assuming improvement.

## Continue after the current dual study completes

First preflight both commands; until final `selection.json` and `results.json`
exist and agree, they fail before training or creating output directories.

```powershell
python scripts/train_stage2_controls.py --source-root "PATH/TO/TRAINING_CHECKOUT" --mode preflight
python scripts/train_stage2_controls.py --source-root "PATH/TO/TRAINING_CHECKOUT" --mode train
python scripts/evaluate_locked_study.py --source-root "PATH/TO/TRAINING_CHECKOUT" --mode preflight
python scripts/evaluate_locked_study.py --source-root "PATH/TO/TRAINING_CHECKOUT" --mode evaluate
```

Use `--resume` with the same output/config/source to continue interrupted work.
Keep the P11 source/configuration fixed throughout a continuation. The training
runner saves optimizer/scheduler/scaler/RNG state each epoch; the evaluation
runner binds each completed record to a full plan SHA. To stop C/S at a checkpoint, create `STOP_AFTER_EPOCH` in its output directory;
remove that exact marker before explicit resume. Status is written after each epoch.
Do not edit the original
study, relabel interim checkpoints as final, or run concurrent GPU studies.
The C/S continuation has not been trained or performance validated yet.

## Protocol differences and interpretation

The supplied original plan's primary condition was K=2, 50% target coverage,
viewpoint/missing-wall, with validation-selected threshold. Current training
uses random K=1..8, K=8 checkpoint selection and fixed threshold 0.5. Report
these deviations explicitly. A retrospective K=2 / mid-coverage /
viewpoint+missing_wall slice does not recreate preregistration or guarantee
exact 50% wall coverage. If the slice is empty, report it empty.

Swapped donors come from different physical groups within the same held-out
partition, matched by actual K. Only waveforms change; target poses and observed
geometry stay fixed. Donor reuse is allowed, so this is not a balanced permutation.
Donor IDs are saved for inspection. This control tests sensitivity to mismatched
acoustics; it does not prove useful acoustic geometry reasoning.

Zero-waveform is not K=0 and is not a trained missing-audio deployment fallback.
Timing shifts are zero padded, exactly 0.1875 ms at 16 kHz. No circular wrap.
Window truncation preserves emission time, input shape and original peak scale.
It is **not a retrained window sweep**. The +/-5 cm pose condition rejects
perturbations leaving observed free space, so its effective distribution is
bounded and geometry-conditioned; some poses may remain unchanged after 32 tries.

Boundary sets use valid M=0 only. If both sets are empty, F1=1 and distance=0;
if only one is empty, F1=0 and distance is the grid diagonal. Distances measure
raster boundaries, not exact structural mesh error. A Brier improvement is not
calibrated uncertainty, reliable abstention or real-world readiness.

## Still conditional / not delivered by a report

1. Finish the active dual study and freeze its final selection/results.
2. Train full C/S using the gated continuation and compare the matched results.
3. Run the full locked-study diagnostics; report per-seed effects without
   treating overlapping folds/seeds as new independent room populations.
4. Train separate short-window models before making a window-sweep claim.
5. Validate calibration/rejection under independent device/room shifts.
6. Audit new measured rooms with paired partial scans, measured RIRs, known
   metric poses/transforms, timing/calibration metadata and ground-truth labels.
7. Validate runtime/environment, package selected weights with model/data
   provenance and rollback instructions before a production release.

External datasets remain disabled: DIFFRIR/RAF/FLAIR require separate audits;
SoundSpaces/Replica would be a separate simulated study. Paused downloads and
`data/raw.zip` remain untouched. H2 uploads/editing and H3 redesign are out of scope.

## Verification and review

`tests/test_stage2.py` checks deterministic interventions, matching donors,
no pose/mask/label mutation, exact SNR/timing behavior, missing-cell scoring,
validation-only calibration, room-level weighting and prerequisite/hash gates.
Run it through the existing CPU PyTorch CI job. Real results and any incomplete
gates are summarized in `reports/p11_stage2_check.md`.

One module branch, explicit source/doc/report staging, no weights/data in Git.
Teammate approval is required before merging. P10 PR #8 remains independently
reviewable; P11 branches from main and does not silently merge it.

### Render inspection figures

After the original P9 diagnostic run completes:

```powershell
python scripts/render_stage2_report.py --source-root "PATH/TO/TRAINING_CHECKOUT" --results artifacts/p11_original_fold0
```

This adds coverage/occlusion plots and six deterministic negative cases, one per
room, ranked by D K=2 occupancy IoU, with at least 20 hidden interior cells for
informative figures. The numerical report retains empty/tiny hidden-target cases. Each figure shows observation/A/D/truth in
the same scanner grid. Figure provenance records sample IDs, script and result
hashes; figures stay local with AcousticRooms CC BY 4.0 attribution.
