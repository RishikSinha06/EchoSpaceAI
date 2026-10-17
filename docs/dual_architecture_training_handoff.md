# EchoFusion and WallFusion training study

The user authorized the complete local study on 17 October 2026: both full models,
five physical-room-group folds, three seeds, geometry/positions controls, selected
ablations, and no paid compute. Independent measured rooms will be supplied later.
This is an internal research study with a reproducible training workflow, not a
claim that the experimental model is production validated or novel.

## Fixed protocol

Source: `configs/dual_architecture_study.json`.

| Phase | Runs | Data |
| --- | ---: | --- |
| Pilot | 21 | Seeded fold-0 subset: 20 train groups, 5 validation groups, 2 augmented draw sets; test never scored |
| Full model comparison | 30 | EchoFusion D and WallFusion D, 5 folds x 3 seeds |
| Controls | 45 | EchoFusion A/P and WallFusion P, same 5 folds x 3 seeds |
| Full ablations | 6 | Fold 0 x 3 seeds, temporal-only WallFusion and single-scale WallFusion |

Total: **102 fits** including the pilot. Every full fit uses its fold's entire
eligible training partition, with ten precomputed augmented draw sets and 20 views
per room per draw. The draw sets cycle during optimization; they are not fresh
independent rooms or unlimited online augmentation. For fold 0 that is 171 rooms,
99 room groups and 34,200 augmented training observations.

The same fold-0 freeze used in the earlier study is reused after checks. Folds
1-4 get separate source-bound, journaled freezes. Train/validation/test ownership
is checked against the committed physical-room-group split; both architectures
see identical arrays and random-K schedules for a given seed and epoch.

Batch 8, AdamW 3e-4, weight decay 1e-4, five-epoch warmup, cosine schedule,
AMP, clipping at 1, max 100 epochs, patience 15. Pilot max 40, patience 10.
Primary selection is equal-room mean occupancy/boundary IoU on valid M=0 at K=8.
Both heads use the existing masked BCE+Dice loss. No new dependency was added.

All scheduled fits finish and checkpoint identities are locked before final test
scoring at K=1/2/4/8. The complete study does not tune architectures on its test
scores. The pilot is a technical/development check; it is not a separate discovery
of optimal hyperparameters. Architecture settings were fixed before pilot results.
The score table must report null and negative changes as well as positive ones.

## Experimental WallFusion model

`src/echospace/models/wallfusion.py` implements a versioned `WallFusion` family:

- The existing U-Net spatial backbone, skip connections and two dense output heads.
- A learned temporal encoder with local time tokens and source/mic Fourier encoding.
- A separate first-order infinite-plane evidence branch. Each 32x32 grid point
  proposes 32 candidate horizontal wall normals; full 3D source-image path lengths
  determine emission-relative delays, nominal c=343 m/s.
- Physical evidence uses the signed waveform and three fixed absolute-envelope
  smoothing scales (5/17/65 samples), sampled at exact original sample times.
  A trainable confidence/projection follows alignment. Learned temporal features
  stay in the residual attention route; the physical branch does not backpropagate
  through a large interpolated CNN feature bank.
- Evidence aligns across valid pairs for each shared candidate plane before
  orientation aggregation, with a null term and +/-0.5 ms tolerance samples.
- Geometry-conditioned gates inject the evidence at 16x16 and 8x8.
- Positions control P withholds all waveform content. It retains the same modules
  and parameter count as D, including parameters inactive without acoustic evidence.
- Invalid rows are sanitized before encoding. Pair order is irrelevant. A valid
  null attention token permits K=0 without NaNs; this alone does not establish a
  calibrated or trained no-audio deployment fallback.

The fixed-envelope branch was chosen after profiling a learned-feature sampling
prototype that incurred expensive deterministic gather gradients. That initial
prototype was not trained on validation or test data. Its profiling artifacts
are preserved separately. The finalized model uses under 2 million parameters;
exact counts and measured memory/speed are recorded by the driver.

Physical limitations: infinite planes do not establish finite wall-segment support
or specular visibility. Floor/ceiling/furniture/high-order echoes can match spurious
hypotheses. Sidedness, window bounds and learned/null residual routes mitigate this
but do not solve it. No hidden target walls enter the operator. Sound speed is an
explicit nominal assumption; actual sensor calibration and simulator validation
are still required before physical accuracy claims.

Implementation differs from the broader research proposal: no continuous plane
refinement, explicit finite-segment visibility model, signed-distance head,
calibrated uncertainty head, learned pose-uncertainty model, or full acoustic
renderer is included. These are potential later ablations, not delivered features.

## Running and recovery

```powershell
python scripts/train_dual_study.py --mode profile
python scripts/train_dual_study.py --mode all --resume
```

Local outputs: `artifacts/dual_study_20261017/`. Progress is written atomically to
`status.json` and appended to `events.jsonl`. Each fit has `run.json`, `last.pt`,
and `best.pt`. Full-fold probabilities are not automatically packaged as a release.
On Windows the active process requests inhibition of idle system sleep, releasing
the request on exit. It does not change the power plan, force the display to stay
on, or override closing the lid, explicit sleep/hibernate, shutdown or power loss.

Preparation writes an atomic NPZ followed by a metadata/hash journal entry. Resume
verifies every committed sample and skips completed deterministic simulation jobs.
A truncated final journal line can be regenerated; changed complete samples fail.
An interrupted packed sidecar is preserved under a unique name before rebuilding.
Interrupted training resumes optimizer/scheduler/scaler and CPU/CUDA RNG states.
An initial failed epoch with no checkpoint is preserved and restarts from the seed.

The study binds model/training/data Python source and protocol hashes. Changing
implementation during a run requires a new study version; do not bypass identity
checks. All Python/CUDA resume tests require exact equality. Deterministic gather
and integer pooling avoid unsupported CUDA adaptive-pooling/grid-sampling backward.

To stop at a safe checkpoint, create `STOP_AFTER_EPOCH` inside the study directory.
The driver records a failed/stopped stage after saving the epoch. Remove that exact
file before an explicit resume. No unattended automatic retries or paid services.

## Evaluation outputs and interpretation

- `selection.json`: all checkpoint/data identities locked before test scoring.
- `full/fold*/MODEL/seed*/test_K*.json`: per-mask rows and sample-weighted metrics.
- `quality_K8.json`: missing-region boundary F1 at 0.20/0.40 m, symmetric mean nearest
  boundary distance, and uncalibrated occupancy/boundary Brier scores.
- Boundary sets are restricted to valid M=0; when only one set is empty the distance
  penalty is the grid diagonal, and F1 is zero. Both empty gives distance zero/F1 one.
- `results.json`: room-averaged out-of-fold results per seed, exploratory paired
  room-group intervals for WallFusion D minus EchoFusion D, and pending release gates.

The prior dataset has already influenced project development. Internal cross-validation
does not create a fresh external test. Overlapping training partitions and repeated
seeds do not provide independent room populations; do not present per-seed intervals
as a comprehensive uncertainty estimate. No deployment promotion happens automatically.

## Still needed for a production release

Independent measured-room tests; robust RIR measurement/deconvolution, calibration and
device response handling; sensor/noise/pose/timing stress tests; calibrated confidence
and rejection behavior; CPU/GPU end-to-end latency and memory budgets; versioned
inference API/viewer integration; contract-verified deployment predictions; locked
runtime environment, a model card and release/rollback package. The user will provide
new real-room data later. Source and weights remain separate from raw data licences.
