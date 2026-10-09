# P7 real baseline foundation check

Date: 2026-10-09. Base: `d87557d` (P6). Branch: `module-7-models`.

**P7 foundation gate: PASS.** B0 wall extrapolation, A geometry-only U-Net and
P geometry-plus-positions control ran on the same complete fold-0 freeze and
produced validation and held-out test predictions. Loss and metrics use valid
unobserved cells (`M=0`) only. This is one seed with fixed masks and K=8, without
augmentation; it is not the five-fold research benchmark or a fusion result.

## Data provenance

The supplied `eval_samples.zip` is 870,007,107 bytes, SHA-256
`65e92e2ee25eb6b6f8d7b9ed5a256e02304fa01e83eff6c6e1f29244782d07b6`.
Every one of its 4,642 samples passed the original P6 content checksum and
contract checks. Training uses only masks belonging to fold-0 training rooms.
No room or duplicate group crosses splits.

| Split | Masks | Rooms | Room groups |
| --- | ---: | ---: | ---: |
| Train | 3,256 | 171 | 99 |
| Validation | 488 | 27 | 18 |
| Test | 898 | 51 | 30 |

Frozen checksum:
`63a2cd68d43f3490cee38dc20631d88039be9ccd01b9331045c47aeda57cec11`.
The accompanying [machine-readable summary](p7_real_baseline_summary.json)
records the committed fold/spec hashes, configuration, checkpoint hashes and
gate metrics. P3 targets and boundary conventions were not regenerated.

The archive supplies fixed masks but no room caches. The explicit
`fixed-masks` training route freezes all masks assigned to training rooms;
there are no online scan redraws or training augmentations in this run.

## Training and results

Seed 0; batch 8; AdamW learning rate 0.001 and weight decay 0.0001; maximum
100 epochs; early-stopping patience 10 and minimum improvement 0.0001.
Both heads receive equally weighted masked BCE + Dice. CUDA AMP is used during
training; validation and exported predictions use float32. Validation alone
selects checkpoints. Test targets never select model weights.

| Variant | Parameters | Selected epoch | Completed epochs | Test loss | Occupancy IoU | Boundary IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0 | — | 0 | 2.754830 | 0.168427 | 0.033253 |
| A | 488,722 | 8 | 18 | 0.563483 | 0.660023 | 0.314967 |
| P | 510,898 | 7 | 17 | 0.562507 | 0.658535 | 0.321829 |

IoUs are sample averages on valid unobserved cells at probability threshold
0.5. A and P are close in this one-fold run; these numbers do not establish a
significant position benefit. Neither model consumes RIR waveforms.

Environment: Python 3.11.9, NumPy 1.26.4, PyTorch 2.5.1+cu121, CUDA 12.1,
RTX 4060 Laptop GPU, two CPU threads. Deterministic algorithms and seeds are
recorded in the runner; numerical identity across hardware is not promised.

## Verification

- Final full Python suite: **114 passed in 182.26 seconds**.
- All 4,642 frozen samples were rechecked for integrity and eligibility.
- All three completed run manifests bind to the same freeze and source hashes.
- All 4,158 exported predictions (488 validation + 898 test per variant)
  passed the P6 prediction contract and sample/grid/frame/checkpoint bindings.
- Selected A/P checkpoints were loaded on CPU and replayed on the first,
  middle and last test indices. Maximum probability differences from CUDA
  exports were 0.000001967 for A and 0.000001907 for P, below the 0.0001
  cross-device tolerance. This replay checks six predictions, not every export.
- Twenty uniformly selected test samples (seed 0, independent of metrics)
  were visually inspected for both occupancy and boundary on all ten pages.
  Grid registration and orientation were consistent in these examples.

Visual failures are retained: B0 can extend walls across the canvas and
overfill open outlines. A/P can miss hidden wings, smooth corners, predict
spurious extensions and produce interior boundary responses. Their raw
outputs on observed cells can be weak or speckled because those cells are
excluded from the loss. A consumer should preserve the measured geometry
layer separately. These predictions do not demonstrate full room recovery.

## Reproduce and locate outputs

```powershell
python scripts/train_baselines.py freeze --training-source fixed-masks --out-dir data/cache/p7_fold0_fixed
python scripts/train_baselines.py run --frozen data/cache/p7_fold0_fixed/manifest.json --out-dir artifacts/p7_fold0_fixed --device cuda
python scripts/check_real_baselines.py
python scripts/review_baseline_predictions.py
```

Use a fresh destination for a new freeze/run; nonempty destinations are
refused. The checker defaults to the paths above. Local outputs are under
`artifacts/p7_fold0_fixed/`: run manifests, best/last checkpoints, prediction
NPZs, `gate_summary.json` and `review/`. Frozen inputs are under
`data/cache/p7_fold0_fixed/`, with extracted P6 data under
`data/processed/eval_samples/`. Dataset binaries, weights and prediction
files remain ignored and are not distributed through Git.

P7's common-sample baseline prerequisite is satisfied for this fold. Full
five-fold evaluation, additional seeds and fusion remain subsequent work.
