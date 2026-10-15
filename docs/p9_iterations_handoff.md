# Small-data model training iterations

This bounded study optimizes balanced room-shape quality on the small verified
dataset. It does not promise perfect reconstruction or acoustic improvement.

## Preset search

`configs/p9_iterations.json` declares six seed-0 trials before training:
the full A reference, compact A, regularized compact A, a slower/longer compact
A, and regularized compact C/D. Compact models use width 64, encoder channels
16/32/64/64 and one fusion block. The reference remains the P8 architecture.
Regularized models use dropout 0.2 and weight decay 0.001. The longer trial
has lr 0.00015, maximum 70 epochs and patience 15; other trials keep the
P9 demo's 40-epoch budget and patience 10. Comparisons include compute budget
differences and do not isolate each hyperparameter's causal effect.

The primary objective is the equal-room average of occupancy and boundary
IoUs on valid unobserved validation cells, K=8, threshold 0.5. Both heads retain
the same masked BCE + Dice training loss. The top two seed-0 settings repeat
with seed 1. Their two-seed average validation score chooses the final setting;
deployment seed is fixed to 0, rather than choosing the luckiest seed.

All models train from scratch on the same 735 masks from 33 rooms / 20 groups.
Validation uses 148 masks from eight rooms / five groups. There is no training
augmentation or scan redraw. Waveform timing, normalization, poses and P6
target definitions remain unchanged. No new dependency is introduced.

## Evaluation separation

The study selects five holdout groups from original fold-0 test groups after
excluding the previous P9 demo's five test groups. Group ranking is a preset
seeded SHA-256 ordering independent of audio and target values. The final
holdout has 174 masks from nine rooms / five groups.

These groups are new to this tuning round, but the broader fold-0 test pool
was evaluated by the earlier P7 model. This is an exploratory assessment,
not a pristine project-wide research test set. Original folds/spec files are
never changed; no group crosses the study's partitions.

`FusionTrainConfig.evaluate_test=False` prevents all trial test dataset loads.
The regression test guards against constructing a test dataset while tuning.
`room_weighted_selection=True` and `selection_metric='mean_head_iou'` select
the balanced validation objective. Defaults preserve the original P8 runner's
occupancy selection and after-training test evaluation. Run/checkpoint fields
include `best_validation_score`; the legacy `best_validation_iou` field also
stores the selection score, so consult the recorded selection configuration.

The search writes `study.json` as `selection_locked` before final evaluation.
The reporter verifies all trial identities, deferral flags, checkpoint hashes,
validation scores and seed-confirmation arithmetic before scoring holdout
quality. It evaluates the A reference at seed 0 and the selected setting at
seeds 0/1. No setting, threshold or checkpoint is changed in response to those
test results. The selection file's checksum must stay unchanged during scoring.

## Commands and outputs

```powershell
python scripts/tune_small.py freeze
python scripts/tune_small.py search
python scripts/report_iterations.py
```

Paths are intentionally versioned separately from the previous demonstration:

- Frozen inputs: `data/cache/p9_iterations/`.
- All eight runs and locked selection: `artifacts/p9_iterations/`.
- Final metrics, checksum-bound prediction NPZs and browser demonstration:
  `artifacts/p9_iterations/final/`.

Nonempty destinations are refused. The search is bounded to six presets plus
two confirmations; it does not retry settings until test metrics look good.
Old runs are preserved. Data, weights and real-data plots stay ignored by Git.

The final reporter validates all frozen samples and all exported predictions,
then replays three deterministic examples per selected checkpoint on CPU with
1e-4 cross-device probability tolerance. The six plot examples use uniform
sampling with seed 0, independently of scores. Raw probabilities in observed
cells are not substitutes for measured geometry. The HTML report includes
validation curves and reference-versus-selected occupancy/boundary plots.

Five holdout groups and a heavily reused small validation partition limit
generalization claims. Full multi-fold/multi-seed studies and validation on
independently collected rooms remain future work.
