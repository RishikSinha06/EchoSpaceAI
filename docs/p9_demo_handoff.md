# P9 small training demonstration

P9's full scope is training comparisons and statistics across folds and seeds.
This user-requested demonstration runs A/P/S/C/D on a smaller verified dataset,
using the full P8 architecture. It does not complete the 75-run research study.

The completed local run is documented in `reports/p9_demo_check.md`: training
and export verification passed, while the S learning check failed and acoustic
gain was not established. The demonstration preserves those results.

## Reproduce

```powershell
python scripts/train_demo.py freeze
python scripts/train_demo.py run
python scripts/report_demo.py
```

Settings live in `configs/p9_demo.json`. The default destinations are
`data/cache/p9_demo/` and `artifacts/p9_demo/`; create new destinations with
`--out-dir` for another run. Nonempty freeze/run/report destinations are refused.
All data and checkpoints stay local and ignored by Git. Existing P7/P8 outputs
are preserved. No new dependency is required beyond the approved model extras.

## Subset and comparison

- Fold 0, selection seed 0, 20 train / 5 validation / 5 test physical-room groups.
- A seeded SHA-256 ordering selects groups separately within the original P6
  partitions, before reading targets or audio. All rooms and masks belonging to
  selected groups are retained. Original folds/spec files are never overwritten.
- The supplied P6 fixed-mask archive is the source. Every selected sample
  undergoes the original P6 checksum/contract check. P8 then checks geometry,
  poses and finite nested waveforms before writing the immutable audio freeze.
- All five models share identical samples, training settings and seed. Training
  draws random nested K from 1..8; validation selects at requested K=8; held-out
  test uses requested K=1/2/4/8. Actual K is capped at available verified pairs.
- Fixed-mask training has no scan redraw or augmentation. The demonstration
  retains P8's architecture, optimizer and loss. The budget is 40 epochs,
  batch 16, three warm-up epochs and patience 10, rather than P8's full defaults.
- Only validation selects checkpoints. Test results are reported after training;
  poor test performance is retained and does not trigger retuning.

The reporter verifies all frozen samples, source hashes, checkpoint checksums,
run identity and identical configurations. It reloads each selected checkpoint,
recomputes every held-out score at every K and exports contract-valid probability
NPZs bound to the source sample, grid, transform, checkpoint and frozen checksum.
The replay tolerance is 1e-4. For a CPU replay use `--device cpu` and a fresh
report destination through `--out-dir`.

## Demonstration and statistics

Open `artifacts/p9_demo/demonstration/index.html` for the results, learning curves
and ten seeded, score-independent held-out examples comparing observed inputs,
targets and all five models. Plots show raw probabilities; observed cells are
excluded from loss, so model outputs there need not reproduce measurements.
The existing viewer's assumed-height extrusion is not recovered 3D geometry.

`summary.json` includes sample-weighted runner metrics, equal-room-weight
metrics, coverage/occlusion breakdowns and exploratory paired D-minus-A
occupancy differences. Bootstrap resampling uses whole room groups; with only
five test groups and one seed its interval is diagnostic, not evidence of a
general acoustic benefit. A training-only mean shape supplies the S learning
check using the same sample-weighted evaluation domain. Test comparison never
changes checkpoint selection or hyperparameters.

Full multi-fold/multi-seed studies, hyperparameter search and broader research
metrics remain subsequent work. This module supplies a reproducible small
demonstration and records failures alongside successes.
