# Small-data model iterations

Training completed 2026-10-13; report recorded 2026-10-15.
Base: `3d1ad01`. Protocol: `configs/p9_iterations.json`.

**Eight training fits complete; integrity and prediction gate PASS. No tested
change improved the confirmed validation result over the reference A model.**
Perfect reconstruction was not achieved. The selected architecture remains
geometry-only A, with deployment seed fixed to 0 before holdout scoring.

## Protocol and results

Training used 735 masks / 33 rooms / 20 groups; validation used 148 masks /
eight rooms / five groups. All trials used the same verified data. The primary
selection metric was equal-room mean occupancy/boundary IoU on valid unobserved
cells at K=8, threshold 0.5. There was no augmentation in this fixed-mask study.

| Setting | Parameters | Seed-0 selected epoch | Completed epochs | Validation score |
| --- | ---: | ---: | ---: | ---: |
| A reference | 1,674,818 | 9 | 19 | 0.481735 |
| A compact | 355,234 | 26 | 36 | 0.439247 |
| A compact regularized | 355,234 | 11 | 21 | 0.440708 |
| A compact slower/longer | 355,234 | 23 | 38 | 0.417286 |
| C compact pooled fusion | 421,202 | 22 | 32 | 0.459128 |
| D compact EchoFusion | 429,842 | 18 | 28 | 0.465391 |

The two strongest settings repeated with seed 1. A scored 0.478765 (epoch 12,
22 epochs completed); D scored 0.437260 (epoch 14, 24 epochs completed).
Two-seed validation means were **A 0.480250, D 0.451326**. Both finalists
received the same 40-epoch maximum and patience 10. The separately declared
slower/longer trial had a 70-epoch maximum and patience 15. All fits stopped
early. Shrinking the models or extending training did not improve the selected
validation objective in this study.

Trial test evaluation was disabled and guarded by a regression test. The
selection file was locked before final holdout scoring and its checksum stayed
unchanged during evaluation. No test result changed the selected setting,
checkpoint, seed or threshold.

## Final holdout assessment

The tuning-round holdout contains 174 masks / nine rooms / five groups, excluding
the previous P9 demo's test groups. The broader original fold-0 test pool was
evaluated earlier in P7: this is exploratory, not a pristine project-wide test.
It differs from the earlier P9 demo holdout, so its metrics are not a direct
improvement comparison with the earlier 0.650 occupancy result.

| Selected A setting | Room occupancy IoU | Room boundary IoU | Mean of heads |
| --- | ---: | ---: | ---: |
| Seed 0, deployment | 0.664467 | 0.266580 | 0.465524 |
| Seed 1, confirmation | 0.638678 | 0.237829 | 0.438253 |

Deployment checkpoint:
`artifacts/p9_iterations/A_reference/seed0/best.pt`, SHA-256
`0861687e82716a01b58e30d1ca76e6951108763808bbb5c819a187ab24f22862`.
The selection was validation-based; the seed-0 deployment rule was preset.

## Verification and review

- Full Python suite: **130 passed in 211.62 seconds**.
- All frozen sample checksums and all eight completed trial manifests verified.
- All **348 predictions** passed the prediction contract and exported identity,
  grid, transform, frozen-data and checkpoint bindings.
- CPU replay checked first/middle/last holdout examples for both A checkpoints.
  Maximum probability errors were 0.000003278 and 0.000006229, below 0.0001.
- Validation curves and six uniformly sampled holdout examples were reviewed
  for occupancy and boundary. Both seeds retain missed details, rounded or
  undersized extents and occasional spurious extensions. Observed-cell raw
  probabilities are not trained to reproduce measurements and must remain
  separate from the measured layer. No acoustic benefit is established here.

Frozen checksum:
`7744fb4cfb9f206cb8fae261c67565031b61b687bc1e6e1ad0413fb860c765c1`.
See `reports/p9_iterations_summary.json` for the locked selection and metrics,
and `docs/p9_iterations_handoff.md` for reproducible commands. Local browser
demonstration: `artifacts/p9_iterations/final/index.html`. Data, checkpoints
and real-data figures remain ignored by Git. Next larger-data training should
use the full fold and fresh augmented scans, with model selection on validation.
