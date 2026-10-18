# P11 Stage II implementation and original-checkpoint diagnostics

## Gate and boundaries

**PASS: implementation and frozen original-P9 diagnostics.**
**Not complete: the entire conditional Stage II research program.**
The active dual study has not frozen its final selection/results. C/S full-study
training, final dual-study controls, retrained window sweeps and independent
measured-room validation remain pending. No production-readiness claim.

This module branches from main `0cd95cc`. The running training checkout/source
and original P9 checkpoint/data artifacts were left untouched. No new datasets,
dependencies, GPU fits or paid services were started. No weights/arrays are in Git.

## Completed experiment

- Original P9 fold 0, seed 0; A/P/D checkpoint hashes verified against its
  frozen selection lock and immutable sample manifest.
- 898 test scans, 51 room IDs, **30 physical-room groups**. Repeated masks and
  related rooms are not independent confidence-interval replicates.
- 22 conditions/K/model combinations, 19,756 sample predictions.
- Correct A/P/D at K=1/2/4/8; D at K=2 under zero/swapped audio, 20 dB noise,
  +/-6 dB gain, +/-3-sample delay, 40/60 ms truncation and bounded pose stress.
- Calibration fitted only on the 27 validation rooms at K=2; fixed scalar
  temperatures reused on test and all conditions, threshold retained at 0.5.
- All scoring uses valid unobserved cells. Equal-room means and physical-group
  paired bootstrap intervals, 2,000 draws, seed 0. Previously explored test pool.

## Main K=2 results (all masks)

| Model / audio | Missing occupancy IoU | Hidden-wall F1 @ 0.20 m | Boundary distance (m) |
| --- | ---: | ---: | ---: |
| A | 0.728179 | 0.687052 | 1.067600 |
| P | 0.732222 | 0.681841 | 1.145918 |
| D, correct | 0.752108 | 0.707489 | 0.930823 |
| D, swapped | 0.752055 | 0.707616 | 0.930759 |
| D, zero waveform | 0.754249 | 0.707340 | 0.931468 |

D-A occupancy change is **+0.02393** (exploratory group-bootstrap interval
[+0.00891, +0.04101]); D-P is +0.01989 [+0.00244, +0.03841]. These architecture
comparisons do not identify waveform benefit: swapping audio changes D by
-0.000053 [-0.000292, +0.000215], and zeroing audio changes it by
+0.002141 [-0.000487, +0.007255]. The old checkpoint provides little evidence
of useful waveform dependence. Report this null/negative finding directly.

### Retrospective primary-plan slice

K=2, mid coverage, viewpoint/missing_wall contains 193 scans from 49 room IDs
and 30 physical groups. A/P/D occupancy IoU is 0.769450 / 0.775985 / 0.785858.
D-A is +0.01641 with interval [-0.00227, +0.03752]; hidden-wall F1 change is
-0.00244 [-0.01992, +0.01397]. This does not establish a reliable improvement
in the planned condition. It is retrospective, not preregistered. Mid coverage
does not guarantee exact 50% observed-wall coverage.

### Calibration and robustness

D validation selects temperature 2.0 for both heads. At correct-audio K=2,
occupancy Brier goes from 0.013681 to 0.012432 and boundary Brier from
0.008066 to 0.007271. This is a basic scalar calibration result on the explored
population; it is not a validated uncertainty/rejection system.

Most waveform perturbations barely change the old model. This must not be sold
as proven acoustic robustness when it may reflect weak audio use. Pose stress
changes occupancy IoU to 0.749465. Shorter windows are zero-padded truncations
of the existing 80 ms inputs, not independently retrained window models.

## Figures and failure cases

Local outputs in `artifacts/p11_final_fold0/`: `report.md`, `summary.json`,
`conditions.png`, `coverage_occlusion.png`, `negative_cases_1.png`,
`negative_cases_2.png`, and `figure_provenance.json`; per-condition rows and
swap donor IDs remain available for audit.

The numerical worst-case list includes empty/tiny hidden interior targets.
Some high-coverage cases have no hidden positive interior cells; any false
positive then gives IoU zero. To avoid misleading pictures, the figures rank
cases deterministically by D K=2 IoU, require at least 20 hidden interior cells,
and show six distinct rooms with observation/A/D/truth. All masks remain in
numerical summaries; the support filter is for diagnostic figures only.
Figures carry AcousticRooms/Meta CC BY 4.0 attribution and remain local.

## Verification

- CPU regression: **169 passed**, two CUDA-only resume cases deselected to
  leave the ongoing GPU study alone.
- Final focused P11 checks: **19 passed**, including a later orchestration
  test proving all C/S fits lock before any C/S test scoring.
- Both future continuation commands correctly reject the unfinished parent
  before starting GPU work or creating new experiment outputs.
- All evaluation source hashes remained unchanged during the final run.
- Scripts compile; coverage/occlusion and negative-case figures rendered;
  the strata plot and first negative-case page were visually inspected.

The C/S continuation and completed-dual-study evaluator are implemented and
synthetically checked, but have not run against a completed real parent study.
Their full empirical gates therefore remain pending.

## Next dependent gates

1. Finish the current dual study, freeze/check final identities and results.
2. Train the full C/S continuation; lock all fits before evaluating.
3. Run matched final-study controls and per-seed room-group effects.
4. Train separate window models before claiming a trained window sweep.
5. Audit user-supplied measured rooms; evaluate calibration and rejection under
   new device/room conditions, then validate release/runtime/rollback packaging.

No automatic promotion or merge. Teammate review is required.
