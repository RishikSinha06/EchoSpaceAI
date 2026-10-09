# P8 EchoFusion check

Date: 2026-10-09. Base: `2a9e814` (P7 on main). Branch: `module-8-echofusion`.
Machine: CPU only (no CUDA). Not an AcousticRooms result.

**Synthetic gate: PASSED** (`scripts/check_echofusion.py`, summary in
`p8_synthetic_gate_summary.json`). The real-data study (fold 0 and beyond) has
not been run; it needs a GPU.

## What the gate tests

Synthetic rooms hide the far wall from the scan; every RIR carries a
band-limited echo from that wall (image source) plus a diffuse tail. Geometry
alone cannot place the wall; the echo delay can. A, P, S, C and D train on
the same frozen set (200 training, 12 validation, 16 test rooms; 4 masks each;
small configuration, 40 epochs, CPU) and are scored on unseen rooms.

Pass criteria, fixed before the first run and never changed:
D - A >= 0.05 occupancy IoU at K = 8; S - mean room shape >= 0.05;
P - A <= 0.02; all runs complete.

| Variant | Test IoU K=1 | Test IoU K=8 | Best epoch |
| --- | ---: | ---: | ---: |
| A (geometry only) | 0.686 | 0.686 | 12 |
| P (positions, no audio) | 0.685 | 0.685 | 12 |
| S (audio only) | 0.923 | 0.939 | 39 |
| C (pooled fusion) | 0.934 | 0.966 | 39 |
| D (EchoFusion) | 0.744 | 0.779 | 40 |
| mean room shape | | 0.687 | |

D beats A by 0.093, S beats the mean shape by 0.252, P does not beat A.
Every audio variant improves from K = 1 to K = 8.

## Findings to carry into the real study

1. **The plan's acoustic encoder cannot see echo timing.** Convolutions then
   global average pooling give the same features for an echo at sample 300
   and at 700 (difference 1e-6); on band-limited echoes it learns echo delay
   with R^2 = -0.02. A normalised-time input channel fixes it (R^2 = 0.99;
   attention pooling over time also 0.99, not adopted). EchoFusion uses the
   time channel by default (documented deviation, regression test).
2. **D learns slowest.** Validation IoU stays at the geometry-only plateau
   (~0.74) for ~30 epochs; S and C leave it after ~25, D only after ~35, and
   was still improving at its last epoch. Audio variants show a plateau then a
   sudden jump, so short runs or aggressive early stopping can hide the audio
   benefit. The plan's patience of 15 with 100 epochs should be kept, and
   cross-attention may need tuning (learning rate, warm-up) before Experiment C.
3. On this task the simple pooled fusion C beats cross-attention D. This is a
   synthetic task with one hidden wall; it says nothing yet about real rooms,
   but Experiment C must be run with an adequate budget for D.

## Attempts before the passing run (all recorded)

| Attempt | Change | Result |
| --- | --- | --- |
| 1 | 48 training rooms, 20 epochs, one-sample echo spikes | failed: all variants 0.72-0.73, mean shape 0.71 |
| 2 | + time channel in the encoder | failed: same picture |
| 3 | echoes made band-limited with a diffuse tail (one-sample spikes are unlike resampled real RIRs; no encoder learned from them) | failed: same picture |
| diagnostic | S and A only, 200 rooms, 40 epochs | S 0.939 vs A 0.686: S learns after a ~25-epoch plateau |
| 4 (this) | gate resized to 200 rooms, 40 epochs | passed |

The test design (echo shape, size, epochs) changed after failures; the pass
thresholds did not.

## Real-data plumbing (smoke only)

On real fold-0 data (fixed val/test masks, 40 online training draws) the
freeze, the CLI and every variant ran end to end: the full-size D (2.09 M
parameters) trained, selected a checkpoint on validation and scored the 898
test masks at K = 1, 2, 4, 8. Two training batches only; the numbers are not
results. Speed on this CPU: D ~30 samples/s, A ~40.

## Tests

13 P8 tests pass (synthetic data only). Not run here: CUDA, deterministic
CUDA algorithms with the new layers, and any real-data training.
