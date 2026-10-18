# P10 inference gate

**Gate: PASS.** The inference service replays all 898 fixed test masks (51 rooms, fold 0) for `echofusion-d-fold0-seed0` through the same path the API uses, on cpu, at code revision `73ff57a`.

| Check | Result |
| --- | --- |
| all samples verified and scored | pass |
| k8 matches recorded within tolerance | pass |
| glb matches grid geometry | pass |

## Reproduction of the recorded score

| K | rooms | masks | occupancy IoU | boundary IoU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 51 | 898 | 0.751518 | 0.437769 |
| 2 | 51 | 898 | 0.752108 | 0.439234 |
| 4 | 51 | 898 | 0.749908 | 0.438031 |
| 8 | 51 | 898 | 0.749712 | 0.438172 |
| recorded K = 8 | 51 | 898 | 0.749712 | 0.438172 |

Absolute difference at K = 8: occupancy 1.42e-10, boundary 2.64e-09 (tolerance 0.001).

Room-weighted: each room's masks are averaged, then rooms count equally. IoU is on valid unobserved cells (M = 0) at the registered threshold, from the raw network output, exactly as P9 scored it. Only K = 8 was recorded at training time; the other K values are reported, not gated.

## Exports and runtime

- Every K = 8 grid document validated as `echospace.grid.v2`; 36 GLBs (every 25th sample) were parsed back and their bounds matched the grid geometry.
- Model load 2.92 s; per reconstruction median 0.0227 s, p95 0.0261 s on cpu.
- Failures: 0.

## Scope

- one fold, one seed; fold-0 test pool had earlier project use; exploratory, no general acoustic-gain claim.
- These are test-pool numbers reproduced by the service, not new evidence of acoustic benefit; D vs P remains unresolved (see `docs/architecture_research_2026-10-17.md`).
- The service only accepts committed evaluation samples; arbitrary scans or recordings are out of scope for H1.
