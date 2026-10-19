# Software sensor lab exit gate - 2026-10-19

Status: PASS for the bounded software simulation/replay module. Physical hardware
validation and a high-order acoustic solver are outside this gate.

## Checks actually run

| Check | Result |
| --- | --- |
| Full local Python CPU regression | 174 passed; 2 CUDA resume checks deliberately deselected; 91.09 seconds |
| Focused simulation + existing inference after final engine changes | 32 passed; 6.59 seconds |
| Browser grid / GLB geometry agreement | PASS |
| Matched comparison and exact threshold maps | PASS |
| JavaScript syntax checks | PASS |
| Vite production build | PASS; existing large Three.js bundle warning remains |
| Real registered A/P/D inference gate | 12 predictions across rectangle, L room, sweep faults and exact replay; PASS |
| Replay tensor identity and original-route prediction identity | PASS for Apartments_idx_0__doorway__low__1000000 |
| GLB magic and RoomScene schema on all 12 results | PASS |
| Observation NPZ excludes targets/valid-cell labels | PASS |
| Default ZIP excludes reference targets and clean theoretical RIR | PASS |
| Virtual and real-replay CLI capture | PASS |
| Live browser default/L-room/multiple-pair capture | PASS; 2 accepted pairs at K=2 |
| Live browser A/D comparison | PASS; synthetic diagnostic banner visible |
| Live browser replay Capture / Next / Play / Pause | PASS |

The full real gate is reproducible with `scripts/check_simulation.py`. Local
metadata and bundles are under `artifacts/simulation_review`; dataset/weight
payloads and generated artifacts are intentionally excluded from Git.

## Analytical and safety evidence

Tests cover metric direct/reflection path lengths, finite visible reflections in
an L room, vertical paths, fractional delays within half a sample, deterministic
capture, unknown-space semantics, range clipping and pair eligibility. Device
checks cover sweep recording length, deconvolution, clipping, finite normalized
signals and explicit uncompensated timing-fault metadata.

Tests mutate full targets and valid-cell labels without changing any model input.
Exact replay preserves original identity/checksum and model tensor bytes. No
new sample becomes training-eligible. Immutable observation payload references
survive later scanner movement while retained in the bounded capture cache.

Schema/config checks cover bounds, nonfinite values, unknown fields, duplicate
pairs and replay fault rejection. API checks cover stale capture refusal,
unknown samples/models, seen-room rejection, bundles and reference separation.

## Interpretation and limits

A successful model call on a virtual room is integration evidence, not a result
about learned acoustic benefit or real-room readiness. No new performance claim
or production-readiness claim is made. In particular, first-order specular RIRs
lack the full reverberant field of AcousticRooms or a physical room. No real
hardware, high-order acoustic solver, furniture, SLAM accumulation, user upload
adapter or real-room calibration is included.

This work was isolated from the paused study checkout. The original training
STOP_AFTER_EPOCH marker remains present and training was not resumed.
