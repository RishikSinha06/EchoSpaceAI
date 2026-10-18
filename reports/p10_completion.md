# P10 completion verification — 2026-10-18

## Scope and outcome

Completed the original P10 handoff against main `0cd95cc` in an isolated
`module-10-completion` worktree. The active training source/configuration
was left untouched. A/P/D are the original P9 fold-0 seed-0 checkpoints,
not interim dual-study weights. WallFusion promotion remains conditional
on study selection and is not a blocked H1 feature.

## Checks

| Check | Outcome |
| --- | --- |
| CPU Python regression suite | 154 passed, 2 GPU-specific resume cases deselected |
| New inference tests | 12 passed, including schema/provenance, restart recovery and nested ZIP reading |
| Browser comparison checks | PASS: same sample/frame/K, observation-only panel, exact pre-rounding threshold maps |
| Browser/GLB geometry agreement | PASS on rotated contract fixture |
| Production viewer build | PASS; existing Three.js bundle exceeds 500 kB warning |
| A full replay | PASS: 898 scans / 51 rooms; occupancy 0.728179141, boundary 0.415570554 |
| P full replay | PASS: 898 scans / 51 rooms; occupancy 0.736872866, boundary 0.406416540 |
| D full replay after completion | PASS: 898 scans / 51 rooms; occupancy 0.749711865, boundary 0.438171729 |
| Actual API RoomScene | PASS: Draft 2020-12 contract validation |
| Actual source mesh endpoint | PASS: audited OBJ returned; 258 local inventory entries |
| Browser manual inspection | Comparison occupancy/boundary controls, real reconstruction, source mesh panel |
| Native Blender GLB inspection | PASS: scale/orientation/material groups; perspective and orthographic renders inspected |

The initial attempt to hide CUDA from the full suite still let two CUDA-only
cases enter through PyTorch availability detection, then fail with an invalid
device ID. The final command explicitly deselected those two cases. No GPU
resume guarantee is claimed from this CPU verification; the ongoing study
kept the GPU. CPU resume cases passed.

Replay details and measured CPU timings are in the A/P/D completion reports.
Timings were measured under concurrent local training/verification load and
are not a serving benchmark or production SLA.

## Native export evidence (local, excluded from Git)

`artifacts/p10_review/prediction.glb` was downloaded through the API for
`Apartments_idx_0__doorway__low__1000000`, D, K=8. Its same-room source OBJ
was fetched through the audited mesh endpoint. Blender 4.5.7 LTS imported the
actual GLB into the three named groups: floor, observed_walls, inferred_walls.
The official portable archive was SHA-256 verified:
`2311800b9761de4ee1af9ad0c5924b9d5e0fb2dba6c975649e4852612d8a6a40`.

Source-to-scene is `(x,y,z) -> (x,z,-y)`; Blender's glTF importer then maps
scene Y-up to native Z-up `(x,y,z) -> (x,-z,y)`, so source native XYZ is
unchanged in the final comparison. Both panels retain metres and differ
only by an explicit 10 m lateral layout offset. The source bounds are
3.234 x 7.200 x 2.900 m. The source render uses a labelled 1.7 m cutaway;
the full unmodified source mesh remains in the saved `.blend` reference.
The GLB retains its assumed 2.7 m display height. Teal measured walls and
amber inferred walls remain distinct. There is no ceiling-height recovery.

Perspective/top renders, `.blend`, source OBJ, downloaded grid/RoomScene,
and Blender script/settings are under `artifacts/p10_review/` locally.
No dataset or model binaries were added to Git. This difficult sample has
D unseen occupancy IoU 0.247 and boundary IoU 0.153; visual export correctness
does not imply accurate room completion on every input.

## Boundaries and provenance

RoomScene measured segments depend only on observed free/hit raster
interfaces. Both layers are quantized 2D projections at floor Y=0, not
exact raw wall geometry. Observation NPZ references exclude full targets;
truth appears only as an explicitly labelled benchmark reference.

Restart persistence retains at most 128 request JSON files and reconstructs
on demand with model/sample verification and deterministic-ID matching.
It is a single-process local cache, not a distributed production store.
The code map used for navigation had unresolved external endpoints and
collapsed relationships; conclusions above use source inspection and tests.

The test pool has previous project use. These scores remain exploratory,
not independent evidence of general acoustic gain or real-world readiness.
The next model can be registered only after the study selects it and its
own replay gate passes. New measured-room validation remains future work.

Merge this branch only after teammate review.
