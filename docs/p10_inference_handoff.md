# P10 handoff: inference service, viewer integration and GLB export

P10 serves a trained checkpoint on verified held-out samples, shows the result
in the existing Vite/Three.js viewer, and exports grid JSON, GLB and a
provenance sidecar. It adds no training and changes no P1–P9 contract.

## What exists

| Piece | Where | What it guarantees |
| --- | --- | --- |
| Model registry | `configs/models.json`, `src/echospace/inference/registry.py` | A model is served only if its checkpoint SHA-256, format, family/variant, frozen dataset, folds file and evaluation spec all match the entry |
| Verified samples | `src/echospace/inference/samples.py`, `scripts/prepare_inference_samples.py` | Only committed evaluation samples, each checked against `data/splits/eval_masks.json`; inputs built by the same conversion P9 scored |
| Reconstruction | `src/echospace/inference/predict.py` | Raw probabilities kept apart from the display composition; metrics on valid unobserved cells only; deterministic ids; provenance for every result |
| Exports | `src/echospace/inference/export.py` | `echospace.grid.v2`, GLB in scene metres (Y up), contract prediction NPZ |
| API | `src/echospace/inference/api.py`, `apps/api/README.md` | `/health`, `/v1/models`, `/v1/samples`, `/v1/reconstructions[/{id}[/grid.json, scene.glb, provenance.json, prediction.npz]]` |
| Viewer | `apps/web/src/service.js`, `grid.js`, `main.js` | Reconstruction panel; measured vs inferred walls; truth outline; source/receiver markers; unseen-region IoU |
| Agreement check | `apps/web/scripts/check-grid.mjs`, `tests/fixtures/grid_v2_agreement.json` | Browser geometry equals the GLB exporter's on a rotated fixture |
| Gate | `scripts/check_inference.py`, `reports/p10_inference_check.md` | Replays every fold-0 test mask through the API path |

## Gate

**Passed** (`reports/p10_inference_check.md`). All 898 fold-0 test masks
(51 rooms) were rebuilt locally and matched the committed SHA-256 byte for
byte, then replayed through the service on CPU. Room-weighted K = 8 IoU is
occupancy 0.749712 and boundary 0.438172, equal to the training-time record
to within 3e-9 (tolerance 1e-3). Every prediction satisfies contract v0.1.0;
36 sampled GLBs reproduce the grid geometry. A reconstruction takes about
23 ms on CPU after a 3 s model load.

Observation, not gated: K = 1/2/4/8 give occupancy 0.752/0.752/0.750/0.750.
On this fold, more RIRs do not raise the served model's score, consistent
with the unresolved D-vs-P question in `docs/architecture_research_2026-10-17.md`.

## Decisions

- **Display composition.** Measured free cells are pasted inside (100 % lie
  inside the room on four training rooms). Measured wall-hit cells are drawn
  outside: only 48 % have their centre inside the room, so their occupancy is
  not measured, and drawing them outside places the wall within 0.2 m of the
  hit. This affects the drawing only; scores always use the raw output.
- **Observed walls.** An edge is observed when the cell outside it is a
  measured wall hit. The previous viewer rule needed both sides measured,
  which a scanner inside the room never achieves, so measured walls were drawn
  as inferred.
- **Seen rooms are refused.** A room in the model's training or validation
  partition returns 409 unless `allow_seen_rooms` is set, so the viewer cannot
  present a memorised room as a reconstruction.
- **grid v2.** P4 grids are rotated to the scan's dominant wall; v1 could only
  express axis-aligned grids, so v2 carries origin and axes in scene metres.
  v1 files still load.
- **Weights stay out of Git.** Put the original D checkpoint at
  `artifacts/models/echofusion_d_fold0_seed0/best.pt` (from
  `EchoSpace_EchoFusion_D.zip`). The registry reports it unavailable, with the
  reason, until the hash matches.

## Run it

```powershell
python -m pip install -e ".[model,api,audit,test]"
python scripts/prepare_inference_samples.py --fold 0 --split test   # ~16 min, needs data/raw
python scripts/check_inference.py                                   # ~2 min, the gate
uvicorn echospace.inference.api:app --host 127.0.0.1 --port 8000
pnpm --dir apps/web dev                                             # open the printed URL
```

## Completion update (2026-10-18)

A/P/D are now registered against their original P9 run/checkpoint hashes.
A and P passed all 898 fold-0 samples (51 rooms); see the A/P replay reports.
The D completion replay verifies the same path after these changes.
Weights and evaluation arrays stay local and are never committed.

The viewer's **Compare observation / A / D / truth** action reconstructs A
and D on the same sample and K, checks grid/mask/truth equality, and opens
four panels. Occupancy and boundary are selectable; each model can be
inspected in 3D. This is a held-out benchmark comparison, not user uploads.

**RoomScene v1** is downloadable beside grid/GLB/NPZ. Its measured layer is
built only from observed free/hit interfaces, independent of the prediction.
The model layer contains the remaining composed floor-plan interfaces.
Both are quantized 2D floor projections; they do not recover wall height.
Observation references resolve to an observation-only NPZ without target
labels. Acoustic records retain the verified positions and emission timing.

The source mesh panel now proxies to the API when no extracted folder is
configured. Set `ECHOSPACE_ACOUSTICROOMS_ROOT` to the dataset directory.
The audited P2 inventory locates extracted OBJs or nested ZIP members; ZIPs
are read through P2 without bulk extraction. Source meshes are reference
geometry, never additional reconstruction input.

The last 128 request records persist in `artifacts/inference/results`.
After a restart a requested result is deterministically recomputed and its
ID checked. Weights and samples are reverified. Set `ECHOSPACE_RESULTS_DIR`
to an empty string to disable disk persistence.

WallFusion promotion remains conditional on selection by the ongoing dual
study. No interim checkpoint has been promoted. Arbitrary scans/WAV uploads,
editing/redesign, GPU serving and authentication remain outside H1.

See `reports/p10_completion.md` for final checks and the native export check.
This completion branch must receive teammate review before merging.

### Restore the original A/P/D weights locally

If `artifacts/p9_full_fold0` is already present on this machine:

```powershell
foreach ($variant in @('A','P','D')) {
    $target = "artifacts/models/echofusion_$($variant.ToLower())_fold0_seed0"
    New-Item -ItemType Directory -Force $target | Out-Null
    Copy-Item "artifacts/p9_full_fold0/$variant/best.pt" "$target/best.pt"
}
```

The registry checks every SHA before serving; copying a different study's
checkpoint into these paths will be rejected. On another machine transfer
these permitted model weights separately from Git and place them at the
registry paths. Do not regenerate eval samples if the verified P6 folder
already exists. A missing file should appear unavailable, never silently
fall back to another model.
