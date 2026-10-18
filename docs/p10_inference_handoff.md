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
- **Weights stay out of Git.** Put the D checkpoint at
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

## Left for the teammate

These parts of P10 are not done. Each can be added without changing the code
above, unless stated.

1. **A and P checkpoints.** Only D was available. Copy
   `artifacts/p9_full_fold0/A/best.pt` and `P/best.pt` to
   `artifacts/models/`, add two entries to `configs/models.json` (hashes from
   their `run.json`; recorded K = 8 room-weighted: A 0.728179 / 0.415571,
   P 0.736873 / 0.406417), then run
   `python scripts/check_inference.py --model <id>` for each. No code change.
2. **Side-by-side comparison view.** The plan asks to compare observation,
   A, D and truth. The viewer shows one model with the truth outline and lets
   you switch models; a split or overlay view of several reconstructions of the
   same sample still needs building in `apps/web/src/service.js`. The API
   already serves any registered model on the same sample.
3. **RoomScene v1 export.** The contract's `room-scene.schema.json` is not yet
   written by the service; only grid v2, GLB, NPZ and provenance are. Add an
   exporter that places measured geometry and inferred geometry in separate
   provenance layers.
4. **WallFusion.** If the dual-architecture study selects WallFusion, register
   its checkpoint (the loader already builds the `WallFusion` family through
   `models/factory.py`) and run the gate for it.
5. **Native-viewer check by eye.** GLBs were verified programmatically. Open
   one in Blender once to confirm colours, names and orientation look right
   next to the source OBJ rotated by `source_to_scene`.
6. **AcousticRooms mesh panel.** The existing panel needs an extracted
   `room_mesh_obj_format/` folder (`ACOUSTICROOMS_ROOT`); with the zipped
   dataset it reports unavailable. Serving meshes from the zips through the
   P2 adapter would fix it.
7. **Persistence.** Results are kept in memory (last 128). Restarting the
   service forgets them; ids are deterministic, so re-posting recreates them.

Out of scope for H1: arbitrary user scans or WAV uploads (H2), editing and
redesign (H3), GPU serving, authentication.
