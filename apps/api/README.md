# EchoSpace inference API (P10)

The service code lives in `src/echospace/inference/api.py` so it is importable
and tested like the rest of the pipeline. It is a local research service: it
binds to `127.0.0.1` and serves only registered, hash-verified checkpoints on
committed, hash-verified evaluation samples.

```powershell
python -m pip install -e ".[model,api,audit]"
python scripts/prepare_inference_samples.py --fold 0 --split test   # once, ~15 min, needs data/raw
uvicorn echospace.inference.api:app --host 127.0.0.1 --port 8000
```

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Status, versions, servable models, prepared samples |
| `GET /v1/models` | Registry entries with `available` and the reason when not |
| `GET /v1/samples?split=test&model_id=...` | Prepared samples and their split for the model's fold |
| `POST /v1/reconstructions` | `{"schema_version": "echospace.reconstruction_request.v1", "model_id", "sample_id", "k": 1-8}` |
| `GET /v1/reconstructions/{id}` | Summary, metrics and provenance |
| `GET /v1/reconstructions/{id}/grid.json` | `echospace.grid.v2` (see `docs/scene-format.md`) |
| `GET /v1/reconstructions/{id}/scene.glb` | Native-viewer GLB, scene metres, Y up |
| `GET /v1/reconstructions/{id}/provenance.json` | Model, sample, K, threshold, code revision, conventions |
| `GET /v1/reconstructions/{id}/prediction.npz` | Contract v0.1.0 prediction (raw probabilities) |

Errors: unknown model or sample `404`; unknown request fields, wrong schema
version or K outside 1..8 `422`; a room from the model's training or validation
partition `409` unless `"allow_seen_rooms": true`; a registered model whose
checkpoint is missing or fails its hash `503`. Reconstruction ids are
deterministic (model checkpoint, sample hash, K, threshold).

Settings: `ECHOSPACE_MODELS`, `ECHOSPACE_EVAL_DIR`, `ECHOSPACE_EVAL_SPEC`,
`ECHOSPACE_FOLDS`, `ECHOSPACE_DEVICE` (default `cpu`). The web viewer reaches
the API through its `/svc` proxy; override the target with
`ECHOSPACE_API_URL` in `apps/web/.env.local`.

## P10 completion routes and storage

- `GET /v1/reconstructions/{id}/room-scene.json`: schema-valid RoomScene v1,
  separate observation-only measured interfaces and model interfaces.
- `GET /v1/samples/{sample_id}/observation.npz`: observation masks, points,
  verified audio and poses; no complete target labels.
- `GET /v1/acousticrooms`: audited local source mesh catalogue.
- `GET /v1/acousticrooms/mesh?id=category/room_id`: raw OBJ via P2 archive reader.

Set `ECHOSPACE_ACOUSTICROOMS_ROOT` to the dataset directory and optionally
`ECHOSPACE_MESH_MANIFEST` to an audited P2 inventory. These reference meshes
are not model inputs. The Vite `/api/acousticrooms` proxy supports zipped data.

`ECHOSPACE_RESULTS_DIR` defaults to `artifacts/inference/results`; an empty
value disables persistence. Only the last 128 small request records are
saved. GET after restart recomputes and verifies the deterministic ID.
Model/sample validation failures remain errors rather than serving stale data.
Use one local service process; this filesystem cache is not a distributed store.
