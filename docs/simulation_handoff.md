# Software sensor lab and dataset replay

## Implemented scope

This module adds a local software-in-the-loop acquisition engine to the P10
FastAPI app and existing Three.js viewer. It does not start training, write
training caches, or grant synthetic captures eligibility for training.

Two modes use the existing Sample v0.1.0 model adapter:

* **Audited replay:** load an existing verified evaluation sample by ID. Preserve
  its original room identity, checksum, masks, waveforms, pair order and poses.
  Replay speed changes presentation only. The existing model-fold seen-room
  refusal remains active. This is the reliable dataset demonstration path.
* **Interactive virtual room:** place a scanner, speaker and microphone in a
  metric rectangle or L-shaped room; capture a partial scan and synthetic RIRs;
  run registered A/P/D models on that observation. These are out-of-distribution
  diagnostic experiments, never held-out research results or validated hardware.

This branch is based on `module-10-completion` (PR #8), which supplies A/P model
registration, comparison, RoomScene and archive-backed mesh serving. Review this
module after that dependency; it is a separate PR against the dependency branch.
It does not depend on the unmerged P11 branch.

## Start it

From the checkout containing this module:

```powershell
python -m pip install -e ".[model,api,audit,test]"
uvicorn echospace.inference.api:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
pnpm --dir apps/web dev
```

Open the Vite URL, then **Open sensor lab**. The default API URL is
`http://127.0.0.1:8000`; set `ECHOSPACE_API_URL` in `apps/web/.env.local` for
another port. The existing verified `data/processed/eval_samples` and registered
checkpoint files are needed for replay and inference. They are not in Git.
Virtual capture itself needs neither real dataset assets nor checkpoint files.
Models without local verified weights are shown as unavailable.

The review instance runs at http://127.0.0.1:5180 with API port 8012. These are
local processes, not installed startup services.

## Presentation sequence

1. Capture the default rectangle. Read the observation and RIR plots.
2. Enable **Show reference targets / paths** to inspect the separate truth and
   theoretical arrivals. Those values are not model inputs.
3. Use **Compare A / D / truth** on the same capture. The comparison banner
   explicitly identifies synthetic diagnostics.
4. Change to an L-shaped room, move devices by dragging, marker arrow keys or numeric coordinates,
   and capture again. Use the 3D buttons to inspect room setup or observations.
5. Add source/microphone pairs and choose K. Both endpoints must be in observed
   free space; rejected pairs are reported and compacted out of model inputs.
6. Open device effects and try sweep excitation, noise or timing offset. Capture
   again; configuration changes invalidate old results until recaptured.
7. Switch to **Audited dataset replay**, capture, then compare or use Next/Play.
8. Download the capture bundle or RIR WAV. Existing reconstruction export buttons
   expose grid, GLB, provenance and RoomScene after inspecting a model result.

The scanner may move within a session, but the grid stays anchored to its first
pose and declared heading. A new room or replay sample starts a new session.
Moving a device captures an independent frame, not an accumulated SLAM map.
The viewer shows a horizontal depth/LiDAR-like slice, not a full 3D LiDAR scan.

## Engine and signal model

Coordinates are canonical X-right, Y-up, Z-forward, in metres. A 64x64 canvas at
0.20 m/cell is fixed from the first scanner pose; clipping is rejected instead
of choosing a hidden-truth crop. Ray casting stores free traversal, wall hits
and unknown separately. Range limits, Gaussian range noise and dropped rays are
configurable; dropped rays create no free-space evidence.

Acoustics uses visible direct paths plus finite specular first-order vertical
wall, floor and ceiling paths. Path delay is length / sound speed; default sound
speed is 343 m/s. First-order amplitude uses a scalar reflection coefficient and
inverse distance. Fractional-sample impulses distribute energy to two adjacent
samples. The same engine rejects occluded direct/reflection segments in the
nonconvex L room. No higher-order reverberation, diffraction, scattering,
furniture, material frequency response, directivity or calibrated device response
is implemented. Relative signal amplitude is not calibrated sound pressure.

The virtual source emits an impulse or a 2048-sample logarithmic sweep. Convolution
produces a microphone recording. Optional low-pass filtering, input gain,
recording noise, timing offset, clock drift, clipping and ADC quantization model
specific acquisition faults. Sweeps use regularized FFT deconvolution. This is a
controlled digital experiment, not a validation of physical sweep measurement.
Known emission timing is retained. Selected timing faults deliberately remain
uncompensated and are labeled in metadata. Model preprocessing uses the existing
P5 peak normalization and 16 kHz / 80 ms window. Gain before peak normalization
can cancel in a linear signal; clipping/noise can still affect the waveform.
Low-pass cutoffs are bounded strictly below 8 kHz. Reported pose noise is distinct
from true reference positions and preserves observed-free eligibility.

Discrete events record scan, emission, recording and capture completion. Replay
playback does not resample RIRs or change this clock. The engine is deterministic
for its configuration, initial frame, ordinal and seed in the same numerical
software environment; it does not promise bit identity across library versions.

## Interfaces and provenance

`contracts/simulation-config.v1.schema.json` defines the configuration envelope:

```json
{"schema_version":"echospace.simulation_config.v1","config":{}}
```

Defaults and limits are available from `GET /v1/simulation/capabilities`.
Unknown fields, nonfinite numbers, invalid bounds and duplicate pairs are rejected.
Replay accepts sample ID, seed and K but rejects virtual room/device overrides.

Routes under `/v1/simulation`:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/capabilities` | Limits, defaults, explicit limitations |
| POST | `/sessions` | Create versioned configuration |
| GET / PATCH | `/sessions/{id}` | Inspect / replace configuration |
| POST | `/sessions/{id}/capture` | Capture current sensor state |
| POST | `/sessions/{id}/infer` | Registered model ID, optional K |
| GET | `/captures/{id}` | Public observation and diagnostics |
| GET | `/captures/{id}/observation.npz` | Model-facing observations only |
| GET | `/captures/{id}/signals?pair=0` | Excitation and microphone recording |
| GET | `/captures/{id}/audio.wav?pair=0` | Processed float32 RIR |
| GET | `/captures/{id}/reference.json` | Explicit privileged reference |
| GET | `/sessions/{id}/bundle.zip` | Portable current-capture bundle |

The ordinary `/v1/reconstructions` route continues to accept audited sample IDs
only. Virtual captures enter only through the explicit simulation route.
Unknown session/capture/model is 404; invalid configuration is 422; no current
capture, no eligible pair or seen replay room is 409. Missing weights yield 503.

The model input adapter takes observed masks/points, reported poses, RIRs and valid
pair flags only. Full occupancy, boundary, true device poses and clean theoretical
RIRs are reference-only. Tests mutate targets and valid-cell labels and verify
unchanged model input tensors. The admin configuration knows the selected room;
that does not make its polygon an input to the model.

Default bundles contain `session.json`, `observation.npz` and (for virtual mode)
`device_signals.npz`. They contain no target arrays or clean room impulse response.
The manifest records the operator configuration, stable frame, history, seed,
engine version and capture digest. Explicit `?include_reference=true` adds named
`REFERENCE_ONLY.json` and `REFERENCE_SIGNALS_ONLY.npz`. These files are never a P6
training manifest. Every capture and bundle has `training_eligible=false`.

Memory is bounded to 32 sessions / 128 captures, with 256 recent events and 32
history entries per session. A service restart forgets sessions; downloaded
bundles retain the current capture. There is no bundle-import/resume UI yet.
Virtual reconstruction results use the ordinary bounded result cache, and their
observation references point to immutable capture IDs while those captures exist.
They are not promised recoverable after a service restart. Download complete
bundles/exports if needed later. GLB prediction extrusion height remains assumed.

## Command-line and verification

```powershell
python scripts/simulate_room.py --config tests/fixtures/simulation/rectangle.json --output artifacts/simulation/rectangle
python scripts/simulate_room.py --sample-id <verified_sample_id> --source-root . --output artifacts/simulation/replay
python -m pytest tests/test_simulation.py tests/test_inference.py -q
python scripts/check_simulation.py --api-url http://127.0.0.1:8000
node apps/web/scripts/check-grid.mjs
node apps/web/scripts/check-comparison.mjs
pnpm --dir apps/web build
```

CLI capture refuses a nonempty output directory. Add `--include-reference` only
when intentionally exporting evaluation/reference data. No physical devices,
paid compute, new dependencies or model training are needed for this module.
The full integration script requires locally registered A/P/D checkpoints and
verified evaluation samples. It checks all three models on rectangle, L room,
sweep faults and audited replay, exports GLB/RoomScene, checks observation-only
bundles, and verifies replay matches the original route exactly.

See `reports/simulation_check.md` for the actual measured gate. Remaining work
before physical hardware use: real device drivers, synchronization/calibration,
measured real-room acquisition, simulation-to-real validation and (if needed)
a higher-fidelity acoustic solver. Virtual success does not establish those gates.
