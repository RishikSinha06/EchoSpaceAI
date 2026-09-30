# EchoSpace sample contract 0.1.0

This is an exploratory contract for the H1 pipeline. The AcousticRooms audit may
change the grid dimensions, resolution, and scene orientation. Every record
therefore carries its own grid and source-to-scene transform. A breaking change
increments the schema version and supplies a migration; it does not reinterpret
existing samples silently.

## Artifacts

- `manifest.jsonl`: one JSON object per sample, validated by
  `manifest.schema.json` and `echospace.contract_v0.validate_manifest_record`.
  Paths are relative to the manifest's directory. Samples from the same room
  retain the same `room_id`; it identifies the physical room across derived
  scans and releases, and downstream split logic must group by that ID.
- `sample.npz`: numeric arrays listed below, loaded with `allow_pickle=False`.
  The JSONL record supplies the sample ID, grid, RIR IDs, sample rate, and
  coordinate transform. A missing RIR uses `rir_valid=0`; the row still exists
  and must not be treated as a measured signal.
- `prediction.npz`: probabilities on the input sample's grid. The inference
  service will bind a prediction to `sample_id`, checkpoint ID, and contract
  version in its response metadata during Step 10.
- `RoomScene v1`: a separate metric application scene, checked by
  `room-scene.schema.json`. It preserves measured and model-inferred geometry
  as different provenance layers; it is not a raster label or a claim of true
  3D reconstruction.

| Sample NPZ key | Shape | Meaning |
| --- | --- | --- |
| `observed_points_scene_m` | `N,3` | Partial scan points in scene metres |
| `observed_cells` | `H,W` | Cells visible in the partial scan |
| `observed_free` | `H,W` | Visible cells measured as free interior space |
| `observed_wall` | `H,W` | Visible cells measured as wall |
| `valid_cells` | `H,W` | Cells eligible for labels and metrics |
| `target_occupancy` | `H,W` | Supervised room occupancy |
| `target_boundary` | `H,W` | Supervised wall boundary |
| `rir_waveforms` | `R,T` | One fixed-length, finite waveform per `rir_id` |
| `rir_positions_scene_m` | `R,2,3` | Source and receiver positions in scene metres, in that order |
| `rir_valid` | `R` | Which waveform rows are measured and usable |

Prediction NPZ keys are `occupancy_probability` and
`boundary_probability`, each `H,W` with finite values in `[0,1]`.
`H=grid.height`, `W=grid.width`, and `R=len(rir_ids)`. The per-sample
`origin_scene_m`, orthogonal unit axes, and `cell_size_m` locate grid cells in
the scene. `source_to_scene` is an invertible 4×4 affine matrix; its last row
is `[0,0,0,1]`. The meaning of scene axes is written in `frame.description`.
`target_occupancy=1` means inside the room; `target_boundary=1` marks a wall
cell. Position rows are zero placeholders when the corresponding RIR is
invalid; an RIR becomes valid only after its waveform and poses are verified.
`observed_free` and `observed_wall` are disjoint subsets of `observed_cells`.
They are measured or simulated observations, never copied from the complete
target during inference.
Both positions of every valid RIR must project into `observed_free`; this
prevents a hidden source or receiver from leaking unseen room geometry.

The synthetic fixture under `tests/fixtures/` is deliberately tiny and is not
an AcousticRooms sample. It tests the handoff format without shipping raw data.
