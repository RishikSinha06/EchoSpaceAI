# EchoSpace viewer scene format v1

The viewer accepts ordinary mesh files (`.glb`, `.gltf`, `.obj`, `.ply`) and an EchoSpace grid JSON. Mesh files are displayed as supplied; their coordinate units are assumed to be meters unless the source says otherwise.

The grid JSON is a lightweight exchange format for **2D model output**, separate from the canonical editable `RoomScene` planned for the research pipeline:

```json
{
  "format": "echospace.grid.v1",
  "name": "Example room",
  "cell_size_m": 0.2,
  "origin_m": [0, 0],
  "wall_height_m": 2.7,
  "interior": [[0, 1], [0, 1]],
  "observed": [[1, 1], [0, 0]]
}
```

`interior` is the **final predicted occupancy** (1 = inside room, 0 = outside). `observed` marks cells whose state was measured (1 = observed, 0 = unknown). Both arrays have the same height and width and contain only 0 or 1. Row 0 starts at `origin_m[1]`, with columns increasing along +X and rows along +Z. The renderer creates vertical walls where an inside cell borders an outside cell; an edge is observed only if both cells touching that edge are observed. Other edges are colored as inferred. `wall_height_m` is an assumed visualization height, not a model prediction.

The model adapter should paste verified observed cells into the final occupancy before writing this file and preserve the real metric transform. Do not derive `origin_m` from held-out complete geometry at inference time. This format makes no claim about room ceiling, furniture, or native 3D reconstruction.

For raw mesh datasets, an adapter must establish units and axes. A mesh load alone does not create an `observed` mask or paired RIR input.

## Grid v2 (`echospace.grid.v2`, P10)

v2 is what the inference service writes. It keeps v1's meaning of `interior`,
`observed`, `cell_size_m` and `wall_height_m`, and adds what a real scan frame
needs. v1 files still open unchanged.

| Field | Meaning |
| --- | --- |
| `width`, `height` | Grid size; must equal the array shapes |
| `origin_scene_m`, `axis_u_scene`, `axis_v_scene` | The sample's own grid frame in scene metres (X right, Y up, Z forward). Cell `(row, col)` covers `origin + cell * (col * u + row * v)` to the next corner. P4 aligns the grid with the scan's dominant wall, so the axes are generally rotated. |
| `observed_wall` | Cells holding a measured wall hit (subset of `observed`) |
| `occupancy_probability`, `boundary_probability` | Raw network output on every cell (3 decimals) |
| `truth` | `{interior, boundary}` labels when known, else `null`; never an input |
| `measurements` | Source and receiver positions (scene metres) of the K RIRs used |
| `wall_height_assumed` | Always `true`: the model predicts a 2D footprint |
| `display_convention` | How `interior` was composed (below) |

`interior` is a display composition, not the scored output: thresholded
occupancy on unobserved cells, measured free cells pasted inside, measured
wall-hit cells drawn outside (only about half of them have their centre inside
the room, so their occupancy is not measured). Metrics always use the raw
probabilities on valid unobserved cells.

**Wall rule (v1 and v2).** A wall face stands on every edge between an
interior cell and a non-interior cell or the canvas border. It is *observed*
when the cell outside that edge was measured (v2: `observed_wall`; v1:
`observed`), otherwise *inferred*. The browser (`apps/web/src/grid.js`) and the
GLB exporter (`src/echospace/inference/export.py`) implement the same rule;
`node apps/web/scripts/check-grid.mjs` and `tests/test_inference.py` hold both to
the shared fixture `tests/fixtures/grid_v2_agreement.json`.

The GLB from the service uses the same scene metres and Y-up frame, the floor
at Y = 0 (AcousticRooms' floor datum), and three named meshes: `floor`,
`observed_walls`, `inferred_walls`. An AcousticRooms OBJ rotated by the audited
`source_to_scene` (`(x, y, z) -> (x, z, -y)`) lands on the same coordinates.
