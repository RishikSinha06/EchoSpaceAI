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
