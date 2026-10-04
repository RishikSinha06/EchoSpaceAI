# P3 geometry-label handoff

P3 turns an **audited AcousticRooms OBJ** into a complete interior target and
an outer structural-wall target. It uses the D0 `floor_slice` implementation at
1.1 m above the mesh floor. D0 found AcousticRooms to be metres with Z up and
verified the source-to-scene rotation `(x, y, z) -> (x, z, -y)`.

The caller **must supply a scanner anchor and heading**. The 64×64 grid uses
0.20 m cells (12.8 m square), centred on that anchor. It is never centred or
rotated using the hidden room bounds. Heading is measured in the scene XZ
plane from +X toward +Z. The anchor will eventually come from P4's partial
scan; P3 does not choose a scan pose from the complete room.

`label_acousticrooms_mesh(mesh, GridFrame(...))` returns `occupancy`,
`boundary`, `valid_cells`, the grid frame, and diagnostics. `occupancy=1` for
cell centres inside the room footprint. `boundary=1` for interior cells within
half a cell diagonal of its outer boundary. Internal closed loops are filled:
the OBJ has no semantics, so furniture and columns are excluded from the
structural-wall target. This also means wall-attached built-ins can alter the
outline; inspect those cases. The raster is a label, not a scan observation.

The function rejects a missing or disconnected room outline, a scanner anchor
outside it, a room that clips the fixed canvas, an invalid mesh, and slice
heights outside 1.0–1.2 m. D0's meshes are not watertight, so P3 does not use
mesh volume or `contains` to define the interior. D0's snap-rounded slice
avoids false gaps and chords from triangle soup.

For a local one-room diagnostic, with an **explicitly chosen** source-frame
scanner pose:

```powershell
python scripts/build_geometry_labels.py `
  --mesh data/samples/acousticrooms/Bathrooms/Bathrooms_idx_21/Bathrooms_idx_21.obj `
  --anchor-source-xyz -2.68 5.15 1.5 --heading-deg-scene 0 `
  --out-dir artifacts/p3-example
```

The example uses a receiver position from the attributed P2 sample as an
explicit **diagnostic** anchor; it is not a P4 scanner pose. The command writes
target arrays, a JSON diagnostic, and an outline-over-raster PNG into ignored
`artifacts/`. It does not create a `Sample v0.1.0`: observed cells, RIR
selection, and split assignment belong to P4–P6. Only build training samples
after the P4 anchor/heading and observed-free mask pass the contract checks.

Run `python -m pytest tests/test_geometry_labels.py -q` with the existing
`.[test,audit]` extras. Before accepting real labels, inspect at least 20
overlays across room categories and log clipping, internal loops, and
wall-attached furniture. The real-label acceptance gate remains pending P4
scanner poses; no generated target in `artifacts/` is training eligible yet.

Two attributed P2 sample meshes were exercised locally with explicit receiver
poses used **only as diagnostic anchors**. `Bathrooms_idx_21` produced a
2.8212 m² footprint, 70 interior cells, 20 boundary cells, and no clipping.
`MeetingRoom_idx_14` produced a 5.1889 m² footprint, 118 interior cells,
22 boundary cells, and no clipping. The latter has 0.131 open slice-linework
fraction from internal geometry; its outer room loop still closes. Both
outline-over-raster images were inspected. These local outputs stay under
ignored `artifacts/` and are not part of the training manifest.
