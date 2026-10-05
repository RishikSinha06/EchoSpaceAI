# EchoSpace AI

Research workspace for acoustic-assisted completion of partially observed single-room floor plans.

## What works now

- Browser 3D viewer for `.glb`, `.gltf`, `.obj`, `.ply`, and EchoSpace grid JSON files.
- Observed and inferred walls are shown in different colors for grid predictions.
- Export the currently displayed grid reconstruction as `.glb` for Blender or another native 3D viewer.
- A small example scene is included so the viewer can be tried before a dataset is connected.

The viewer is for inspection. An extruded grid uses an **assumed display height** and is not a recovered 3D scan. Imported meshes retain their source geometry.

## Run the viewer

Install Node.js 20.19+ and pnpm, then:

```powershell
pnpm install
pnpm --dir apps/web dev
```

Open the URL shown by Vite. Use **Open files** or drag files onto the page. For `.gltf`, select its dependent `.bin` and texture files together. The **Example prediction** button loads the included fixture.

## Native viewing

Open a grid JSON in the browser and click **Export GLB**. Open the exported file in Blender or any GLB-capable native viewer. GLB is a portable snapshot of the visible reconstruction; original source meshes can also be opened directly in a native mesh viewer.

## Dataset and prediction contract

See [docs/scene-format.md](docs/scene-format.md). Source datasets are not included. Keep large archives and generated model outputs under ignored `data/` and `artifacts/` folders.

The research pipeline's provisional JSONL/NPZ and metric RoomScene interfaces
are in [contracts/v0.1.0/README.md](contracts/v0.1.0/README.md). Run their
synthetic contract checks using [docs/step1_contract_handoff.md](docs/step1_contract_handoff.md).

P3's scanner-anchored geometry labels and diagnostic CLI are described in
[docs/p3_geometry_handoff.md](docs/p3_geometry_handoff.md). These are complete
geometry targets, not partial scans or training samples.

## Repository map

`apps/web` browser viewer; `src/echospace` future dataset adapters, geometry, models and inference; `configs` experiment configuration; `docs` formats and decisions; `scripts` CLI entry points; `tests` small fixtures; `data` local datasets; `artifacts` outputs.
## Open the local AcousticRooms checkout

Create apps/web/.env.local with:

ACOUSTICROOMS_ROOT=C:/path/to/AcousticRooms-clean-main/AcousticRooms-clean-main

Then restart pnpm dev. The viewer lists the OBJ meshes from room_mesh_obj_format and loads each on demand, without copying the dataset into Git. The section cut slider hides upper geometry for inspection. Export GLB writes the original full mesh for a native viewer; it does not bake the section cut.

The local D0 audit indexed 258 OBJ meshes across ten categories and verified
the main RIR and metadata archives; see [reports/d0_audit.md](reports/d0_audit.md).
The browser currently lists meshes but does not display RIRs or source/receiver
markers. Some `simulation_info` JSON names disagree with their folders; the
audited folder key must be used when joining positions. AcousticRooms OBJ files
use Z as the vertical axis. The browser rotates them into its Y-up frame.
