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

## Repository map

`apps/web` browser viewer; `src/echospace` future dataset adapters, geometry, models and inference; `configs` experiment configuration; `docs` formats and decisions; `scripts` CLI entry points; `tests` small fixtures; `data` local datasets; `artifacts` outputs.
## Open the local AcousticRooms checkout

Create apps/web/.env.local with:

ACOUSTICROOMS_ROOT=C:/path/to/AcousticRooms-clean-main/AcousticRooms-clean-main

Then restart pnpm dev. The viewer lists the OBJ meshes from room_mesh_obj_format and loads each on demand, without copying the dataset into Git. The section cut slider hides upper geometry for inspection. Export GLB writes the original full mesh for a native viewer; it does not bake the section cut.

The dataset checkout currently tested contains 258 OBJ meshes across ten categories. Its single_channel_ir.zip, metadata.zip, and depth_map.zip files are Git LFS pointers, not downloaded archives. The mesh browser therefore does not display RIRs or source/receiver markers. Some simulation_info folder names disagree with the name inside simulation.json; those positions must be audited before they are overlaid on a mesh.
AcousticRooms OBJ files use Z as the vertical axis. The dataset browser rotates them into the viewer's Y-up coordinate system before measuring dimensions or exporting GLB.
