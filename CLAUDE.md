# Working in EchoSpace AI

This file is for Claude or any coding assistant working in this repository.
Read `README.md`, `contracts/v0.1.0/README.md`, and the relevant module docs
before editing. The project is a research MVP, so data provenance and
reproducible experiments matter as much as a working demo.

## Goal and boundaries

- H1 predicts a **2D single-room floor-plan occupancy and boundary** from a
  partial geometric observation plus paired room impulse responses (RIRs).
  Compare fusion against geometry-only, acoustics-only, and position-only
  baselines on rooms unseen during training. A null acoustic gain is a valid
  research result.
- The existing 3D web view and GLB export inspect a 2D prediction with an
  assumed display height. Do not describe that extrusion as recovered ceiling
  height, furniture, or complete 3D geometry.
- This is a modular monorepo: offline Python data/training jobs, one future
  inference API, and the existing Vite/Three.js viewer. Add an independently
  deployed service only when an actual runtime boundary requires it.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/echospace/` | Python adapters, geometry, scans, acoustics, data, models, training, evaluation, inference |
| `contracts/v0.1.0/` | Versioned JSONL/NPZ and metric RoomScene definitions |
| `apps/web/` | Existing JavaScript Vite/Three.js viewer and GLB export |
| `apps/api/` | Future inference API; currently a placeholder |
| `configs/` | Versioned preprocessing and experiment settings |
| `tests/fixtures/` | Tiny synthetic fixtures safe for Git and CI |
| `data/` and `artifacts/` | Local datasets, processed caches, and outputs; see `.gitignore` |

## Contract rules

1. Use `src/echospace/contract_v0.py` to validate manifest records, NPZ samples,
   and predictions. `manifest.jsonl` links room identity, sample, grid, frame,
   and RIR IDs. NPZ arrays include observed geometry, observed free/wall/visible
   masks, targets, waveforms, source/receiver poses, and RIR validity.
2. Keep measured geometry separate from inferred geometry. `RoomScene v1` has
   explicit provenance layers. The browser's `echospace.grid.v1` JSON is a
   display adapter format, not the sole source of scene truth.
3. Grid size, 0.20 m resolution, and scene orientation are **provisional until
   the AcousticRooms audit**. The 4×4 synthetic fixture is only a contract
   example. Read each sample's grid axes, origin, cell size, and complete
   `source_to_scene` affine transform. `source_units_to_meters` is audit
   metadata; do not apply its scale again.
4. A valid RIR needs verified waveform and source/receiver poses. Both poses
   must fall in observed free space for that scan. Keep room-level split IDs
   stable across all derived scans and RIR subsets; never place derivatives of
   one physical room in different splits.
5. If a field or meaning changes, version the contract and update the Python
   validator, JSON schemas, synthetic fixtures, tests, and docs together.
   Never silently reinterpret existing processed samples.

## Working one module at a time

- Start from the current default branch on a named feature branch. Implement
  one handoff module, with its small fixture and tests, then request review
  before the next dependent module uses it. Separate modules can be coded in
  separate branches or notebooks against the contract; real-data training and
  final integration wait for the upstream audit and preprocessing gates.
- The next data gate is the AcousticRooms D0 audit: verify mesh/RIR identity,
  units, source and receiver poses, coordinate transform, timing convention,
  license, single-room eligibility, and exclusion reasons on a small paired
  sample. Do not infer compatibility from matching filenames or train on
  unverified pairs.
- Keep Python training logic in importable modules. A Kaggle notebook should
  call those modules and record configuration, seed, dataset/split version,
  checkpoint ID, and metrics; it should not become the only implementation.
- Make the handoff reviewable: state input/output contract, failure cases,
  evidence from tests or visual overlays, and any blocked real-data checks.

## Data and Git safety

- Never commit `data/raw/`, `data/raw.zip` (a roughly 37 GB backup), processed
  samples, checkpoints, tokens, or notebook outputs with private data.
  Check `git status` and `git check-ignore` before staging. The five dataset
  routes are not automatically compatible or jointly trainable.
- Paused RAF and SoundSpaces downloads must stay paused unless a human asks
  to resume them. Do not upload source archives or a real fixture to public
  GitHub without verifying redistribution rights. Synthetic fixtures belong
  in Git; real-data checks can run on the machine with the dataset.
- Keep the repository's existing MIT `LICENSE`. Use the account owner's
  approved Git identity when creating commits; do not embed credentials in
  notebook cells, remotes, code, or logs.

## Quick checks from the repository root

```powershell
python -m pip install -e . -r requirements-contracts.txt
python -m unittest discover -s tests -p 'test_contract_v0.py' -v
pnpm install --frozen-lockfile
pnpm build
```

GitHub Actions runs the contract tests and web build without downloading any
dataset or requiring a GPU. Add focused checks for each new module; reserve
full training and external dataset evaluation for recorded experiment runs.

## Branch and commit conventions

- One branch per module, named `module-N-<slug>` (for example
  `module-2-acousticrooms-audit`), started from the current `main`. Never
  commit to `main` directly; merge through a pull request reviewed by the other
  teammate.
- Conventional Commits scoped by area: `feat(io): ...`, `fix(io): ...`,
  `test(io): ...`, `docs(audit): ...`, `chore(data): ...`, `ci: ...`. Keep each
  commit small and to one logical change.
- Before every `git add`, run `git status` and `git check-ignore -v` on anything
  under `data/`. Stage explicit paths only; never `git add .` or `git add -A`.
- Never commit `data/raw/`, `data/cache/`, `*.zip`, `*.npz`, checkpoints or
  `runs/`. The only `.wav` and `.obj` files allowed in Git are the attributed
  samples under `data/samples/`.
- Never force-push, and never amend, rebase or squash commits that have been
  pushed. Do not rewrite another teammate's commits; add new ones.
- The pull request description states the module's exit gate and whether it
  passed, with evidence (test output, report, figures) and any blocked checks.
- Ask before adding a new dependency. Keep code deterministic (seeded sampling,
  versions logged, no network calls at runtime) and free of absolute paths.
