# Step 1 handoff: package and contract foundation

The Python contract module is `src/echospace/contract_v0.py`. Its public
functions are `validate_manifest_record`, `read_manifest`, `validate_sample`,
`validate_prediction`, and `load_npz`. The machine-readable manifest schema,
array contract, synthetic fixture, and CI workflow live alongside it.

The repo keeps Python pipeline code in `src/echospace`, tests in `tests`,
versioned data interfaces in `contracts`, run configuration in `configs`, the
existing web app in `apps/web`, and the future inference service in `apps/api`.
Raw data, generated sample caches, and checkpoints are local artifacts, not
source files.

To check this module locally from the repository root:

```powershell
python -m pip install -e . -r requirements-contracts.txt
python -m unittest discover -s tests -p 'test_contract_v0.py' -v
```

The synthetic test writes a JSONL manifest and two NPZ files to a temporary
directory, then reads and validates them through the production contract
functions. It also checks failure cases for mismatched grids and RIR counts,
invalid transforms and probabilities, duplicate IDs, and unsafe object arrays.

## Step 2 input

The dataset auditor fills the source IDs, transform, units, grid dimensions,
and RIR sample rate from verified AcousticRooms evidence. `split` remains
`unassigned` until the room-level split is frozen. The example fixture's
orientation and 4×4 grid are illustrative, not project defaults. If auditing
reveals a missing field or a different sample representation, update the
contract version and its fixture, tests, and migration before producing a
large processed cache.

For a grid cell at column `u` and row `v`, its scene-space centre is
`origin_scene_m + cell_size_m * ((u + 0.5) * axis_u_scene + (v + 0.5) * axis_v_scene)`.
`source_to_scene` is the complete affine transform into scene metres;
`source_units_to_meters` records the source unit conversion for audit and
must not be applied a second time.
