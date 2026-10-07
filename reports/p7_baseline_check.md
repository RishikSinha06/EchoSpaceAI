# P7 baseline foundation check

Date: 2026-10-07. Base: `d87557d` (P6 merge).
Branch: `module-7-models`.

**Synthetic foundation gate: PASS. Full P6 held-out gate: PENDING.**
The generated P6 room caches and fixed evaluation NPZs are absent from this
checkout. Their committed metadata and P6 report were reviewed, but no real
AcousticRooms baseline accuracy claim is made here. Fusion remains gated on
the common real-data baseline run and prediction review.

## Implementation and verification

- B0 extends observed straight wall runs and flood-fills from observed free
  cells. No parameters, target geometry or acoustic positions are read.
- A is a geometry-only U-Net; P adds masked paired position embeddings and
  source/mic density maps. Neither consumes acoustic waveforms.
- Both output occupancy/boundary logits. Training uses BCE + Dice only on
  valid unobserved cells (`M=0`), equally weighting the heads.
- One immutable frozen sample set supplies all baselines. Room/group crossings,
  duplicate IDs and edited manifest/sample content are refused.
- The runner implements seeded sampling, deterministic algorithms, CUDA AMP,
  validation-selected best checkpoints, early stopping, best/last states,
  run manifests and held-out probability exports. CPU explicitly disables AMP.

Pulled baseline: **103 tests passed** in 96.99 seconds.
Final full suite: **113 tests passed** in 118.90 seconds, including 10 P7 tests.
The focused P7 suite passed in 8.34 seconds.

P7 tests verify zero gradients on observed/invalid cells, ignored observed
targets including NaNs, empty supervision, model independence from targets and
waveforms, A's independence from positions, P's use of valid positions and
permutation invariance, invalid padding, B0 stopping rules, frozen checksums,
split isolation, deterministic repeated training, early stopping, checkpoint
reload and unchanged training state after altering only test targets.

## Common synthetic held-out gate

Both final CPU and CUDA runs used this freeze content checksum:

`93b5a633275304cc37d2b04511b0a609b724953922c4bb714b4bc2faf657e557`

Six artificial rooms have disjoint identities/groups: four training rooms with
eight masks, one validation room with two masks, and one test room with two
masks. No measured RIRs are invented or consumed. The baseline variants consume
the identical masks and position rows. A/P train for three epochs with width 8;
this is a smoke configuration, not the default 100-epoch width-16 benchmark.

| Variant | Parameters (smoke) | CPU | CUDA | Held-out NPZs per run |
| --- | ---: | --- | --- | ---: |
| B0 | 0 | Pass | Pass, no AMP | 2 |
| A | 122,826 | Pass | Pass, AMP enabled | 2 |
| P | 128,538 | Pass | Pass, AMP enabled | 2 |

Each run also exported two validation predictions. A/P selected epoch 3 in this
short smoke. Best-checkpoint reload and early stopping are tested separately.
CPU and CUDA consumed the same freeze but were not expected to have identical
floating-point outputs. Synthetic accuracy values are not research findings.

Runtime: Python 3.11.9, NumPy 1.26.4, PyTorch 2.5.1+cu121; CUDA 12.1 on
an NVIDIA GeForce RTX 4060 Laptop GPU.

The first CPU held-out probability overlay was inspected. Orientation and grid
alignment match the targets and observation outline. B0 overfills the open
upper region, consistent with its stated heuristic limitation; the short A/P
runs emit smooth finite probabilities rather than recovered final room plans.

## Reproduction and artifacts

```powershell
python -m pip install -e ".[test,audit,model]"
python -m pytest tests -q
python scripts/check_baselines.py --out-dir artifacts/p7_final_cpu --plots
python scripts/check_baselines.py --device cuda --out-dir artifacts/p7_final_cuda
```

Use a new output folder when repeating runs; outputs intentionally refuse
overwrites. This host set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to bypass an
unrelated globally installed pytest plugin. PyTorch was already installed;
the optional dependency was added with the user's approval.

Local ignored output folders contain the freeze, `run.json`, `best.pt`/
`last.pt`, validation/test prediction NPZs and bound manifests, summaries and
CPU overlays. Source/code hashes bind runs even before the new Git commit.
No processed sample, checkpoint or prediction binary is committed.

The new CI baseline job installs CPU PyTorch, runs the focused tests and
synthetic foundation gate. Local verification does not imply remote CI has run.

## Full P6 held-out exit gate

On the machine with the matching P6 binaries, run the freeze/run commands in
`docs/p7_baselines_handoff.md`. B0/A/P must share the resulting freeze checksum,
and produce reviewed held-out predictions before expanding fusion. Missing
binaries cause an explicit CLI error; the scripts do not download raw data,
regenerate committed folds/masks or substitute the two-room sample.
