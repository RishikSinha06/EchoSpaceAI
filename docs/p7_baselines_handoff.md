# P7 training foundation and baselines

Base: `d87557d`, the P6 merge on main. Work branch: `module-7-models`.
P7 implements B0/A/P and an importable training foundation. Fusion and acoustic
encoders are outside this module. PyTorch is an approved optional `model` extra;
the existing dataset remains framework independent.

## Model inputs and outputs

| Variant | Implementation | Inputs |
| --- | --- | --- |
| B0 | `models/baselines.py`, wall extrapolation + flood fill | observed free/wall masks |
| A | `models/spatial.py`, geometry-only U-Net | free, wall and observed-cell masks |
| P | same U-Net, geometry plus paired positions | A inputs + source/mic positions and validity |

All return `occupancy_logits` and `boundary_logits`, each `(B,64,64)`. Sigmoid
produces contract-shaped probabilities. No forward path reads hidden labels,
`valid_cells`, waveforms, room identity, fold, coverage or split. No acoustic
encoder or fusion model is implemented.

A is a three-level U-Net (64/32/16/8 spatial resolutions), with two convolutions,
GroupNorm and SiLU at each level, nearest upsampling and skip concatenation.
The default base width is 16. Two independent output channels predict occupancy
and boundary. The smaller foundation gate uses width 8 to reduce runtime.

P adds two source/mic density maps. A shared MLP embeds each paired six-coordinate
vector, then masked mean pooling broadcasts its embedding into the bottleneck.
This retains pairing and provides a permutation-invariant position control.
Positions must already be P6 grid-frame metres `(column, row, height)`, with
horizontal coordinates normalized by 12.8 m and height by the configured 3 m
scale. Height scaling is numerical normalization, not inferred room height.
Invalid positions are zeroed before projection/embedding; no padded row enters
the pooling denominator. An empty position set is finite, although P6 primary
samples require usable pairs.

B0 extends observed contiguous horizontal/vertical wall runs (minimum three
cells) by up to 64 cells in each direction, stopping at observed free cells or
the canvas edge. Flood fill from observed free cells through the extrapolated
barriers estimates occupancy; barriers themselves count as occupancy. No labels
or room extent are consulted. B0 has no fitted parameters. It can overfill an
open outline and extrapolate furniture or fail on angled walls; these are
limitations to measure, not hidden target information to correct it with.

## Common frozen samples

`training/frozen.py` materializes P6 items once in internal format `p7.frozen.1`.
It stores observation/label masks, grid-frame positions and validity, along with
sample/room/group identity, the grid and any P6 spatial transform. It does not
store waveforms, and does not redefine contract-v0 scene coordinates.

The freeze draws P6 training epoch zero, including training augmentation once.
Val/test use the committed P6 fixed evaluation masks. Every baseline consumes
the same immutable freeze, without redrawing masks each epoch. This foundation
design deliberately trades online variation for an auditable baseline comparison;
P8 can add synchronized epoch-specific freezes under a separately recorded study.

`--training-source fixed-masks` is the alternative when the supplied P6 archive
contains only fixed masks. It verifies the committed sample checksums and uses
every mask in each fold-assigned train/val/test room set. Training receives only
training-room masks, with no P5 or spatial augmentation; K is fixed explicitly
(8 by default). No room caches are required. This setting is recorded as
`P6_fixed_masks`, and must not be described as online P6 training draws.

Manifest and sample content checksums are verified on every load. Duplicate IDs,
room/group split crossings, invalid poses, inconsistent masks and different grid
settings are refused. A destination must be empty; create a new version instead
of overwriting samples or silently regenerating committed P6 masks. Optional
sample limits are recorded and mean the run is a subset, not a full-fold benchmark.

## Loss and validation

`training/losses.py` uses `valid_cells=1 AND observed_cells=0` for **both heads**.
Each sample receives mean BCE over this domain plus soft Dice over the same
domain; the two heads are averaged. Default BCE and Dice weights are both 1.
Observed/invalid cells are masked before arithmetic and have zero gradient.
Samples with no unobserved supervision contribute differentiable zero, and the
freeze refuses them so training batches cannot silently lose all supervision.

Validation loss selects the best checkpoint; test targets never select model
state, settings or early stopping. Loss and threshold-0.5 IoU are reported only
on valid unobserved cells, averaged per sample. Empty-union IoU is 1. These basic
diagnostics are not the full P9 metric suite or room-weighted research results.
Probabilities are exported for the full grid with no ground-truth overwrite or
observed-cell loss. Consumers must retain measured geometry as its own layer.

## Training lifecycle

`training/runner.py` provides AdamW, deterministic seeds, CUDA AMP, validation,
early stopping and atomic best/last checkpoints. `configs/p7_baselines.json`
records defaults: 100 epochs, batch 8, learning rate 0.001, weight decay 0.0001,
patience 10 and minimum improvement 0.0001. CPU is the portable default; pass
`--device cuda` for GPU AMP. Validation and prediction export use float32.

Seeds cover Python, NumPy, torch and CUDA. Batch order has a seeded generator per
epoch. Deterministic algorithms are required; cuDNN benchmarking and TF32 are
disabled, and cuBLAS workspace configuration is recorded. Repeatability is scoped
to the same hardware/software, not cross-device bit equality. See the official
[PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
and [AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

Checkpoints (`p7.checkpoint.1`) contain model specification, weights, optimizer,
scaler, epoch, best validation state, bad-epoch counter, torch/CUDA RNG state,
training config and frozen dataset checksum. `load_checkpoint` uses
`weights_only=True`, reconstructs the model and checks the expected freeze.
The runner reloads `best.pt` before held-out prediction export. No automatic
resume command is provided; `last.pt` preserves state for a deliberate P8 resume.

`run.json` (`p7.run.1`) binds source-code hashes and Git revision, model/config,
seed controls, versions/device, frozen checksum, sample IDs/groups, history,
checkpoint hash, stop reason and held-out results. Failed runs remain marked
failed. B0 records its model specification with zero parameters and no checkpoint.
Prediction manifests bind each NPZ to the sample hash, grid, spatial transform,
model variant, frozen checksum and selected checkpoint. Generated artifacts stay
under ignored output directories.

## Commands

```powershell
python -m pip install -e ".[test,audit,model]"

# Small foundation smoke: no external data, no claim of dataset accuracy.
python scripts/check_baselines.py
python scripts/check_baselines.py --device cuda --out-dir artifacts/p7_check_cuda

# On the machine with P6 generated artifacts: freeze once, reuse for all models.
python scripts/train_baselines.py freeze --out-dir data/cache/p7_fold0 --fold 0
python scripts/train_baselines.py run --frozen data/cache/p7_fold0/manifest.json --out-dir artifacts/p7_fold0 --device cuda

# With eval_samples.zip only: full fold-0 masks, no room caches or augmentation.
python scripts/train_baselines.py freeze --training-source fixed-masks --out-dir data/cache/p7_fold0_fixed --fold 0
python scripts/train_baselines.py run --frozen data/cache/p7_fold0_fixed/manifest.json --out-dir artifacts/p7_fold0_fixed --device cuda
python scripts/check_real_baselines.py
```

Use `--room-cache-dir` and `--eval-dir` when P6 binaries are stored elsewhere;
`--folds` and `--eval-spec` select their matching committed manifests. Freeze
defaults to K=8 for val/test (`--k-eval` is explicit). `run` supports `--epochs`
and `--variants`; overrides are recorded in the run manifest. All runs refuse
nonempty destinations.

## Gates and remaining work

The synthetic foundation gate freezes eight training masks, two validation
masks and two test masks across six disjoint artificial rooms. B0 runs and A/P
train for three epochs before exporting held-out predictions from the identical
freeze. CPU and CUDA AMP both passed locally. Tests additionally exercise
early stopping, checkpoint reload, deterministic reruns, masked gradients,
position padding/order and independence from changed held-out targets.

The initial synthetic gate on 2026-10-07 lacked P6 binaries. On 2026-10-09 the
provided `eval_samples.zip` supplied all 4,642 fixed masks. The real foundation
run uses complete fold 0 through `--training-source fixed-masks`: 3,256 train,
488 validation and 898 test masks, K=8, no augmentation, seed 0. This does not
require room caches and does not claim online P6 training draws or five-fold
research results. Its completed-run checker verifies every held-out prediction
against the P6 contract and sample/checkpoint checksums, plus CPU checkpoint
replay on three deterministic test indices per trained model.

Review real held-out predictions with `python scripts/review_baseline_predictions.py`;
the default picks 20 test samples uniformly with seed 0, independent of metrics.
The real gate report records run completion and visual review. P3 boundary
conventions remain those bound to P6; this module did not regenerate targets.

See `reports/p7_real_baseline_check.md` for the completed real-data gate and
`reports/p7_baseline_check.md` for the initial synthetic evidence.
