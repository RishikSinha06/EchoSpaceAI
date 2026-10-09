# P8 EchoFusion handoff

P8 implements EchoFusion (model D) and, from the same network, the plan's
siblings A (geometry only), P (positions, no audio), S (audio only) and C
(simple pooled fusion), with a training runner and a freeze that carries the
P5 waveforms. P7's B0/A/P baselines stay as the foundation comparison; P8's A
and P share EchoFusion's backbone so that "does sound help?" compares like
with like.

## Model (`src/echospace/models/echofusion.py`)

| Variant | Geometry | Audio | Positions | Fusion | Parameters (default) |
| --- | --- | --- | --- | --- | ---: |
| A | yes | no | no | self-attention only | 1.67 M |
| P | yes | zeroed | yes | cross-attention | 2.09 M |
| S | coordinates only | yes | yes | 64 learned queries cross-attend | 2.10 M |
| C | yes | yes | yes | pooled + concat at the bottleneck | 1.92 M |
| D | yes | yes | yes | cross-attention (EchoFusion) | 2.09 M |

Spatial encoder 32 -> 64 -> 128 -> 128 (64, 32, 16, 8 cells) on 5 channels
(observed free, observed wall, observed mask M, x/y metres); the 8 x 8
bottleneck is 64 tokens with a 2D Fourier encoding of their centres. Shared
acoustic 1D CNN (kernel 7, stride 2, 32 -> 64 -> 128 -> 128, global average
pool) per RIR plus a Fourier MLP of the six source/mic coordinates, summed into
one token per measurement. 3 pre-norm fusion blocks (cross-attention with 4
heads and a key padding mask for invalid rows, self-attention, feed-forward).
Transposed-convolution U-Net decoder, occupancy and boundary heads (64 x 64).

Guarantees (tested): invalid RIR rows and measurement order never change the
output; A ignores audio and positions, P ignores audio, S ignores geometry; no
forward path reads targets, `valid_cells`, room identity, coverage or split.

### Deviation from the plan: the time channel

The plan's acoustic encoder (convolutions, then global average pooling over
time) is translation-invariant in time. An echo at sample 300 and the same
echo at sample 700 produce features that differ by 1e-6, so reflection delay,
the physical quantity that encodes wall distance, cannot reach the model. On
band-limited synthetic echoes the plan's encoder learns echo delay with
R^2 = -0.02; with a second input channel holding normalised time t/T it reaches
R^2 = 0.99 (attention pooling over time gave 0.99 too, so the simpler fix was
kept). `time_channel` is on by default and recorded in every model spec.

## Data (`src/echospace/training/audio_frozen.py`)

`p8.frozen.1` = P7's frozen arrays + the 8 nested P5 waveforms (float16) +
occlusion type / coverage bin. Training draws come from either

- `online` (default, needs the P6 room caches): E epochs of P6 online draws
  (fresh scan, +-1 m shift, P5 and spatial augmentation), materialised once;
  training epoch e reads draw e mod E. Default E = 10, 20 masks per room
  (~34 k samples, ~0.85 GB, ~40 min with 6 workers); or
- `fixed-masks`: the fixed P6 masks of training rooms, no augmentation (when
  only `eval_samples` is available, as on the P7 machine).

Val/test are always the committed P6 fixed masks. Random K (uniform 1..8 per
sample and epoch) zeroes rows >= K, keeping P5's nested prefix.

## Training (`src/echospace/training/fusion_runner.py`)

AdamW lr 3e-4, weight decay 1e-4, 5-epoch warm-up then cosine decay, batch 32,
gradient clipping at 1, AMP on CUDA, P7's masked BCE + Dice (valid unobserved
cells), early stopping on inner-validation missing-region occupancy IoU at
K = 8 (patience 15). Test is read only after selection and scored at
K = 1, 2, 4, 8 on the same masks, with per-sample rows (`test_rows_K*.json`)
for P9's room-weighted statistics. `mean_shape_baseline` is the plan's trivial
predictor for the S learning gate.

```powershell
python -m pip install -e ".[test,audit,model]"
python scripts/check_echofusion.py                       # synthetic gate, ~1 h on CPU
python scripts/train_echofusion.py freeze --out-dir data/cache/p8_fold0 --fold 0 --workers 6
python scripts/train_echofusion.py run --frozen data/cache/p8_fold0/manifest.json `
    --out-dir artifacts/p8_fold0_seed0 --device cuda --seed 0
```

Speed: on this machine's CPU (no GPU) D trains at ~30 samples/s, A at ~40;
one fold-0 epoch of 3 420 draws takes ~2-3 min. The plan's study (5 configs x
5 folds x 3 seeds = 75 runs) needs a GPU.

## Gate

Synthetic gate **passed** (`reports/p8_echofusion_check.md`): test occupancy IoU
at K = 8 was A 0.686, P 0.685, S 0.939, C 0.966, D 0.779, mean shape 0.687.
D learned slowest (plateau ~35 epochs, still improving at 40); keep the plan's
budget and patience for D, and tune it before Experiment C. Real-data training
is not yet run (needs a GPU).

## Not done here

- Hyperparameter search (plan: Optuna on fold-0 inner validation, ~20
  trials). Optuna would be a new dependency; not added.
- The 75-run study and its statistics (P9).
