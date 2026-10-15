# Full-fold A/P/D training

The user approved a larger round before P10: A geometry baseline, P positions
control and D EchoFusion on all fold-0 rooms. Settings are fixed in
`configs/p9_full_fold0.json`; this is one fold and seed, not the 75-run study.

| Partition | Rooms | Room groups | Inputs |
| --- | ---: | ---: | --- |
| Train | 171 | 99 | 20 fresh simulated masks per room per draw set |
| Validation | 27 | 18 | 488 committed clean fixed masks |
| Test | 51 | 30 | 898 committed clean fixed masks |

Ten deterministic training draw sets produce **34,200 augmented observations**,
3,420 per optimization epoch. Epoch e reads draw e modulo 10, with additional
runtime random nested K=1..8. These are multiple views of 171 rooms, not 34,200
independent rooms, and not a fresh scan redraw on every optimization epoch.

## Provenance and augmentation

Before preparation, every one of the 249 fold-room caches is checked for its
waveform checksum, expected P5 processor/config, finite arrays, room/group and
RIR identities, source waveform/metadata hashes, accepted D0 record hash and
geometry-signature binding. Cache file hashes are checked again after freezing.
Every training room must be available and feasible; unavailable rooms do not
silently disappear. The preflight also validates three real augmented draws.

Training uses P6 scans with its observation-frame shift and spatial transforms,
plus locked P5 training augmentation: gain ±6 dB, SNR 20–40 dB, low-pass
4–7.9 kHz, pose noise 5 cm and timing jitter ±0.2 ms. Both poses must remain
in observed free space. Clean signals retain emission timing, peak
normalization, 16 kHz and 80 ms. Validation/test are not augmented. No raw
archives, other dataset routes or unverified pairs are introduced.

## Training and selection

All three models use the P8 width-128, three-block backbone, batch 32, AdamW
lr 0.0003, weight decay 0.0001, five-epoch warm-up and cosine schedule, AMP on
CUDA, gradient clipping and deterministic seeds. Maximum budget is 100 epochs
per model with patience 15. BCE + Dice applies to valid unobserved cells only.
Selection uses equal-room mean occupancy/boundary validation IoU at K=8,
threshold 0.5. Actual K is capped by available verified pairs.

Test evaluation is disabled during all three fits. After all checkpoints and
the selected variant are locked on validation, every model is scored at
K=1/2/4/8. K=8 probabilities are exported and contract checked for all 898 test
masks per model. CPU replay verifies three deterministic examples per model.
The HTML review contains learning curves and six score-independent examples.
Compare D against P as well as A before attributing any difference to audio.
Earlier use of the fold-0 test pool is recorded; general acoustic-gain claims
need a broader study and independently held-out data. The earlier S learning
failure is not resolved merely by running A/P/D.

## Efficient verified loading and recovery

An optional `p8.packed.1` sidecar stores the same checked frozen arrays in
read-only memory-mapped NPY files. It avoids repeatedly opening compressed
archives without requiring the full dataset to fit in RAM. The pack is bound
to the original freeze; whole-file checksums and per-sample hashes are verified.
Returned tensors own copies, so runtime K truncation cannot alter the cache.
Tests compare packed/original arrays across K and epochs and reject edits.

Checkpoint resume restores model, optimizer, scaler, scheduler, epoch history,
CPU/CUDA RNG and early-stopping state. Source, dataset, versions, model and
configuration must match. CPU and CUDA regression tests reproduce an
uninterrupted run exactly after a simulated interruption. Completed fits are
reused; partial fits resume from `last.pt`. Incomplete data-generation or packing
directories are preserved and require a fresh version, rather than silently
accepting partial input data.

```powershell
python scripts/train_full_fold.py preflight
python scripts/train_full_fold.py all --resume
```

Use `all` without `--resume` for a completely new round. Optional `--out-dir`,
`--frozen-dir` and `--packed-dir` create a separately recorded version. Resume
requires identical paths and configuration. Keep the local machine running
for the process to progress. Data preparation can take hours; the configured
two preprocessing workers limit memory pressure on this machine.

## Local outputs

- `artifacts/p9_full_fold0/status.json`: latest actual stage/progress/error.
- `artifacts/p9_full_fold0/events.jsonl`: append-only progress log with UTC times.
- `artifacts/p9_full_fold0/preflight.json`: room-by-room provenance checks.
- `data/cache/p9_full_fold0/`: immutable augmented/fixed input freeze.
- `data/cache/p9_full_fold0_packed/`: verified disk-backed sidecar.
- `artifacts/p9_full_fold0/{A,P,D}/`: best/last weights and run histories.
- `artifacts/p9_full_fold0/selection.json`: validation-only locked selection.
- `artifacts/p9_full_fold0/results.json`: final per-K metrics after fitting.
- `artifacts/p9_full_fold0/review/index.html`: final prediction review.

`status.json` must say `complete` before claiming the full workflow is done.
All dataset arrays, weights and real-data figures remain ignored by Git.
