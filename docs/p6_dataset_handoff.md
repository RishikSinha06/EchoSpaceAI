# P6 dataset handoff: room caches, folds, fixed test masks, loader

P6 turns the audited rooms (P2), labels (P3), partial scans (P4) and processed
RIRs (P5) into what training and evaluation read: one cache per room,
room-group folds, a fixed set of evaluation samples, and a dataset class.
Nothing here trains a model.

## Pipeline (run in order, on the machine with the dataset)

```powershell
python scripts/build_room_cache.py      # ~45 min: data/processed/rooms/<room>.npz (all 300k RIRs)
python scripts/make_eval_masks.py       # ~1 h, resumable: data/processed/eval_samples/ + data/splits/eval_masks.json
python scripts/make_splits.py           # seconds: data/splits/folds.json
python scripts/check_dataset.py         # gate report: reports/p6_dataset_check.md
```

| Artifact | In Git | What |
| --- | --- | --- |
| `data/splits/folds.json` | yes | 5 folds of D0 duplicate groups, inner validation, checksum |
| `data/splits/eval_masks.json` | yes | every fixed evaluation sample with its content SHA-256, and per-room (type, bin) feasibility |
| `data/processed/rooms/*.npz` | no (~0.8 GB) | geometry + all processed RIRs per room (float16) |
| `data/processed/eval_samples/` | no | the fixed samples as contract v0.1.0 `manifest.jsonl` + NPZ |
| `reports/p6_dataset_check.md`, `reports/figures/p6/` | yes | gate evidence |

`make_eval_masks.py` and `make_splits.py` refuse to overwrite the committed
JSON files; pass `--replace` only for a deliberate, recorded regeneration.

## Using the dataset

```python
from echospace.data import EchoSpaceDataset, collate, load_eval_spec, load_folds

folds = load_folds("data/splits/folds.json")
spec = load_eval_spec("data/splits/eval_masks.json")
train = EchoSpaceDataset("data/processed/rooms", folds, fold=0, split="train", eval_spec=spec, seed=0)
test = EchoSpaceDataset("data/processed/rooms", folds, fold=0, split="test",
                        eval_dir="data/processed/eval_samples", eval_spec=spec, k_eval=4)
train.set_epoch(epoch)                       # new masks every epoch, reproducible
batch = collate([train[i] for i in range(8)])
```

It is plain Python, so `torch.utils.data.DataLoader(train, batch_size=..., collate_fn=collate,
num_workers=...)` works without making PyTorch a project dependency. On Kaggle,
upload `data/processed/rooms/`, `data/processed/eval_samples/` and
`data/splits/`; the meshes and source archives are not needed.

Each item is a dict:

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `observed_cells`, `observed_free`, `observed_wall` | `64x64` uint8 | P4 observation (M and G) |
| `target_occupancy`, `target_boundary`, `valid_cells` | `64x64` uint8 | P3 labels on the same grid (Y) |
| `rir` | `8x1280` float32 | P5 clean (eval) or augmented (train) waveforms, rows past K are zero |
| `rir_valid` | `8` uint8 | first `k` rows are real |
| `src_pos`, `mic_pos` | `8x3` float32 | grid frame metres `(u, v, h)`: `u = column x 0.2`, `v = row x 0.2`, `h` scene height |
| `k`, `k_requested` | int | pairs delivered / asked for |
| `coverage`, `coverage_bin`, `occlusion_type` | | P4 scan |
| `room_id`, `room_group`, `sample_id`, `grid`, `spatial_transform` | | identity and how to map back to the scene |

## Rules this module enforces

- **Splits by physical room.** Folds are dealt over D0 duplicate groups
  (`room_group` in `reports/d0_duplicates.json`), stratified by footprint-size
  tertile and convex vs. non-convex, so every fold has non-convex rooms; ~15 %
  of each fold's training groups are inner validation. A room cannot appear in
  two parts of a fold, and every group is tested exactly once.
- **Fixed test masks** (plan 1.5): per room, applicable occlusion type and
  coverage bin, up to 3 scans that have **at least 8 usable pairs**, so K =
  1, 2, 4, 8 are evaluated on the same masks (P5's nested ranking). The same
  files serve validation. Each is a validated contract sample; a loader
  refuses any whose content differs from `eval_masks.json`.
- **Usable pairs** (plan 1.4, contract rule 4): both source and receiver in
  `observed_free`; selection never reads targets or audio.
- **Training draws**: a fresh scan per item and epoch with the ±1 m frame
  shift, a (type, bin) chosen uniformly from those the room supports, and K
  uniform in 1..8. The scan is redrawn up to 8 times until it has K usable
  pairs; if none does, the best one is used with fewer pairs (`k <
  k_requested`). Then P5 augmentation and a random 90° turn plus column flip
  (plan 1.6) applied to every grid and position together.
- Rooms with no evaluation mask cannot be tested, so they are left out of the
  folds entirely (listed in `folds.json`).

## Decisions and limits

- Room caches keep **every** audited pair (300k RIRs). This became practical
  after fixing the archive reader (`e18b29d`): stored nested zips were read
  through `ZipExtFile`, which rescanned the category zip on every seek
  (509 ms per WAV, about 40 h in total; now ~1.5 ms).
- Waveforms are stored as float16 (the plan's cache format). P5's clean
  output is float32; the round trip changes values by < 1e-3 of peak.
- Validation uses the same fixed masks as testing (on different rooms).
- A training item takes ~0.5 s in one process (scan simulation dominates);
  use several `DataLoader` workers.
- No model, training loop or metric is implemented here.

## Gate

See `reports/p6_dataset_check.md`.
