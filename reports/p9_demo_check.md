# P9 small training demonstration

Date: 2026-10-11. Base: `ee983f5` (P8 merge). Branch: `module-9-training-demo`.

**Integrity/training demonstration: PASS. S learning check: FAIL. Acoustic
improvement: not established.** All five models completed training, produced
held-out scores and exported verified predictions. This user-requested small
run is not completion of P9's 75-run research study.

## Data and settings

The verified P6 fixed-mask archive supplied the samples. Seeded SHA-256 ranking
selected 20/5/5 groups inside the original fold-0 train/validation/test
partitions, without reading targets or audio. All selected-group rooms and
masks were retained. Original fold/spec files were unchanged.

| Partition | Groups | Rooms | Masks |
| --- | ---: | ---: | ---: |
| Train | 20 | 33 | 735 |
| Validation | 5 | 8 | 148 |
| Test | 5 | 9 | 171 |

Each partition contains all four occlusion types and all three coverage bins.
Room groups do not cross partitions. Every selected sample passed the original
P6 checksum/contract check and P8's audio/pose/mask checks. All 171 test masks
have eight verified available pairs.

Frozen checksum:
`cd97bb2f499c377698940eb45a46863eb85b12db3ec8847dd2056e38c418ea84`.
The [machine-readable report](p9_demo_summary.json) records selected groups,
upstream hashes, selected checkpoint hashes and the comparison results.

All five models use P8's full architecture and matched settings: seed 0,
AdamW lr 0.0003, weight decay 0.0001, batch 16, masked BCE + Dice on valid
unobserved cells, CUDA AMP, gradient clipping, three warm-up epochs, cosine
schedule, maximum 40 epochs and early-stopping patience 10. The shorter budget
is configured before training. Fixed-mask training has no augmentation.
Training draws nested K uniformly from 1..8; validation selects at K=8.
Test is scored at K=1/2/4/8 only after checkpoint selection.

## Results

The following IoUs average masks within each room, then give each test room
equal weight. They use valid unobserved cells and probability threshold 0.5;
empty unions count as IoU 1, following the existing runner.

| Variant | Inputs | Selected epoch | Completed epochs | Occupancy IoU K=8 | Boundary IoU K=8 |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Geometry | 9 | 19 | 0.650026 | 0.206469 |
| P | Geometry + positions | 10 | 20 | 0.624818 | 0.255218 |
| S | Audio + positions | 14 | 24 | 0.186380 | 0.082222 |
| C | Geometry + audio + positions, pooled fusion | 13 | 23 | 0.606377 | 0.239322 |
| D | Geometry + audio + positions, EchoFusion | 11 | 21 | 0.635104 | 0.245930 |

All runs stopped early. Training loss continued falling after validation IoU
plateaued, so the demo uses validation-selected weights rather than last-epoch
weights. No test result changed hyperparameters, selected samples or checkpoints.

D-minus-A room-averaged occupancy difference at K=8 is **−0.014922**.
Exploratory whole-group bootstrap (2,000 draws, seed 0) gives a percentile
interval **[−0.073825, 0.032995]**. Five groups and one seed are insufficient
for a general acoustic-gain claim; the interval includes zero.

S's sample-averaged occupancy IoU is **0.195724**, below the training-only
mean-shape baseline's **0.270860** on the same test masks. This learning check
fails. The result is retained; a convincing audio-only real-data learning gate
and any acoustic-gain claim remain unresolved. Sample-averaged and
room-averaged metrics are distinct and are labeled in the JSON report.

## Verification and visual review

- Full Python suite: **129 passed in 90.77 seconds**.
- Every frozen sample was checked again for content integrity.
- All five runs bind to identical frozen samples, configuration, source hashes
  and versions; selected checkpoint checksums/epochs were verified.
- Reloading each selected checkpoint reproduced every test metric row at all
  four K values **exactly (maximum score replay difference 0.0)**.
- **3,420 predictions** (5 models × 4 K values × 171 test masks) passed the
  prediction contract, with exported sample/grid/transform/checkpoint bindings.
- The learning curves and all ten occupancy/boundary comparison pages were
  inspected. Ten examples were sampled uniformly with seed 0 independently
  of scores. Registration and orientation were consistent in these examples.

The plots retain failure cases: hidden wings may be missing, inferred room
extents are rounded or too small, and boundary probabilities may respond to
interior structure. S tends to predict a diffuse central shape and fails to
recover the walls. Raw observed-cell probabilities are not trained to match
measurements because those cells are excluded from the loss; measured geometry
must remain a separate layer. These outputs are not complete room recovery.

The reporter initially caught mismatched CUDA precision settings; matching the
runner's deterministic settings fixed replay without retraining. Its prediction
validator argument order was also corrected before the successful complete
export. The failed report directory is preserved locally; the PASS report is
the `demonstration/` directory below.

## Demonstrate and reproduce

Open **`artifacts/p9_demo/demonstration/index.html`** in a browser for learning
curves, the comparison table and held-out examples. Local weights are
`artifacts/p9_demo/{A,P,S,C,D}/best.pt`; each directory also contains `last.pt`,
`run.json` and per-K test metric rows. Contract NPZ predictions and checksum
manifests are under `artifacts/p9_demo/demonstration/predictions/`.

```powershell
python scripts/train_demo.py freeze
python scripts/train_demo.py run
python scripts/report_demo.py
```

Create fresh destinations to reproduce; scripts refuse overwrites. See
`docs/p9_demo_handoff.md` and `configs/p9_demo.json`. Data, checkpoints,
prediction arrays and real-data figures remain local and ignored by Git.
The machine uses Python 3.11.9, NumPy 1.26.4, PyTorch 2.5.1+cu121 and an
RTX 4060 Laptop GPU. The exact runtime versions are logged per run.

Full multi-fold/multi-seed training, tuning and broader research metrics are
subsequent work, outside the requested small demonstration.
