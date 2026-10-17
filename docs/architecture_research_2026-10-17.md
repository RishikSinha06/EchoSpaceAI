# EchoSpace next architecture: research and decision

Research date: 17 October 2026. Status: design proposal, not implemented or benchmarked.
Deployment inputs confirmed by the user: partial room scan, measured RIRs, known speaker/microphone positions.

## Decision

Recommend a compact **physics-guided wall-hypothesis fusion model** as EchoSpace v2.
Keep the current EchoFusion checkpoint as the reference. Preserve echo delay resolution,
evaluate evidence for candidate walls using image-source travel times, and fuse those
hypotheses with observed geometry at multiple resolutions. Retain a learned temporal
branch for reflections that the simple physics cannot explain.

This is the candidate I judge most promising for our data, inputs and local hardware.
It is not an established best architecture. Its proposed combination may support a
research contribution, but novelty and performance remain unverified. No full training
was launched as part of this research task.

## 1. What our existing experiment establishes

Source: `artifacts/p9_full_fold0/results.json` and `review/summary.json`.

| Model | Test occupancy IoU | Test boundary IoU |
| --- | ---: | ---: |
| A: geometry | 0.728179 | 0.415571 |
| P: geometry and positions | 0.736873 | 0.406417 |
| D: geometry, positions and RIRs | 0.749712 | 0.438172 |

These are room-averaged, K=8, over 51 rooms and 898 masks. D was selected on
validation. Training used 171 rooms, 99 room groups, and 34,200 augmented views.
Those views are not 34,200 independent room geometries.

The exploratory paired occupancy intervals, bootstrapping 30 room groups, are:

- D minus A: +0.021533, 95% percentile interval [0.007933, 0.036075].
- D minus P: +0.012839, interval [-0.003001, 0.029493].

The D/P result does not yet establish a reliable incremental waveform benefit.
These intervals also omit training-seed variation. This was one fold, one seed,
with previous project use of the test pool. Repeated masks do not add independent rooms.

The current encoder reduces each 1,280-sample RIR to one pooled token. Its time
channel enables timing information to survive, but does not guarantee preservation
of separate overlapping echoes. Whether this bottleneck causes reconstruction errors
is a hypothesis to test, not a measured diagnosis. Fusion currently occurs at the
8x8 bottleneck; boundary details must propagate through the decoder.

## 2. Closest literature and overlap

Only primary papers, official implementations, and author pages support this review.
Abstract-only access is identified. This is a targeted literature search, not an
exhaustive novelty certification. Benchmark numbers from different tasks are not comparable.

| Work | Relevant evidence | Implication for EchoSpace |
| --- | --- | --- |
| [Antonacci et al., 2012, Inference of Room Geometry](https://www.microsoft.com/en-us/research/publication/inference-of-room-geometry-from-acoustic-impulse-responses/) | Reflection arrival times constrain planar reflectors; Hough voting resolves competing assignments. Author publication abstract inspected. | Travel-time wall voting is established prior art. |
| [Dokmanic et al., 2013, Acoustic echoes reveal room shape](https://pmc.ncbi.nlm.nih.gov/articles/PMC3725100/) | Uses echo sorting and distance geometry for convex room reconstruction. | Sound-to-shape and image-source reasoning are not new; identifiability assumptions matter. |
| [Purushwalkam et al., ICCV 2021, Audio-Visual Floorplan Reconstruction](https://openaccess.thecvf.com/content/ICCV2021/papers/Purushwalkam_Audio-Visual_Floorplan_Reconstruction_ICCV_2021_paper.pdf) | Combines audio/visual features to infer floorplans beyond visible regions. Indexed paper passages inspected; direct PDF retrieval failed. | Audio-assisted completion of unseen floorplan regions is already demonstrated. Inputs differ from our scan plus positioned monaural RIRs. |
| [Tuna et al., 2023, Data-driven 3D Room Geometry Inference](https://arxiv.org/abs/2308.14611) | A Radon/beamforming representation feeds a convolutional recurrent network; evaluation includes measured RIRs. Abstract inspected. | Physics-derived representations plus learned reconstruction are established; their array geometry differs from our sparse positioned pairs. |
| [Yeon and Choi, RGI-Net, IWAENC 2024](https://arxiv.org/abs/2309.01513) | Predicts wall parameters and existence; learns from higher-order reflections when first-order echoes are absent. Latest abstract and earlier full HTML inspected. | Wall confidence and non-convex inference are prior art. A first-order-only redesign would lose valuable information. |
| [Arikan et al., JASA 2024, Reflective boundary estimation](https://arl.nus.edu.sg/wp-content/uploads/2025/03/A-deep-learning-method-for-reflective-boundary-estimation.pdf) | Hough-inspired arrival-time images plus multiresolution CNN regression; includes tank experiments. Full paper accessible. | A neural Hough wall estimator alone would have a particularly weak novelty claim. |
| [RoomFormer, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Yue_Connecting_the_Dots_Floorplan_Reconstruction_Using_Two-Level_Queries_CVPR_2023_paper.pdf) | Uses two-level transformer queries to predict room polygons from scans. | A useful structured-output comparator; polygon queries themselves are not new. |
| [PolyRoom, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06708.pdf) | Refines uniformly sampled polygon representations with room-aware initialization and attention. | Strong alternative for vector quality; adapting it introduces polygon targets and another evaluation contract. |
| [xRIR, CVPR 2025](https://arxiv.org/abs/2504.10746) | Combines geometric context and reference RIRs; introduces AcousticRooms. | Relevant data/encoder baseline, but predicts RIRs rather than hidden room occupancy. |
| [AV-DAR, ICCV 2025](https://humathe.github.io/avdar/) | Uses visual material cues and differentiable beam tracing for acoustic rendering. | Physical reflection models with learned corrections are established. Rendering complete waveforms needs more geometry/material knowledge than we have. |
| [FLAC and AGREE, CVPR 2026](https://github.com/AmandineBtto/FLAC) | Flow-matching RIR generation and joint acoustic/geometry embeddings. Official repository inspected. | Diffusion, multimodal attention and joint acoustic/geometry embedding are not sufficient novelty claims. Official full FLAC training uses an 80 GB H100, unlike our 8 GB GPU. |
| [Wang et al., JASA July 2026](https://pubmed.ncbi.nlm.nih.gov/42455750/) | Reciprocity-regularized neural acoustic field followed by acoustic-to-layout translation. Publisher-supplied abstract inspected through PubMed; full paper not obtained. | Broad physics-regularized sparse-RIR-to-layout claims already overlap. Need full-text comparison before submission. |
| [Loom, author-listed under review](https://research.ethanlyu.top/) | Author abstract describes physics-guided conditional floorplan generation from commodity speakers, ambient echoes and adaptation. PDF link exceeded browser retrieval size; abstract only. | Close competing direction. Its publication date/status and full architecture require verification; do not claim priority over it. |

Also relevant: [deep ensembles, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)
provide an established uncertainty baseline. Ensemble disagreement is not automatically
calibrated confidence, particularly under sensor/domain shift.

## 3. Why select this candidate

| Candidate | Strength for this project | Main limitation | Decision |
| --- | --- | --- | --- |
| Larger generic transformer/U-Net | Straightforward capacity comparison | Could fit room priors without using waveform evidence; weak novelty | Capacity-matched control |
| RoomFormer/PolyRoom adaptation | Explicit corners and closed polygons | New targets/matching, limited independent training shapes; no acoustic mechanism by itself | Later vector-output comparator |
| Diffusion/flow layout completion | Represents multiple plausible hidden layouts | Sampling, calibration and evaluation complexity; more uncertain local compute; not intrinsically novel | Defer unless ambiguity experiments justify it |
| Full differentiable acoustic inverse rendering | Explicit physical explanation | Needs visibility/material/source modelling; first-order approximation cannot reproduce furnished-room RIRs | Avoid as initial full reconstruction objective |
| Delay-preserving wall-hypothesis fusion | Matches known sensor poses and emission timing; testable physical mechanism | First-order model mismatch and noisy poses need explicit handling | Recommended research candidate |

This ranking is engineering judgment, not the result of a head-to-head benchmark.

## 4. Proposed v2 architecture

### 4.1 Inputs and unchanged scientific task

- Scan raster: observed free, observed wall, observed mask, metric coordinates.
- Up to 8 verified monaural RIRs with emission-time alignment, 16 kHz, 80 ms.
- Source/microphone 3D positions transformed into the observation-defined grid frame.
- Same 64x64 occupancy and boundary probability outputs, evaluated on valid M=0.
- Real users must measure/deconvolve RIRs; arbitrary phone audio is not an RIR.
- Actual capture calibration must establish the scene frame, gravity, pose accuracy,
  source/receiver response, sample rate and emission-time reference.

Keep P5 normalization and augmentation fixed in the first architectural comparison.
Longer RIR windows or alternative amplitude normalization require separately versioned
data experiments. The 80 ms window is not adequate evidence for a full reverberation-time estimate.

### 4.2 Geometry branch

Use a residual encoder/decoder with a multiresolution feature pyramid. Start with
the existing 32/64/128/128 channel budget for a fair control. Inject acoustic
features at 8x8 and 16x16; the dense head stays at 64x64. Do not assume every
wall is axis-aligned, nor reduce concave rooms to intersections of half-planes.

### 4.3 Delay-preserving acoustic branch

Encode each RIR as a sequence of time-local features with explicit absolute time.
Start with a high-resolution shallow branch and a strided context branch; retain
their mapping to sample-time centres and receptive fields. Avoid global averaging
before geometry fusion. Band-energy/envelope features can supplement the signed
waveform; their windowing must not leak test-derived statistics.

Delay resolution must be validated with impulse pairs, overlapping echoes, fractional
delays, filtering and the locked timing/pose augmentations. More tokens alone are
not proof that wall distances can be recovered accurately.

### 4.4 Candidate-wall travel-time layer

For a unit horizontal normal n=(cos(theta), sin(theta), 0) and offset rho, a vertical
candidate wall is n dot x = rho. Reflect a 3D source s across that plane:

    s_image = s - 2 * (n dot s - rho) * n
    t_reflection = ||s_image - microphone|| / c

Use interpolation to sample learned delay features near this predicted arrival.
Use uncertainty-width kernels rather than an exact-time hard match. Nominal c must
match the audited simulator; measured deployment c requires calibration/temperature
handling. Do not learn unconstrained speed/timing offsets that can explain arbitrary walls.

This is a first-order infinite-plane hypothesis, not proof that a physical wall exists.
Require appropriate source/microphone sidedness, reject arrivals outside the waveform,
and retain a learned validity/confidence term. A finite wall additionally needs a valid
specular reflection point and visibility; candidate segment support must be checked
where available. Missing reflections are weak evidence, not automatic wall absence.

Use an orientation/offset bank with coarse-to-fine continuous refinement. Size and
angular/offset spacing must be profiled and tested against the 0.20 m grid; they are
not fixed by this research memo. Chunk hypothesis evaluation to bound GPU memory.
Use full 3D path lengths even though outputs are 2D. Never use hidden target walls
to construct inference-time hypotheses.

### 4.5 Evidence across measurements and learned residuals

Aggregate per-pair evidence with permutation-invariant, validity-masked set attention.
Wall hypotheses gain support when multiple positioned RIRs agree on the same plane.
Condition fusion on observed geometry, with a null/unexplained-echo option so floor,
ceiling, furniture, scattering and higher-order reflections are not all forced into
vertical walls. Keep a separate learned temporal route for those unresolved cues.

Learn a reliability gate for the acoustic correction to the geometry prediction.
Train on missing/noisy audio and retain an explicit geometry-only mode for K=0.
This requires changing the current D forward contract, which demands at least one
valid pair. The gate and fallback must be tested; neither guarantees uncertainty awareness.

### 4.6 Dense prediction and uncertainty

Predict occupancy and boundary with the existing loss and output contract first.
Then ablate a signed-distance auxiliary head, supervised only in eligible missing
cells, to encourage coherent boundaries. Ground-truth-derived distances are training
targets, never inference inputs. No hard Manhattan or convexity constraint.

Return raw missing-region probabilities separately from measured geometry; a display
adapter can compose them without overwriting measured observations. A new uncertainty
field needs its own versioned response/metadata contract. Initially compare calibrated
single-model confidence with a three-seed ensemble and measure actual selective risk.

Implementation target: a few million parameters and one-GPU training. No parameter
count, memory claim or latency claim is established until implementation and profiling.
Prefer PyTorch operations already available; ask before introducing dependencies.

## 5. What could be a contribution

Working hypothesis: observation-conditioned fusion of continuous delay features and
candidate-wall reflection evidence, with pose/timing uncertainty and rejection of
unexplained echoes, can improve unseen-region completion over generic fusion.

The potentially distinct contribution is this specific mechanism in the stated
partial-scan/sparse-positioned-RIR task, supported by controlled evidence. It is not
the invention of U-Nets, wall queries, Hough voting, image-source physics, attention,
confidence heads, signed distances or acoustic floorplan inference.

Before claiming novelty: obtain full text of the closest 2026 JASA and Loom work,
trace references/citations, and compare their operators, inputs, supervision and
evaluation. If the mechanism substantially overlaps, describe our work as an adaptation
and focus the contribution on the benchmark, controls or demonstrated robustness.
Absence from this search does not establish absence from the literature.

## 6. Training and research protocol

### Stage A: development pilot

1. Preserve all old checkpoints and manifests; put v2 in new versioned directories.
2. Audit verified pair provenance, duplicate groups, room eligibility and physical
   timing. Record which room groups have appeared in prior training/evaluation.
3. Check synthetic reflection timing, coordinate transforms, gradient correctness,
   invalid-pair masking, pair-order invariance and K=0 behavior.
4. Compare current D, temporal-token-only D, and the proposed physics-guided model
   on identical development groups and equal tuning budgets. Start with three seeds.
5. Compare geometry-only and positions-only versions of the new backbone so extra
   capacity cannot masquerade as an acoustic contribution.
6. Profile preprocessing, wall-bank memory, batch size and training throughput on
   the actual RTX 4060 8 GB. Estimate the larger study from measured epoch times.
7. Pilot on development data only; choose architecture/loss using validation.

Do not immediately run a very large study of an unverified physical operator.

### Stage B: controlled ablations

- Temporal tokens versus single pooled token.
- Learned-only fusion versus matched-capacity wall-guided fusion.
- With/without the uncertainty-width travel-time kernel.
- With/without the learned residual/null-echo route.
- With/without multiscale injection and distance auxiliary loss, one change at a time.
- K=0,1,2,4,8; source/receiver positions only; waveform content withheld.
- Validation-only diagnostics for deliberately shuffled/misaligned waveforms while
  preserving masks and positions. Mark those as corrupted controls, never verified
  training pairs or deployment inputs. Do not permute matching position/RIR rows
  independently in normal data augmentation.

### Stage C: full internal study

Freeze architecture and search budget before final outer-fold scoring. Use five
room-group folds and three seeds for the old D, new positions control, and new full
model: 45 core fits, with geometry and selected ablations budgeted explicitly.
Each outer fold uses only its own inner validation for checkpoints, thresholds,
calibration and selection. A confirmatory nested search would repeat tuning inside
each outer training partition. A single globally tuned architecture on this already
explored dataset is an internal estimate and must be labelled accordingly.

Use all eligible training groups in each fold; do not train on that fold's validation
or test groups. Generate deterministically seeded new scan/augmentation draws each
epoch where affordable; record unique draw counts. More derivatives do not substitute
for independent room groups. Retain P5 timing, normalization and augmentation for
matched comparisons. Start optimizer tuning from the current AdamW/warmup/cosine
recipe; bound trials and tune loss weights exclusively on development validation.

Primary selection: predeclared room-averaged mean occupancy/boundary IoU on M=0.
Report both heads separately, room/group weighting, seed variability and paired
group-level confidence intervals. Do not treat masks, seeds or folds sharing
training rooms as independent new room evidence.

### Stage D: independent measured-room evaluation

The existing room pool has prior project exposure. Merely reshuffling it does not
create a pristine final test. Obtain additional independently captured room groups
with permitted mesh/scan/RIR pairings; freeze them before inspecting predictions.
Keep real calibration/development rooms separate from the final measured test rooms.
HAA/RAF or another public collection must pass license, pose, geometry and single-room
eligibility checks first. They are not automatically drop-in compatible, and paused
downloads must not be resumed by this plan.

A practical initial collection target is 10-20 diverse measured rooms across more
than one device setup, subject to feasibility and a precision/power assessment.
That count is a pilot target, not evidence of adequate population coverage. If only
simulation is available, release claims must remain simulation/research claims.

After evaluation and selection, train the deployment model on the declared permitted
development pool. Keep calibration groups independent of fitted weights and the final
measured test locked. This deployment fit is separate from cross-validation checkpoints.

## 7. Deployment acceptance gates

These are proposed measurable gates, not achievements or universal production standards.

| Gate | Required evidence |
| --- | --- |
| Waveform contribution | Repeatable improvement over the new positions control and old D, including seed variation and paired intervals; report null/negative outcomes honestly. |
| Shape quality | Both missing-region IoUs; boundary F1 at predeclared metric tolerances; distance error in metres; disconnected components and invalid polygon rate if vectorization is used. |
| Generalization | Frozen independent measured-room evaluation, plus category/size/concavity/coverage and device strata. |
| Robustness | Missing pairs, clipped/low-SNR responses, pose/timing/speed-of-sound errors and device response mismatch; corruption settings fixed before final scoring. |
| Observations | Measured geometry preserved by construction in composed output; raw completion output independently scored. |
| Confidence | Validation-calibrated probabilities, Brier score/reliability curves and risk versus retained coverage; verify rejection behavior under domain shift. |
| Runtime | Batch-one CPU/GPU p50/p95 latency, preprocessing included, peak memory and repeated-load behavior on declared target hardware. Set the actual latency budget with the deployment owner. |
| Reproducibility | Immutable data/split hashes, code revision, locked environment, seed records, resume tests, model card and checksum-bound inference bundle. |
| Application | Input validation, K=0 fallback, model/preprocessing version binding, error handling, uncertainty display, API/viewer integration and rollback to the current checkpoint. |

Training longer or obtaining a larger checkpoint does not satisfy these gates.
No promise of perfect shape recovery is appropriate for ambiguous sparse evidence.

## 8. Immediate next implementation milestone

Build the temporal encoder and differentiable candidate-wall evidence layer behind
separate model/config versions; keep v1 loadable. Demonstrate physical correctness
and a small three-seed validation comparison before starting the larger study.
Do not replace the released sharing ZIP or relabel this proposal as a trained model.
