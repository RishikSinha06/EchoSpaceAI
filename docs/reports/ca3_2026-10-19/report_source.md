# EchoSpace AI
## Room Footprint Reconstruction from Partial Scans and Acoustic Responses

Deep Learning Project Report
CA3 Evaluation
19 October 2026

Project title    ____________________________________________________

Team member names and roll numbers
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

College and department    ___________________________________________

Course and section    _______________________________________________

Teacher or guide    _________________________________________________

Submission date    _________________________________________________

This report presents the implemented research prototype, its completed experiments, and the software sensor simulation layer. The central task is to predict the missing two-dimensional footprint of one room while preserving the distinction between observations and predictions.
===PAGE===
# Abstract and keywords

A partial room scan records what a sensor can see; it does not automatically describe what lies behind an occlusion. We investigate whether paired room impulse responses, together with known speaker and microphone positions, can improve reconstruction of the missing room footprint. Our system, EchoSpace AI, converts audited geometry and acoustic observations into a 64 by 64 metric grid and predicts room interior and structural boundary probabilities only where the scan is unknown.

We implement an archive-based AcousticRooms adapter, room-grouped splits, partial-scan generation, emission-referenced acoustic preprocessing, geometry and position controls, and a cross-attention fusion network. The completed EchoFusion A/P/D experiment uses all 171 training room IDs in fold 0 and evaluates 898 fixed scans from 51 test room IDs in 30 physical-room groups. Its training, checkpoint selection and evaluation are complete. The separate WallFusion study is ongoing and excluded from final performance claims. At eight RIRs, the geometry model achieves occupancy IoU 0.7282 and EchoFusion achieves 0.7497. However, the incremental comparison against the positions control is uncertain, and zeroed or swapped RIRs have almost no adverse effect in a later diagnostic evaluation. These controls prevent us from attributing the full improvement to waveform information.

A local API and browser viewer support matched comparisons and export to a metric grid, GLB and provenance-preserving RoomScene. We also implement exact dataset replay and an interactive virtual sensor lab with partial ray scans, synthetic acoustic capture and configurable device faults. Its first-order acoustic model is explicitly a software integration tool, not validated physical hardware. The result is a reproducible research prototype with a positive broad reconstruction comparison and a clearly identified open question about genuine acoustic use.

Keywords: room reconstruction; partial scans; room impulse response; multimodal learning; U-Net; cross-attention; data leakage; sensor simulation.

## Reading guide

Introduction appears on page 3; literature review on pages 4-5; methodology on pages 6-13; results and discussion on pages 14-19; simulation and implementation verification on pages 20-21; interpretation on page 22; conclusion on page 23; references on page 24.
===PAGE===
# 1 Introduction
## 1.1 The room beyond the scan

Consider standing in a classroom doorway with a depth scanner. The near wall and part of the floor may be visible, while a cupboard, a narrow opening or the viewing angle hides the far corner. A conventional scan can represent this uncertainty correctly, but it cannot turn an unseen corner into a measurement. Treating every unobserved cell as empty would create a very confident map of a room we have not actually seen.

Sound offers another source of evidence. After a known excitation is emitted, the microphone receives a direct arrival followed by reflections. Their timing depends on propagation paths, and those paths depend partly on room geometry. The important word is partly: material, occlusion, device response and measurement conditions also affect the recording. We therefore ask a testable question rather than assuming that adding an audio branch must help.

## 1.2 Research question and objectives

Our research question is: on the same held-out room observations, does a model receiving geometry, positioned RIRs and valid-pair masks reconstruct unknown room area better than a geometry-only model? A second question asks whether any improvement comes from waveform content rather than simply knowing where speakers and microphones were placed.

We pursue four objectives. First, establish trustworthy geometry-RIR identity and coordinate conventions before training. Second, construct observations whose unknown region remains genuinely unknown. Third, compare matched geometry, positions and fusion models with room-aware statistics. Fourth, make a saved prediction inspectable through an API, a browser and a native 3D export. The simulation layer extends the fourth objective by letting us vary virtual acquisition conditions without claiming that we have already built calibrated hardware.

## 1.3 Scope and possible use

The learning target is a single-room 2D footprint and boundary. We do not reconstruct furniture, ceiling shape, an entire building or a safety-certified navigation map. The 3D walls shown by the viewer are an extrusion with an assumed display height.

Potential uses include a research assistant for incomplete indoor mapping, preparation of spatial-audio scenes, and inspection of alternative room completions. In a robot workflow, a predicted region could suggest where to scan next; it should not authorize motion through an unseen space. The system's job is to make missing geometry inspectable, not to promote a plausible guess into a measured fact.

Our contribution is the integrated, audited and controlled pipeline. We do not claim that room inference from echoes, U-Nets or multimodal attention are new inventions.
===PAGE===
# 2 Literature review
## 2.1 What echoes can reveal

Dokmanic and colleagues show that echo arrival relationships can recover convex polyhedral room geometry under specific measurement assumptions [1]. Echo sorting is central: a detected peak must be associated with the correct reflector. This work explains why an RIR can contain geometric information and why a collection of peaks is not already a floor plan. Our task differs because we use partial geometry and learn a raster completion rather than solving their complete geometric reconstruction problem.

A useful physical picture is the image source. A reflected path can be represented as a straight path from a mirrored source to the receiver. This converts a candidate wall into a predicted delay. It also exposes the assumptions: a finite wall must support the reflection, the path must be visible, and the echo must be distinguishable. A nonconvex room may violate those conditions even when the candidate infinite plane appears geometrically plausible.

## 2.2 Learning beyond visible regions

Purushwalkam and colleagues combine audio and visual information for floorplan reconstruction beyond directly visible areas [2]. Their work is close to our motivation, but its visual input and environment-level task differ from our scanner-aligned, single-room grid with positioned monaural RIR pairs. We use it to justify a controlled multimodal question, not as a numerical benchmark for our IoU. Its evaluation protocol and ours are not interchangeable.

Yeon and Choi's RGI-Net estimates room geometry when first-order reflections can be hidden and the room may be nonconvex [3]. This is relevant to our later wall-guided model: a first-order physical branch cannot explain every useful acoustic cue. A learned temporal route is therefore retained rather than assuming that all informative reflections fit one simple wall equation.

## 2.3 Dataset relevance

Liu and colleagues introduce xRIR for cross-room RIR prediction and provide AcousticRooms, containing simulated RIRs across 260 rooms [4]. Their task predicts acoustic responses, while ours predicts missing geometry. We reuse the dataset only after checking local mesh, metadata, timing and room identity. The official repository identifies its RIRs as simulated and provides a CC BY 4.0 licence [10]. Calling these recordings measurements made by our team would misrepresent the experiment.

The literature establishes the physical possibility and multimodal motivation. It does not settle whether our particular encoder, observation distribution and available room diversity will make effective use of RIRs. That is the question our controls must answer.
===PAGE===
# 2 Literature review continued
## 2.4 Architectural choices and their limits

U-Net combines a contracting encoder with a decoder and skip connections [5]. We use this pattern because room completion needs both spatial context and local boundary detail. A bottleneck can capture a broad footprint prior; skip connections can retain information from measured walls. U-Net's original biomedical application is not a claim of room-reconstruction performance.

Deep Sets formalizes learning over unordered sets [6]. Our acoustic measurements have no meaningful first or last microphone-pair order, so the representation must respect this symmetry. Shared pair encoders and validity-aware aggregation prevent the network from exploiting arbitrary row positions. Attention provides a way for a spatial query to combine relevant encoded measurements [7]; an attention weight by itself is not a physical explanation of a wall.

| Prior work | What we take from it | What still needs testing here |
| --- | --- | --- |
| Echo geometry [1] | Delays constrain reflectors | Noisy, ambiguous and hidden echoes |
| Audio-visual floorplans [2] | Complementary modalities | Scan plus positioned RIR inputs |
| RGI-Net [3] | Higher-order evidence can matter | Sparse-pair footprint completion |
| xRIR and AcousticRooms [4] | Paired room acoustic data | Reverse task and local data integrity |
| U-Net [5] | Spatial encoder and skip decoder | Unknown-region generalization |
| Deep Sets and attention [6,7] | Set symmetry and feature fusion | Actual waveform dependence |

## 2.5 Research gap and our positioning

Our practical gap is the absence of a trustworthy end-to-end answer for this exact input contract: a partial metric room scan, a small set of paired RIRs, and known source/receiver poses. We address the gap through audit, observation-defined coordinates, matched controls, and deployable inspection. This is a project-level integration and evaluation contribution, not an assertion of first publication priority.

We also distinguish reconstruction quality from modality use. A larger fusion network could outperform a geometry network because of capacity or position information while largely ignoring audio. The positions control and audio interventions are necessary for that reason. A good comparison needs an inconvenient baseline occasionally; otherwise the experiment becomes a demonstration of our preferred model rather than a test of it.

The wall-guided extension tests a possible temporal bottleneck. Its benefit remains unverified, so its design is kept separate from the completed A/P/D results.
===PAGE===
# 3 Methodology
## 3.1 From raw assets to an inspectable completion

Our pipeline has one offline data and training path and one local inference application. This keeps the preprocessing contract shared between experimentation and demonstration. A browser display adapter is useful for viewing; it is not a replacement for a versioned sample with metric coordinates and provenance.

![Figure 1 Overall EchoSpace pipeline and the separation of model inputs from reference labels](figures/pipeline.png|6.55)

Raw meshes, pair metadata and RIRs first pass the identity and geometry audit. A mesh slice supplies room labels. A declared scanner pose and heading define the grid, after which a partial-scan simulator produces observed free cells, wall hits and visibility. Acoustic preprocessing creates emission-referenced waveforms and nested measurement sets. Room groups are split before augmented observations are generated.

The network predicts two probability maps. Evaluation uses the unknown part of the valid grid; final visualization combines predictions with the measured context. The same model-facing tensors are used by the inference service. Dataset replay loads those tensors exactly, while virtual mode creates a separately labeled synthetic capture through the same adapter.

## 3.2 Formal task

Let G be the five-channel observed geometry raster, M the observed-cell mask, H the set of K waveforms, P the corresponding source/receiver poses, and V the valid-pair mask. The model computes f(G, M, H, P, V) = (p_interior, p_boundary), with each output map sized 64 by 64. The supervised region is U = valid_cells AND NOT M.

Full room labels, room IDs, split labels, coverage values and reference mesh bounds are not forward inputs. They support data administration and evaluation. This separation is checked by changing target arrays while requiring unchanged model-input tensors.

Our output is intentionally modest: an estimated footprint with explicit observation context. A technically correct unknown mask is more useful than a visually complete room whose missing parts have quietly been treated as measurements.
===PAGE===
# 3 Methodology continued
## 3.3 Dataset description and acceptance

We use AcousticRooms as our primary study source [4,10]. The required assets are room meshes, per-pair source/receiver metadata and single-channel RIRs. The adapter reads nested archives without bulk extraction and joins assets by the original scene and pair identity. A similar filename alone is not acceptance evidence.

| Local audit or pipeline stage | Count | Interpretation |
| --- | ---: | --- |
| Candidate scene entries examined | 260 | Initial inventory |
| Scenes with mesh and paired assets | 258 | Assets can be joined |
| Scenes accepted by the D0 audit | 257 | Basic data eligibility |
| Scene IDs in the fixed 12.8 m study | 249 | Canvas and observation eligibility |
| Physical-room groups in those splits | 147 | Duplicates remain together |
| Frozen evaluation scans across folds | 4642 | Repeated masks, not new rooms |
| Cached paired RIRs for eligible scenes | 271599 | Acoustic diversity, not independent layouts |

Two candidate entries lack meshes and one fails the room-membership check. Eight accepted scene IDs do not yield the fixed-canvas observations used by the study. Grouping by physical room or identical footprint prevents copies and derivatives of one layout from crossing training, validation and test.

The original OBJ convention has Z vertical. We explicitly map source coordinates into the canonical Y-up scene frame as (x, y, z) to (x, z, -y), with metric units. Source and receiver locations must overlay the same room geometry in that frame. The timing audit distinguishes visible direct paths from blocked ones; the absence of a strong direct peak on an occluded path is not treated as a global onset convention.

Mesh validity is task-specific. A room surface can be nonwatertight yet provide a defensible closed horizontal footprint; a watertightness flag alone would therefore reject useful data. We instead check the slice, room membership, coordinate transform and clipping rule. Furniture loops and internal structure require explicit geometric treatment rather than silent relabeling.

AcousticRooms data and derived figures retain attribution to Meta Platforms and the source authors under CC BY 4.0 [10]. Raster labels and overlays are our adaptations. The project's MIT code licence does not replace the data licence. Other dataset candidates are not silently pooled into this experiment.

Evidence source: local D0 audit summary, P6 dataset report and full-fold preflight. The counts refer to successive filters and must not be added together.
===PAGE===
# 3 Methodology continued
## 3.4 Geometry labels and partial observations

The geometry stage slices a candidate mesh around 1.0-1.2 m above the floor and obtains a defensible connected single-room footprint. It rasterizes interior and structural boundary labels at 0.20 m per cell. A 64-cell side therefore represents 12.8 m. Oversized or ambiguous examples are excluded rather than resized in a way that changes the physical scale.

The crop is anchored to the declared initial scanner pose and heading. It is never centered on the full hidden room bounds. That restriction matters: a crop chosen from the true room could reveal its extent before the model sees a single observation.

![Figure 2 Actual partial-scan diagnostic from the project showing observation and reference geometry](figures/partial_scan.png|6.55)

We simulate viewpoint, missing-wall, doorway and hidden-L-wing observations. Coverage targets are approximately 80%, 50% and 30%, with finite resampling limits and recorded exclusions. Some room/occlusion combinations cannot achieve a valid observation; the pipeline does not manufacture them to balance a table.

Observed free space records traversed cells, observed wall records sensor hits, and visibility records what was actually observed. Unknown cells are neither free nor occupied in the input. Labels describe the complete reference room only for supervision and scoring. Both selected acoustic endpoints must lie in observed free space for the corresponding mask.

Test masks are frozen before comparison and reused across A, P and D. Derivatives inherit their original room group's split. A model cannot receive a friendlier test scan because its first prediction looked poor. We also retain low-coverage and nonconvex examples: these are precisely the situations where completion is difficult and acoustic assistance would be useful.

Figure 2 is a recorded pipeline diagnostic, not a newly drawn example of an ideal outcome. Reference geometry in such figures is shown for inspection; it remains outside the model input.
===PAGE===
# 3 Methodology continued
## 3.5 Acoustic preprocessing and input contract

The local source RIRs are mono 22.05 kHz integer waveforms. We decode the verified pair, retain timing evidence, resample to 16 kHz, and cache an 80 ms window containing 1280 samples. Clean preprocessing preserves emission timing and the source-to-receiver travel delay. We do not shift every direct arrival to sample zero, because doing so would remove a useful metric cue.

Peak normalization is the locked default for the completed full-fold experiment. It reduces dependence on arbitrary file amplitude, while also discarding absolute loudness information. This is a deliberate trade-off, not proof that source power and device calibration no longer matter. The short window represents early acoustic response; it cannot support a reliable full reverberation-time claim.

| Batch field | Shape | Meaning |
| --- | --- | --- |
| G | B x 5 x 64 x 64 | Free, wall, observed mask, metric grid coordinates |
| M | B x 1 x 64 x 64 | Observed region |
| RIR | B x K x 1280 | Emission-referenced waveforms |
| Source and microphone poses | B x K x 2 x 3 | Metric positions in the observation frame |
| RIR valid | B x K | Real pair versus padded row |
| Interior and wall targets | B x 64 x 64 | Reference labels for loss and evaluation |

Measurement sets are deterministic and nested for K = 1, 2, 4 and 8. Increasing K extends the same ordered set rather than choosing an unrelated group of pairs. Invalid padded rows are masked in encoding, attention and aggregation.

Training-only perturbations use gain from -6 to +6 dB, SNR 20-40 dB, low-pass cutoff 4.0-7.9 kHz, position noise with a 5 cm scale and timing jitter up to 0.2 ms. Position perturbation retains observation eligibility. Test and validation inputs remain unaugmented. The upper cutoff is strictly below the 8 kHz Nyquist frequency at 16 kHz sampling.

A later robustness evaluation applies some interventions after preprocessing, which is not the same as retraining with another preprocessing rule. We keep those two interpretations separate. In particular, a 40 ms truncation test is a stress test of an 80 ms-trained model, not a completed 40 ms training experiment.
===PAGE===
# 3 Methodology continued
## 3.6 EchoFusion architecture and controls

The completed reference network has a shared spatial encoder with channels 32, 64, 128 and 128 at resolutions 64, 32, 16 and 8. Its bottleneck contains 64 spatial tokens of width 128. A decoder restores the full grid through transposed convolutions and skip connections, with separate occupancy and boundary logits.

Each RIR passes through the same 1D convolutional encoder. Four stride-2 stages use kernel size 7 and channel widths 32, 64, 128 and 128, followed by pooling. A time channel preserves an explicit temporal coordinate. A Fourier-feature pose encoder describes the source and microphone in metric grid coordinates. Pair features form valid acoustic tokens; their arbitrary order is not a learned scene identifier.

![Figure 3 EchoFusion input branches and the matched comparison variants](figures/architecture.png|6.55)

In D, spatial tokens query acoustic tokens through three fusion blocks with four attention heads and a feed-forward width of 256. The spatial representation also uses position encoding and self-attention. This is the saved full-fold configuration; it is more precise than repeating an earlier plan that allowed two fusion layers.

| Model | Geometry | Positions | RIR content | Parameters |
| --- | --- | --- | --- | ---: |
| A | Yes | No | No | 1674818 |
| P | Yes | Yes | Zero waveform | 2091298 |
| D | Yes | Yes | Correct waveform | 2091298 |

A is the geometry reference. P tests whether endpoint locations and the fusion structure account for an improvement. P and D share parameter count, making their comparison especially useful. B0 wall extrapolation is implemented as a nonlearned baseline; C pooled fusion and S acoustics-only variants exist in the code and earlier small demonstrations. We do not insert uncompleted full-study C/S values into the A/P/D table.

For display, verified observations remain distinct from inferred cells. Quantitative reconstruction scores use raw predictions in unknown valid cells, so pasting the measured context into a rendered image cannot inflate the learned reconstruction score.
===PAGE===
# 3 Methodology continued
## 3.7 WallFusion as an ongoing extension

The second architecture investigates a specific weakness of the reference design: pooling each RIR to one token may make closely spaced echoes difficult to use. This is a hypothesis about representation, not a diagnosed explanation of every failure. The wall-guided model retains a temporal feature route and samples acoustic evidence at delays predicted by candidate wall planes.

For a candidate vertical plane, we mirror the source across the plane, then compute the distance from that image source to the receiver. Dividing by sound speed gives a predicted reflection delay. Temporal features near that delay provide evidence for the wall hypothesis. Same-side checks reject physically incompatible source/receiver placement. Candidate planes are hypotheses, not observed walls.

The implemented configuration uses temporal tokens, a 32 by 32 wall-query grid, 32 sampled orientations and multiscale feature injection. A learned residual branch remains available for acoustic patterns that an infinite-plane first-order model cannot explain. This matters in furnished or nonconvex spaces where finite visibility, higher-order reflections and non-wall interactions differ from the simple model.

There are two separate uses of physics in the project. WallFusion uses an infinite-plane hypothesis layer inside a learned model. The interactive simulator instead validates visible finite reflection paths in a known virtual polygon. A successful simulation capture does not validate the learned wall-hypothesis branch on a real room.

## 3.8 Completed experiment and ongoing study

| Experiment | Status | Role in this report |
| --- | --- | --- |
| EchoFusion A/P/D on full fold 0 | Completed training, selection and evaluation | Source of the reported comparative results |
| Larger EchoFusion and WallFusion grouped study | Ongoing and paused | Architectural extension; no final ranking reported |

Here, full-fold means all eligible training room IDs in fold 0. It does not mean that all five folds and seeds of the larger study are complete. The unfinished extension does not invalidate the completed A/P/D experiment.

At the reporting snapshot, the larger study has 21 completed pilot fits and 76 of 81 planned full fits. These counts describe progress, not a final result. We do not treat the last saved epoch as proof of completion. A future interim evaluation would need a declared stopping point, validation-selected checkpoints and matched folds, seeds and test samples; no such WallFusion comparison is claimed here.
===PAGE===
# 3 Methodology continued
## 3.9 Training procedure and model selection

The completed full-fold experiment trains A, P and D on the same room split and frozen data identity. Fold 0 contains 171 training room IDs in 99 groups, 27 validation IDs in 18 groups, and 51 test IDs in 30 groups. Room derivatives and duplicates stay in the same split. The training cache contains ten augmented draw sets with 20 observations per training room, giving 34200 views; these are not 34200 independent rooms.

| Setting | Recorded value |
| --- | --- |
| Hardware | NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM |
| Framework | PyTorch 2.5.1 with CUDA 12.1 |
| Optimizer | AdamW [8] |
| Initial learning rate and decay | 0.0003 and 0.0001 |
| Schedule | 5-epoch warmup, cosine decay |
| Batch size and precision | 32, automatic mixed precision |
| Limit and early stopping | 100 epochs, patience 15 |
| Training K | Random K from 1 through 8 |
| Checkpoint selection | Room-averaged validation mean of the two head IoUs |
| Training seed for reported study | 0 |

The loss applies binary cross-entropy and soft Dice to each output head, restricted to U, the valid unknown region. In compact form, L = sum over heads of [masked BCE + masked Dice loss], with equal component weights. The mask is applied to the objective and its gradients; observed cells are not a reconstruction shortcut.

A selects epoch 40, P epoch 29 and D epoch 68. Early stopping ends their runs after 55, 44 and 83 epochs respectively. The selected validation mean-head scores are 0.5487, 0.5396 and 0.5572, so D is chosen by validation rather than by test performance.

Each run records configuration, source fingerprints, library versions, history and checkpoint SHA-256. The original run manifests record the Git revision as unavailable, so we rely on their saved source-file hashes rather than inventing a commit association. Resume state includes optimizer and random-state information. Deterministic seeding makes reruns inspectable; it does not create additional statistical replicates.

The reported study has one seed. Seed-to-seed uncertainty therefore remains outside its confidence intervals and is one reason the grouped continuation is necessary.
===PAGE===
# 3 Methodology continued
## 3.10 Evaluation protocol and metrics

Our primary metric is interior intersection over union on valid unknown cells. If prediction and truth masks are P and Y, IoU = |P intersection Y intersection U| / |(P union Y) intersection U|. Boundary IoU is also recorded, while hidden-wall F1 uses a fixed 0.20 m matching tolerance. Boundary distance reports symmetric nearest-boundary error in metres; lower is better. Empty-boundary cases use the evaluator's fixed convention and can strongly affect the aggregate.

We first average repeated masks within each room ID. For paired effects, we keep the matched observations and compute D minus A or D minus P per room. Bootstrap sampling operates over physical-room groups, with 2000 repeats and seed 0. It preserves dependence between related room IDs. Hundreds of masks of one room are not hundreds of independent experimental units.

| Comparison | What it can test |
| --- | --- |
| D versus A | Benefit of the complete fusion configuration |
| D versus P | Incremental waveform contribution with similar capacity and poses |
| Correct versus zero RIR | Whether waveform content is necessary at inference |
| Correct versus controlled swap | Whether the room-specific waveform assignment matters |
| K sweep | Sensitivity to measurement count |
| Noise, pose and timing stress | Local robustness under specified interventions |

The controlled swap draws audio from a different physical-room group while matching actual K. It retains the recipient's geometry and poses. Zero-RIR tests keep pose and validity information. These interventions are performed on a fixed trained model, so they test inference dependence rather than the performance of a model retrained without audio.

The written primary-plan slice uses mid-coverage viewpoint and missing-wall observations at K = 2. The completed historical training selected checkpoints at K = 8 over the broader condition mix. We report both, explicitly labeling the K = 2 slice retrospective. The test population has already been exposed during project development, so none of these numbers is presented as untouched external validation.

Thresholding, calibration and checkpoint choices must be fixed from validation or the recorded protocol, not adjusted after inspecting test failures. Brier scores supplement overlap metrics; they do not prove that confidence is safe under domain shift.
===PAGE===
# 4 Results and discussion
## 4.1 Completed EchoFusion A P D comparison

The A/P/D runs reported here have completed training, validation-based checkpoint selection and test evaluation. They use the selected fold-0 seed-0 checkpoints on the same 898 fixed test masks from 51 room IDs. None of these scores is taken from the ongoing WallFusion study. Scores are averaged within each room and then across room IDs. They are not pixel accuracy, and 0.75 IoU must not be described as 75% accuracy on an entire building.

| Model at K = 8 | Unknown interior IoU | Boundary IoU | Hidden-wall F1 at 0.20 m |
| --- | ---: | ---: | ---: |
| A geometry | 0.7282 | 0.4156 | 0.6871 |
| P geometry and positions | 0.7369 | 0.4064 | 0.6844 |
| D geometry positions and RIRs | 0.7497 | 0.4382 | 0.7077 |

D exceeds A by 0.0215 occupancy IoU, or 2.15 percentage points on the 0-100 IoU scale. The paired room-group bootstrap interval is [0.0079, 0.0361]. D minus P is 0.0128, with interval [-0.0030, 0.0295]. The first comparison supports a positive broad configuration effect in this explored test population; the second does not establish a clear incremental waveform benefit.

![Figure 4 Actual training losses and validation curves from the completed A P D runs](figures/learning_curves.png|6.55)

The training loss falls while validation improvement slows. We retain the validation-selected checkpoint instead of the final epoch. D trains longer before stopping and has more parameters than A, which is another reason to include the capacity-matched P control.

These are one-fold, one-seed results on a previously used test pool. The interval captures room-group resampling variation, not training-seed variation or generalization to measured hardware. Evidence: the original full-fold results and review summary, supplemented by the P11 diagnostic evaluator.
===PAGE===
# 4 Results and discussion continued
## 4.2 Measurement count and the primary condition

Adding more RIRs does not monotonically improve the model. The nested K sweep keeps scan identity and pair ordering fixed, so the table examines measurement-count sensitivity rather than a different random test set at every K.

| K | A occupancy IoU | P occupancy IoU | D occupancy IoU |
| --- | ---: | ---: | ---: |
| 1 | 0.7282 | 0.7307 | 0.7515 |
| 2 | 0.7282 | 0.7322 | 0.7521 |
| 4 | 0.7282 | 0.7345 | 0.7499 |
| 8 | 0.7282 | 0.7369 | 0.7497 |

A is unchanged because it receives neither poses nor RIRs. P's improvement with K shows that more endpoint information can help without waveform content. D peaks at K = 2 in this diagnostic sweep, but the difference is small and does not justify selecting K on this test pool for a new deployment claim.

The written primary condition is narrower: mid coverage, viewpoint and missing-wall masks, and K = 2. In the available retrospectively scored slice there are 193 masks from 49 room IDs, spanning 30 physical-room groups.

| Model on the primary-plan slice | Occupancy IoU | Hidden-wall F1 | Boundary distance m |
| --- | ---: | ---: | ---: |
| A | 0.7694 | 0.7086 | 0.3064 |
| P | 0.7760 | 0.6971 | 0.3123 |
| D | 0.7859 | 0.7062 | 0.3039 |

D minus A occupancy is +0.0164, with bootstrap interval [-0.0023, 0.0375]. Its hidden-wall F1 difference is -0.0024, with interval [-0.0199, 0.0140]. Both intervals include zero. The correct conclusion is that the written primary slice does not establish a reliable benefit, even though the broad aggregate looks better.

This distinction is central to the study. An aggregate over many conditions answers a different question from a predefined operating condition. We report the mixed result rather than moving the goalposts to whichever table has the largest improvement.
===PAGE===
# 4 Results and discussion continued
## 4.3 Does the fusion model actually use the audio

We intervene on D at K = 2 while keeping the geometry, reported poses, valid-pair pattern and selected checkpoint fixed. Correct RIRs provide the baseline. Zero RIRs remove waveform information. Controlled swaps replace the waveform with one from another physical-room group while matching actual K.

| Audio condition | Occupancy IoU | Hidden-wall F1 at 0.20 m |
| --- | ---: | ---: |
| Correct RIRs | 0.752108 | 0.707489 |
| Zero waveform with original poses | 0.754249 | 0.707340 |
| Different-room controlled swap | 0.752055 | 0.707616 |

Zeroing audio raises aggregate occupancy by approximately 0.00214, while swapping changes it by only -0.00005. These changes are tiny relative to the broad D-A improvement. They do not support a claim that correctly matched waveform content is essential for this checkpoint's reconstruction quality.

One interpretation is that the model relies heavily on geometry and position priors. Another is that the pooled audio representation does not retain the particular information needed to resolve unseen walls. The present tests do not identify the training mechanism, so we do not label either explanation as proven. We also cannot conclude that acoustics is inherently useless: the intervention evaluates a particular trained architecture, data distribution and preprocessing window.

Our conclusion is a configuration-level improvement, not a demonstrated waveform-specific benefit. Geometry, endpoint positions, model capacity and learned room priors remain possible explanations; the current controls do not isolate their individual contributions. More training epochs alone would not establish acoustic use: the selected model must also benefit from correctly matched waveforms under controlled evaluation.

For the next iteration, we need evidence of room-specific acoustic dependence as well as a stronger overlap score. The wall-guided temporal model is one candidate. The acceptance rule should require matched held-out performance, correct-versus-swapped sensitivity, seed consistency and pose robustness. A higher score obtained by changing the test set or weakening the controls would not resolve this result.

These findings are retrospective diagnostics on the original selected P9 checkpoints. They must not be presented as the final outcome of the unfinished five-fold WallFusion study.
===PAGE===
# 4 Results and discussion continued
## 4.4 Robustness and confidence

We apply controlled perturbations to D at K = 2 after clean preprocessing. These tests isolate particular input changes; they do not represent every error a physical microphone or scanner can produce.

| Diagnostic intervention | Occupancy IoU | Change from correct input |
| --- | ---: | ---: |
| Correct input | 0.752108 | 0 |
| 20 dB recording-style noise | 0.752067 | -0.000041 |
| Gain -6 dB | 0.752119 | +0.000011 |
| Gain +6 dB | 0.752140 | +0.000032 |
| Timing shift +3 samples | 0.752125 | +0.000017 |
| Truncate to 40 ms | 0.752002 | -0.000106 |
| Bounded 5 cm pose stress | 0.749465 | -0.002643 |

At 16 kHz, three samples correspond to 0.1875 ms. Most waveform perturbations barely change the score, while the pose intervention has a larger effect. Given the zero/swap findings, waveform insensitivity is not automatically a robustness success: it may also reflect weak waveform use. A model cannot earn credit for surviving a corrupted input it was largely ignoring.

Temperature scaling is fitted only on the 27 validation room IDs, using a discrete temperature grid and room-averaged missing-cell Brier score [9]. A temperature of 2 is selected for both heads of A, P and D. For D at K = 8, occupancy Brier improves from approximately 0.01366 to 0.01242. This evaluates probability error under the existing data distribution. It does not establish calibrated confidence for a different sensor or real room.

We also break results down by occlusion. For D at K = 2, occupancy IoU is 0.8218 on missing-wall observations, 0.7345 at doorways, 0.6770 on viewpoint observations and 0.5023 on hidden-L-wing observations. These strata contain different room/mask subsets and should not be interpreted as a controlled causal ranking of occlusion types. They do identify the hidden L wing as a practical failure region worth further study.

Boundary distance has a long-tail sensitivity to missing predicted walls and the evaluator's empty-boundary convention. We therefore co-report F1 and overlap rather than using one attractive distance number to summarize every case.
===PAGE===
# 4 Results and discussion continued
## 4.5 Qualitative results and negative cases

Aggregate scores can hide a completion that looks plausible but is spatially wrong. We inspect observation, prediction and truth on the same scanner-aligned grid. Teal denotes observed context, amber denotes inferred cells and cream denotes reference interior. In the diagnostic below, the selected examples are predetermined low-IoU cases with at least 20 hidden interior cells, with one case per room.

![Figure 5 Actual negative-case output from the fixed K 2 diagnostic evaluation](figures/negative_cases.png|6.55)

The first and third examples show how a small inferred extension can miss the true hidden area even when the visible outline appears well aligned. The middle example has more hidden interior cells and still receives an inadequate completion. A convincing rendering of observed walls does not compensate for an incorrect unknown region, because those visible walls were never the reconstruction target.

A and D share several failure patterns here. This supports examining the observation distribution and room prior, not only the acoustic encoder. These cases were retained as evidence rather than removed from the test population. Dataset attribution: AcousticRooms, Meta Platforms, CC BY 4.0; the rasterization, diagnostic layout and model overlays are project adaptations.
===PAGE===
# 4 Results and discussion continued
## 4.6 From a prediction file to an inspectable scene

A useful demonstration must let a reviewer trace a prediction back to its sample and model. The local FastAPI service loads registered checkpoints with byte hashes and rejects unknown samples, incompatible contracts and seen-room requests under its default evaluation policy. The viewer compares observation, A, D and truth on one grid. RoomScene records measured and inferred provenance separately.

![Figure 6 Actual Blender comparison render of the source OBJ and exported D prediction](figures/blender_comparison.png|6.55)

The source geometry and predicted extrusion are deliberately different representations. The left cutaway exposes the dataset mesh; the right object is a footprint completion extruded to an assumed display height. Its walls are not a recovered ceiling or a furniture model. Colours distinguish measured context from inference. The cutaway and scene-frame conversion support visual inspection of scale and orientation, not a new numerical reconstruction score.

Exports include the browser grid, GLB for native inspection, a provenance sidecar and a versioned RoomScene document. The observation payload keeps its own capture identity. This allows a reviewer to ask which cells came from a scan, which came from a model, which checkpoint produced them and what coordinate frame they use.

The API/viewer demonstration is an implementation result. It does not turn an explored test room into external validation or a first-order simulation into a measured hardware experiment. We retain those labels in the interface so the evidence remains understandable after it leaves the training notebook.
===PAGE===
# 5 Simulation layer and virtual hardware
## 5.1 Two acquisition modes

Our simulation layer supplies a repeatable acquisition workflow before physical devices are available. Audited replay loads original verified masks, RIRs, poses and identity without changing the model tensors. Interactive mode creates a known rectangle or L-shaped room with a movable horizontal scanner and up to eight speaker/microphone pairs. The first scanner pose fixes the metric frame.

![Figure 7 Actual live screenshots of virtual sensor setup and the captured partial observation](figures/simulation_plate.png|4.7)

The virtual scanner supports field of view, range, ray count, dropout and range noise. Acoustic capture uses direct and visible first-order wall, floor and ceiling paths. Impulse or sweep excitation passes through a digital recording chain with filtering, gain, noise, timing offset, clock drift, clipping and ADC quantization; sweeps are deconvolved before the existing RIR preprocessing.

Reference polygons, true poses and clean theoretical RIRs remain separate from model-facing observations. Default bundles exclude reference targets, and every virtual capture is marked training_eligible=false. Discrete acquisition events support inspection; playback speed does not change the acoustic sample rate.

This layer demonstrates software integration and selected fault responses. It does not simulate complete late reverberation, diffraction, furniture or calibrated transducers, and it does not validate physical hardware. Its model scores are labeled synthetic diagnostics rather than held-out research results.
===PAGE===
# 6 Implementation verification and reproducibility
## 6.1 Evidence beyond a screenshot

We test the properties that could invalidate the scientific comparison: coordinate conversion, shared sample identity, unknown-space semantics, mask-restricted gradients, RIR order invariance, padded-pair masking and provenance separation. End-to-end checks also exercise the inference API, checkpoint registry, verified sample store, export geometry and versioned scene schema.

| Recorded implementation check | Outcome |
| --- | --- |
| Simulation-branch CPU regression | 174 passed, 2 CUDA resume checks deselected |
| Final focused simulation and inference tests | 32 passed |
| Real A/P/D integration through the simulation API | 12 successful predictions across 4 cases |
| Exact replay versus original inference | Identical model tensors, prediction identity and metrics |
| GLB format and RoomScene schema | Passed for all 12 integration predictions |
| Observation-only downloads | No target or valid-cell label arrays |
| Browser grid and comparison checks | Passed |
| Production web build and PR CI | Passed |

The four real-checkpoint integration cases are virtual rectangle, virtual L room, sweep with device faults, and audited dataset replay. The models are real saved checkpoints; the virtual rooms are still synthetic. The test count is a software verification result, not a sample size for a reconstruction confidence interval.

## 6.2 Reproduction path

The repository contains scripts to audit assets, build caches and splits, train matched variants, evaluate fixed masks, prepare inference samples and check the service. The demonstration uses the same sample adapter as the offline pipeline. Dataset and weight payloads stay outside source control; manifests and hashes identify the required local files.

The selected original checkpoints are identified by SHA-256 prefixes A 3dae169e5fca, P 0526fb15e766 and D 7482ee9a6c9d. The frozen experiment identity begins 6d4231aa85fa. Full identifiers remain in the result manifests. The simulation implementation is commit 3e132f2 on its review branch. Its PR depends on the separate P10 completion work; a pushed feature is not described as already merged into main.

Training was not resumed to produce this report. The larger study's saved checkpoint and stop marker remain intact. A reviewer can distinguish a completed experiment, a passing integration check and unfinished research without relying on the report's wording alone.
===PAGE===
# 7 Interpretation and limitations
## 7.1 What we can defend

The completed EchoFusion A/P/D experiment shows a broad reconstruction improvement for the full fusion configuration over geometry alone. It does not establish that waveform content causes that improvement: the positions control is competitive, and zeroed or swapped RIRs cause almost no degradation. Geometry, endpoint positions, capacity and learned priors remain possible explanations. WallFusion is a separate ongoing extension; its unfinished study is not the basis of these completed A/P/D results.

The written primary slice is more cautious than the broad average: its occupancy interval crosses zero and its hidden-wall F1 does not improve over A. We therefore do not claim that the central acoustic hypothesis has been conclusively confirmed. This is a result to investigate, not a paragraph to hide behind a colourful 3D render.

## 7.2 Threats to validity

Room diversity is much smaller than the number of RIRs or augmented masks. Group-aware splitting prevents known duplicate leakage, but cannot guarantee that every unrecognized structural similarity has been removed. One training seed leaves optimization variation unmeasured. Earlier use of the test population makes the reported follow-up analyses exploratory.

Observation generation also constrains the problem. A single horizontal slice, a 12.8 m canvas and a 0.20 m cell size limit the shapes we can represent. The source/microphone eligibility rule changes which scan conditions survive. Hidden L-wing cases remain hard, and exclusions must stay visible when interpreting apparent performance.

The dataset responses are simulated. Real capture introduces synchronization errors, transducer response, uncertain poses, background noise and material effects that the dataset and first-order lab do not fully cover. Our robustness checks test selected perturbations, not this entire domain shift. Calibration fitted on the same data source does not solve that gap.

## 7.3 Consequences for use

The current prototype is suitable for controlled research demonstrations and inspecting plausible completions. It is not ready for autonomous safety decisions or claims of hardware-grade room surveying. A predicted wall should remain identifiable as a prediction in the viewer and exports.

Our strongest next step is a completed, frozen grouped study followed by a measured-room audit. Enlarging the network without changing the evidence requirements would make the artifact heavier, but not automatically make the conclusion stronger.
===PAGE===
# 8 Conclusion

We built EchoSpace AI to ask whether sound can help complete the part of a room that a scan has not observed. The project now connects audited geometry-RIR assets, observation-defined rasterization, room-grouped data handling, matched deep-learning controls, offline evaluation, local inference and visual inspection. The software sensor lab adds exact replay and interactive synthetic capture through the same model-facing contract.

The completed EchoFusion A/P/D fold-0 experiment, including checkpoint selection and evaluation, reaches 0.7497 occupancy IoU for D against 0.7282 for A at K = 8. That broad improvement is encouraging, but it is not the whole conclusion. D's advantage over the positions control is uncertain; the primary-plan slice does not establish a reliable gain; and waveform removal or swapping barely affects aggregate performance. We therefore finish with a functioning reconstruction prototype and an unresolved acoustic-use question, rather than a claim that the system can reliably hear every hidden wall.

## 8.1 Continuation to the semester project

The separate WallFusion extension remains ongoing. The next research milestone is to finish the paused grouped study, freeze its selected checkpoints and apply the same matched controls across folds and seeds. No final WallFusion result is claimed in this report. The next architecture comparison should test both shape quality and room-specific waveform dependence. Full pooled-fusion and acoustic-only controls, retrained window experiments and broader uncertainty assessment follow the completed study lock.

The physical milestone is a small measured-room capture protocol: calibrated speaker and microphone response, synchronized emission timing, known poses, paired partial scans and independent reference geometry. Each external room must pass its own identity and coordinate audit before becoming evaluation evidence. The virtual lab can rehearse this workflow and expose software mistakes, while measured data decides whether the assumptions transfer.

A later product milestone can add sensor import adapters, persistent experiment sessions and uncertainty-aware rescan suggestions. Furniture reconstruction, editing and redesign remain separate future tasks because they need different targets and evidence.

## 8.2 Final takeaway

Our project contributes a reproducible way to test an attractive idea, including the cases where the idea has not yet delivered. We can show what was observed, what was inferred, how the model was trained, where it failed and which experiment should come next. That makes the system a useful engineering and deep-learning project even before the acoustic hypothesis is fully resolved.
===PAGE===
# References

[1] I. Dokmanic, R. Parhizkar, A. Walther, Y. M. Lu and M. Vetterli, "Acoustic echoes reveal room shape," Proceedings of the National Academy of Sciences, vol. 110, no. 30, pp. 12186-12191, 2013. https://doi.org/10.1073/pnas.1221464110

[2] S. Purushwalkam, S. V. Amengual Gari, V. K. Ithapu, C. Schissler, P. Robinson, A. Gupta and K. Grauman, "Audio-Visual Floorplan Reconstruction," ICCV, pp. 1183-1192, 2021. https://arxiv.org/abs/2012.15470

[3] I. Yeon and J.-W. Choi, "RGI-Net: 3D Room Geometry Inference from Room Impulse Responses With Hidden First-Order Reflections," IWAENC, pp. 439-443, 2024. https://doi.org/10.1109/IWAENC61483.2024.10694653

[4] X. Liu, A. Kumar, P. Calamia, S. V. Amengual Gari, C. Murdock, I. Ananthabhotla, P. Robinson, E. Shlizerman, V. K. Ithapu and R. Gao, "Hearing Anywhere in Any Environment," CVPR, 2025. https://arxiv.org/abs/2504.10746

[5] O. Ronneberger, P. Fischer and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI, 2015. https://arxiv.org/abs/1505.04597

[6] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov and A. Smola, "Deep Sets," Advances in Neural Information Processing Systems, 2017. https://arxiv.org/abs/1703.06114

[7] A. Vaswani et al., "Attention Is All You Need," Advances in Neural Information Processing Systems, 2017. https://arxiv.org/abs/1706.03762

[8] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR, 2019. https://arxiv.org/abs/1711.05101

[9] C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On Calibration of Modern Neural Networks," ICML, 2017. https://arxiv.org/abs/1706.04599

[10] Meta Platforms, "AcousticRooms," official dataset repository, CC BY 4.0. https://github.com/facebookresearch/AcousticRooms (accessed 19 October 2026).

[11] EchoSpace AI project repository and recorded experiment artifacts. https://github.com/RishikSinha06/EchoSpaceAI (report snapshot 19 October 2026). Internal evidence: D0 audit; P6 dataset summary; P9 full-fold results and run manifests; P11 original-fold diagnostics; P10 exports; simulation exit-gate report.

Dataset-derived figures are adaptations of AcousticRooms under CC BY 4.0. Pipeline diagrams, model diagrams and software interface screenshots are project artifacts. Literature scores from different tasks are not reused as directly comparable project results.
