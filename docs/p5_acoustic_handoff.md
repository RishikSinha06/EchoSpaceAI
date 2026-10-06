# P5 acoustic preprocessing handoff

P5 converts D0-audited AcousticRooms WAVs into deterministic acoustic inputs
for P4 scans. Implementation starts from GitHub `main` at `600f553` (P4 merge)
on `module-5-acoustic-pipeline`. P6 owns dataset assembly, splits and the final
usable-pair requirement. No training dataset or model was produced here.

## Defaults and timing

`configs/p5_acoustics.json` records processor `p5.1`: mono float32, 16 kHz,
80 ms / 1,280 samples. Rational polyphase resampling applies an anti-alias
filter before cropping; shorter signals are right-padded with zeros.

Emission time is preserved, as requested. Source–receiver travel delay remains
in the window. The largest peak is never used to align a waveform: it may be a
reflection. D0 found an emission-origin convention using sampled timing checks
per room, rather than independently verifying the direct onset of every WAV.
The candidate pipeline requires accepted D0 emission evidence and rejects
alternative alignment modes. The lower-level `preprocess_wav` separately
supports explicit, independently evidenced direct-onset alignment or declared
relative-onset alignment; neither is enabled in the D0 integration path.

Signed PCM is scaled by its representable magnitude; unsigned 8-bit PCM is
centered at 128. Floating WAV amplitudes are preserved. There is no default
peak normalization, level equalization, inversion of `IR_norm`, or clipping.
`IR_norm` is retained as uninterpreted provenance because its meaning remains
unverified. Optional peak normalization is explicit in `RirConfig` and changes
the cache key. These initial settings still need the P0/P6 study freeze.

## Identity and eligibility

`load_audited_candidates(reader, room, audit)` requires the matching accepted
D0 room record, metres, Z-up source coordinates, no reported pose conflicts,
and the expected timing verdict. It checks mesh geometry signature, absolute
bounds and audited OBJ byte count (allowing Git's CRLF checkout conversion).
The geometry signature alone cannot establish absolute pose identity because
it is translation invariant. Metadata joins use audited room-folder identity
and numeric source/receiver IDs. Conflicting poses for one ID are refused.

Positions transform once from source `(x,y,z)` to scene `(x,z,-y)` in metres.
P4's observation-derived grid is used to project both source and receiver into
the binary `observed_free` mask. A hidden pose or unverified candidate cannot
become a valid acoustic row. Selection never reads target occupancy or RIR
amplitudes. The loader checks the audited 22,050 Hz release rate and rejects
silent, nonfinite, stereo or malformed input and silent selected windows.

## APIs and contract

```python
from echospace.acoustics import (
    RirCache, RirConfig, load_audited_candidates, process_candidate, build_bundle,
)

config = RirConfig()  # emission timing, preserved amplitude
candidates = load_audited_candidates(reader, room_entry, audit_record)
cache = RirCache(cache_directory)
bundle = build_bundle(
    candidates, scan.grid, scan.observed.observed_free,
    lambda candidate: process_candidate(reader, candidate, config, cache),
    room_id=scan.room_id, selection_key=sample_id, seed=0, k=2, config=config,
)
arrays = scan.contract_arrays()
arrays.update(bundle.contract_arrays())
```

The contract remains `0.1.0`: waveforms `(K,1280)`, poses `(K,2,3)` with source
then receiver, and `rir_valid (K,)`. Set the record's `rir_sample_rate_hz` to
16000 and `rir_ids` to `bundle.rir_ids`; validate the completed sample using
`contract_v0.validate_sample`. The rate, length and timing are explicit, rather
than reinterpreting the original 22,050 Hz data.

Ranking hashes `(seed, room_id, selection_key, rir_id)` independently of K and
candidate order. K=1/2/4/8 use the same ranked prefixes. By default, insufficient
usable pairs raise `AcousticError`. Diagnostic `pad_missing=True` supplies
unique placeholder IDs, zero waveforms/poses and `rir_valid=0`. Padding is not
proof of training eligibility; a K=8 primary experiment must actually have
eight usable pairs or follow an explicitly documented study condition.

## Cache and provenance

`RirCache.get` keys source WAV SHA-256, preprocessing configuration, timing
evidence and processor version. Each atomic NPZ contains clean float32 audio
and JSON provenance with no pickle. Cache hits validate shape, finiteness,
configuration and waveform checksum; corrupted entries fail closed.

Pair provenance also records metadata and audit-record SHA-256, source/mic
positions, native rate/dtype/length, upstream timing and normalization status,
crop start, valid/padded counts and any normalization divisor. No augmented
audio is cached. Changing processing settings or evidence produces a new key.

## Training augmentation

`augment_bundle` requires an assigned `train`, `val` or `test` split. It returns
a copy and leaves clean input unchanged. Validation/test copies are unchanged.
Training applies reproducible per-pair gain (−3 to +3 dB), causal second-order
low-pass filtering (3–7 kHz), and white noise (25–40 dB SNR), followed by small
horizontal pose jitter (0.02 m standard deviation). These are configurable
initial choices, not a claim that every perturbation models real equipment.
The causal filter introduces phase delay; it does not realign emission time.

Pose proposals must keep both positions in observed free space; after eight
failed attempts the original poses remain. Heights and invalid padded rows
remain unchanged. Seed, sample key and pair ID make repeated calls and K
prefixes reproducible. P6 must assign room-group splits before augmentation.

## Integration and P6 handoff

Run `python scripts/check_acoustics.py` after installing existing `[test,audit]`
extras. It uses the already attributed CC BY 4.0 subset (two rooms/eight WAVs),
not external downloads. Output NPZs, provenance JSON, plots and `summary.json`
go to ignored `artifacts/p5_check/`; clean caches go to `data/cache/p5_waveforms/`.
Outputs are diagnostic with `split=unassigned`, not a training manifest.

P6 must measure usable K on the full accepted inventory, redraw/reject scans
with too few eligible pairs, keep `reports/d0_duplicates.json` room groups
together, assign stable splits, and write validated JSONL/NPZ samples with
P4 and P5 configuration versions. It must not pool unverified dataset routes.
See `reports/p5_acoustic_check.md` for this gate's evidence and limits.

## Repository review before P5

- P1 contract is implemented; P2 accepted 257 rooms with 302,671 paired RIRs.
  Accepted inventory is not equivalent to a finished training dataset.
- P3 labels and P4 partial scans are implemented. P4's existing report records
  249 usable rooms and 3,993 validated scans; P5 depends on observed free masks.
- P3's standalone slice rejects extra components, whereas D0 permits a largest
  component covering at least 98%. Boundary labels use an interior distance
  threshold. P6 should review these target conventions before freezing data.
- Models, training and the inference API remain scaffolds. The browser is an
  inspection tool; its assumed extrusion height is not reconstructed geometry.
  Its display wall classification should be checked during P10 integration.
- Raw 32 kHz release files are LFS pointers in the audited checkout, and
  `IR_norm` remains undocumented. P5 makes no raw-waveform or absolute-level
  verification claim. All timing evidence is explicitly recorded.
