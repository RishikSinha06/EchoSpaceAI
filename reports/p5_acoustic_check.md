# P5 acoustic check

Date: 2026-10-06. Base: `600f553` (GitHub main, P4 merge).
Branch: `module-5-acoustic-pipeline`. Processor: `p5.1`.

**Exit gate: PASS for the tiny attributed integration subset.** This is not a
full-dataset training-eligibility gate.

## Reproduction

```powershell
python -m pip install -e ".[test,audit]"
python -m pytest tests -q
python scripts/check_acoustics.py
```

This machine required `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid an unrelated
globally installed pytest plugin. The complete suite passed: **87 tests in
11.19 seconds**, including 16 P5 tests. No new dependency was added.

Checks cover emission delay preservation despite a stronger later reflection,
PCM scaling, optional normalization, padding, anti-alias resampling, explicit
alignment evidence, invalid audio, cache integrity, audited identity and pose
conflicts, nested K selection, hidden/unverified pairs, invalid padding,
reproducible training augmentation and unchanged evaluation inputs. The mesh
fixture also exercises Git CRLF conversion without allowing arbitrary sizes.

## Real sample results

The existing CC BY 4.0 sample contains two rooms with four WAV/metadata pairs
each. All **8 WAVs** processed at 16 kHz / 80 ms, preserving emission timing and
amplitude. All clean-cache round trips matched. Both source and receiver were
checked against each scan's observed-free mask.

| Room | P4 scan | Usable pairs | Result |
| --- | --- | ---: | --- |
| MeetingRoom_idx_14 | viewpoint | 4 | Contract pass |
| MeetingRoom_idx_14 | missing_wall | 0 | Contract pass with invalid padding |
| MeetingRoom_idx_14 | doorway | 3 | Contract pass |
| MeetingRoom_idx_14 | l_wing | — | Not applicable to this room |
| Bathrooms_idx_21 | viewpoint | 4 | Contract pass |
| Bathrooms_idx_21 | missing_wall | 0 | Contract pass with invalid padding |
| Bathrooms_idx_21 | doorway | 1 | Contract pass |
| Bathrooms_idx_21 | l_wing | 4 | Contract pass |

Seven scan samples validated, including training-augmented copies. K=1/2/4/8
selection produced matching prefixes. Val/test waveforms remained exactly
unchanged. The two viewpoint overlays were visually inspected: source/mic
markers occupy observed free space and waveforms span the emission-origin
0–80 ms window. The gate requires at least one scan with two usable pairs per
room; both rooms passed.

The stricter byte check initially failed because Windows Git converted the
OBJ files from LF to CRLF. Removing only CRLF conversion recovered the exact
audited byte counts; signatures and absolute bounds matched. That specific
text conversion is allowed by the final implementation.

## Outputs and limitations

Local `artifacts/p5_check/` holds clean diagnostic NPZs, per-sample JSON
provenance, overlays and machine-readable `summary.json`. Clean waveform caches
are under `data/cache/p5_waveforms/`. Both directories are ignored by Git.
Samples retain `split=unassigned`; no training manifest, split, model or raw
dataset upload was created.

The subset has at most four distinct supplied pairs per room, so it cannot
prove K=8 availability. Zero/one-pair scans are valid diagnostic structures,
but cannot satisfy a primary K>=2 training condition. P6 must measure full
inventory availability, redraw/reject deficient scans and split by D0 physical
room groups. D0 timing is sampled room-level evidence; no independent direct
arrival verification for every WAV or absolute acoustic calibration is claimed.

See `docs/p5_acoustic_handoff.md` for APIs, provenance, repository review and
remaining P6 decisions.
