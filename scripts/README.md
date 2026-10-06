Dataset inspection, preprocessing, training, evaluation, and export CLIs go here.

## `audit_data.py`: AcousticRooms D0 audit (module 2)

Checks whether AcousticRooms geometry and acoustics are paired correctly and
writes the per-room manifest, the report, figures and an optional attributed
sample. It reads the archives in place and never extracts or modifies them.

```powershell
python -m pip install -e ".[test,audit]"
python scripts/audit_data.py --limit 3          # trial: 3 rooms, outputs to data/cache/audit_trial/
python scripts/audit_data.py --rooms Apartments_idx_0 Bathrooms_idx_1  # trial on named rooms
python scripts/audit_data.py --write-sample     # full audit (about 35 min on one CPU)
```

The dataset root is resolved in this order:

1. `--root <folder>`
2. the `ECHOSPACE_ACOUSTICROOMS_ROOT` environment variable
3. `data/raw` under the repository root

The root is the folder that holds the Drive-export zips (`AcousticRooms-*.zip`,
`single_channel_ir-*.zip`, ...). The archive index is cached in
`data/cache/acousticrooms_index.json`; pass `--rebuild-index` after the
archives change. Other options: `--pairs-per-room` (default 64, `0` = all),
`--seed` (default 0), `--figures` (default 12).

A full run writes, relative to the repository root:

| Output | Committed |
| --- | --- |
| `data/manifests/acousticrooms_index.jsonl`, one JSON record per room | yes |
| `reports/d0_audit.md` and `reports/d0_audit_summary.json` | yes |
| `reports/figures/d0_<room>.png` | yes |
| `data/samples/acousticrooms/` (with `--write-sample`), 2 rooms under CC BY 4.0 | yes |

Trial runs (`--limit` or `--rooms`) write to `data/cache/audit_trial/` so they
cannot overwrite the committed outputs. The acceptance thresholds at the top of
the script were fixed before the full run; change one only with a recorded reason.

## `audit_duplicates.py`: which accepted rooms are the same physical room

Run after `audit_data.py`. It slices every accepted room at 1.1 m and groups
rooms that share a geometry signature or a floor footprint (IoU >= 0.999 after
translation, 90-degree turns and mirrors), then writes
`reports/d0_duplicates.json` with the groups, a `room_group` id per room and
the independent-room counts. Room-level splits must keep each group together.

```powershell
python scripts/audit_duplicates.py          # about 1 min; same --root rules as audit_data.py
```

## `audit_reflections.py`: first-reflection check (plan, Stage 0 gate)

Run after `audit_data.py`. For each accepted room it predicts the first
reflection off the floor and off the longest straight wall (image source) and
tests, on the manifest's seeded line-of-sight pairs, whether the RIR peaks
there more than at control times. Writes `reports/d0_reflections.json`.

```powershell
python scripts/audit_reflections.py         # about 45 min; same --root rules as audit_data.py
```

## `simulate_scans.py`: P4 scan-simulator gate

Run after the D0 audit. For every accepted room it builds free-running partial
scans of each occlusion type (viewpoint, missing wall, doorway, L-wing) and one
scan aimed at each coverage bin, validates every sample against contract
v0.1.0, and draws overlays for 20 rooms. See `docs/p4_scan_handoff.md`.

```powershell
python scripts/simulate_scans.py --limit 6     # trial, into data/cache/scan_trial/
python scripts/simulate_scans.py               # all accepted rooms, about 1 h
```

Writes `reports/p4_scan_check.md`, `reports/p4_scans_summary.json` and
`reports/figures/p4/`. Same `--root` rules as `audit_data.py`.

## `check_acoustics.py`: P5 acoustic integration gate

Uses the attributed two-room sample already in Git and the D0 manifest. Checks
all eight WAVs, clean-cache round trips, nested K bundles on P4 observed-free
masks, contract validity, training augmentation and unchanged validation/test
audio. No download or training split is created.

```powershell
python -m pip install -e ".[test,audit]"
python scripts/check_acoustics.py
```

Configuration: `configs/p5_acoustics.json`. Outputs: ignored
`artifacts/p5_check/` and `data/cache/p5_waveforms/`. Use `--out-dir`,
`--cache-dir`, `--config`, `--samples-root` and `--audit-manifest` to override
paths. See `docs/p5_acoustic_handoff.md` for eligibility and gate limits.
