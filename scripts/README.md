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
