Put local source archives in `raw/`, prepared samples in `processed/`, and temporary caches in `cache/`. These directories are ignored by Git. The two supplied planning documents do not contain the actual dataset files.

## AcousticRooms (module 2 D0 audit)

- `raw/` holds the AcousticRooms Drive-export zips exactly as downloaded. Treat
  them as read-only: the audit reads members in place and never extracts them.
  The dataset can live elsewhere: pass `--root` to `scripts/audit_data.py` or set
  `ECHOSPACE_ACOUSTICROOMS_ROOT`; the default is `data/raw`.
- `cache/acousticrooms_index.json` is the archive index the audit builds on first
  run (ignored by Git; rebuild with `--rebuild-index`).
- `manifests/acousticrooms_index.jsonl` is the committed D0 manifest, one line per
  room, with acceptance, reject reasons, geometry signature, timing and eligibility.
  Paths in it are relative to the archive members, never absolute.
- `samples/acousticrooms/` is a tiny unmodified excerpt (2 rooms, a few RIR pairs)
  for integration tests and demos. It is CC BY 4.0; keep its `ATTRIBUTION.md` and
  see `DATA_LICENSE.md` in the repository root.

Never commit `raw/`, `cache/`, `*.zip`, `*.npz`, checkpoints or `runs/`. The only
`.wav` and `.obj` files allowed in Git are under `samples/`.
