Python research code will live here. Suggested modules: `io/adapters`, `geometry`, `scans`, `acoustics`, `data`, `models`, `training`, `evaluation`, and `inference`. Add modules when real dataset formats and contracts are verified.

`contract_v0.py` now validates the first versioned manifest, sample, and
prediction handoff. It accepts per-sample grids and transforms while the
AcousticRooms audit establishes the real-data convention.
