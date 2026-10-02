"""D0 pairing audit for AcousticRooms: is the geometry aligned with the acoustics?

Reads the archives in place (never extracts or modifies them) and writes a
per-room manifest, a markdown report, a few figures and, on request, a tiny
attributed sample. No model, training or preprocessing code lives here.

    python scripts/audit_data.py --limit 3     # quick look at three rooms
    python scripts/audit_data.py --write-sample # full audit

The dataset root comes from ``--root``, then ``ECHOSPACE_ACOUSTICROOMS_ROOT``,
then ``data/raw``. With ``--limit`` the outputs go to a scratch folder so a
trial run cannot overwrite the committed manifest and report.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from echospace.io import audit_geometry as ag  # noqa: E402
from echospace.io.adapters import acousticrooms as ar  # noqa: E402
from echospace.io.audit import (  # noqa: E402
    CANVAS_12_8_M,
    CANVAS_25_6_M,
    SPEED_OF_SOUND_M_S,
    anchored_coverage_counts,
    direct_path_samples,
    first_strong_peak,
    fits_bbox,
    summarize_timing,
)

# Acceptance thresholds. Fixed before the full run; change them only with a
# recorded reason, never to move a room across the line after seeing results.
MIN_INSIDE_FRACTION = 0.95
MIN_CLEAR_PAIRS_FOR_TIMING = 8
POSITION_CONFLICT_TOLERANCE_M = 1e-6
SLICE_HEIGHT_M = 1.1
CANVASES = {"12.8": CANVAS_12_8_M, "25.6": CANVAS_25_6_M}
PAIR_OFFSET_COLUMNS = ["source_id", "receiver_id", "direct_samples", "peak_sample", "offset_samples", "line_of_sight_blocked"]
MAX_SAMPLE_MESH_BYTES = 2_000_000
MAX_FIGURE_BYTES = 300_000


def natural_key(room_id: str) -> tuple[str, int]:
    match = re.match(r"(.*?)(\d+)$", room_id)
    return (match.group(1), int(match.group(2))) if match else (room_id, -1)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


def select_rooms(index: ar.AcousticRoomsIndex, limit: int | None, names: list[str] | None) -> list[str]:
    """All rooms in natural order, or a deterministic round-robin across categories."""
    if names:
        missing = [name for name in names if name not in index.rooms]
        if missing:
            raise SystemExit(f"unknown rooms: {missing}")
        return names
    ordered = sorted(index.rooms, key=natural_key)
    if limit is None:
        return ordered
    by_category: dict[str, list[str]] = defaultdict(list)
    for room_id in ordered:
        by_category[index.rooms[room_id].category].append(room_id)
    picked: list[str] = []
    depth = 0
    while len(picked) < min(limit, len(ordered)):
        for category in sorted(by_category):
            if depth < len(by_category[category]) and len(picked) < limit:
                picked.append(by_category[category][depth])
        depth += 1
    return picked


def positions_by_id(reader: ar.ArchiveReader, room: ar.RoomEntry) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]:
    """Unique source and receiver positions, and how many records disagree with their id."""
    sources: dict[int, np.ndarray] = {}
    receivers: dict[int, np.ndarray] = {}
    conflicts = 0
    for source_id, receiver_id in room.paired():
        record = ar.load_pair_metadata(reader, room.metadata[(source_id, receiver_id)])
        for table, key, xyz in ((sources, source_id, record.source_xyz), (receivers, receiver_id, record.receiver_xyz)):
            position = np.asarray(xyz, dtype=np.float64)
            if key not in table:
                table[key] = position
            elif float(np.abs(table[key] - position).max()) > POSITION_CONFLICT_TOLERANCE_M:
                conflicts += 1
    return sources, receivers, conflicts


def simulation_check(reader: ar.ArchiveReader, room: ar.RoomEntry, receivers: dict[int, np.ndarray]) -> dict[str, Any] | None:
    """Cross-check metadata against the simulator's own config, where one is shipped."""
    if room.simulation is None:
        return None
    config = ar.load_simulation(reader, room.simulation)
    listed = np.array([[item["x"], item["y"], item["z"]] for item in config.get("receivers", [])], dtype=np.float64)
    found = None
    if len(listed) and receivers:
        ours = np.stack(list(receivers.values()))
        nearest = np.linalg.norm(ours[:, None, :] - listed[None, :, :], axis=2).min(axis=1)
        found = float((nearest <= 0.011).mean())  # metadata is rounded to 1 cm
    settings = config.get("simulationSettings") or {}
    return {
        "name_in_file": config.get("name"),
        "name_matches_folder": config.get("name") == room.room_id,
        "n_receivers": len(config.get("receivers", [])),
        "n_sources": len(config.get("sources", [])),
        "metadata_receivers_found_fraction": found,
        "speed_of_sound_m_s": settings.get("speedOfSound"),
        "impulse_length_s": config.get("impulseLengthSec"),
    }


def audit_room(
    reader: ar.ArchiveReader,
    room: ar.RoomEntry,
    up: int,
    pairs_per_room: int,
    seed: int,
    keep_for_figure: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run checks A-G for one room. Returns the manifest record and optional figure data."""
    pairs = room.paired()
    record: dict[str, Any] = {
        "dataset": ar.DATASET_NAME,
        "room_id": room.room_id,
        "category": room.category,
        "mesh_path": room.mesh.display if room.mesh else None,
        "n_rirs": len(room.rirs),
        "n_metadata": len(room.metadata),
        "n_pairs": len(pairs),
        "n_rirs_without_metadata": len(room.rirs.keys() - room.metadata.keys()),
        "n_metadata_without_rirs": len(room.metadata.keys() - room.rirs.keys()),
        "up_axis": up,
        "units": "metres",
        "speed_of_sound_m_s": SPEED_OF_SOUND_M_S,
    }
    reasons: list[str] = []
    if room.mesh is None:
        reasons.append("no_mesh")
    if not pairs:
        reasons.append("no_paired_rirs")
    if reasons:
        record.update(accepted=False, reject_reasons=reasons)
        return record, None

    sources, receivers, conflicts = positions_by_id(reader, room)
    source_xyz = np.stack([sources[key] for key in sorted(sources)])
    receiver_xyz = np.stack([receivers[key] for key in sorted(receivers)])
    per_source = Counter(source_id for source_id, _ in pairs)
    per_receiver = Counter(receiver_id for _, receiver_id in pairs)
    record.update(
        n_sources=len(sources),
        n_receivers=len(receivers),
        position_conflicts=conflicts,
        full_source_receiver_grid=len(pairs) == len(sources) * len(receivers),
        pairs_per_source_min=min(per_source.values()),
        pairs_per_receiver_min=min(per_receiver.values()),
        pairs_per_receiver_median=float(np.median(list(per_receiver.values()))),
        simulation_info=simulation_check(reader, room, receivers),
    )
    if conflicts:
        reasons.append("position_conflicts")

    mesh_bytes = reader.read(room.mesh)
    mesh = ar.load_mesh(reader, room.mesh)
    summary = ag.mesh_summary(mesh, up)
    slices = {height: ag.floor_slice(mesh, up, height) for height in ag.SLICE_HEIGHTS_M}
    cut = slices[SLICE_HEIGHT_M]
    axes = list(ag.horizontal_axes(up))
    if cut.footprint is not None:
        footprint_xy, footprint_source = ag.footprint_vertices(cut.footprint), f"slice_{SLICE_HEIGHT_M}m"
    else:
        low, high = np.asarray(mesh.bounds)[:, axes]
        footprint_xy = np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]])
        footprint_source = "mesh_bounds"
    height = summary["ceiling"] - summary["floor"]
    record.update(
        mesh_bytes=len(mesh_bytes),
        geometry_signature=ag.geometry_signature(mesh.vertices),
        mesh=summary,
        extent_m={"horizontal_0": summary["extent"][axes[0]], "horizontal_1": summary["extent"][axes[1]], "height": height},
        slice={f"{h:.1f}": s.to_dict() for h, s in slices.items()},
        single_closed_interior_all_heights=all(s.single_closed_interior for s in slices.values()),
        footprint_source=footprint_source,
        footprint_area_m2=cut.area_m2 if cut.footprint is not None else None,
        footprint_extent_m=[float(v) for v in footprint_xy.max(axis=0) - footprint_xy.min(axis=0)],
        # The meshes are not watertight, so no exact volume exists. This prism
        # is an estimate that ignores sloped ceilings and furniture.
        volume_m3_prism_estimate=cut.area_m2 * height if cut.footprint is not None else None,
        furniture={
            "interior_loops_at_slice": cut.n_interior_loops,
            "interior_loop_area_m2": cut.interior_loop_area_m2,
            "semantic_labels_in_obj": False,
        },
    )
    if not cut.single_closed_interior:
        reasons.append("slice_not_single_closed_interior")

    all_points = np.vstack([source_xyz, receiver_xyz])
    contained = ag.containment(mesh, cut.footprint, all_points, up)
    record["containment"] = contained
    if cut.footprint is not None and contained["inside_fraction"] < MIN_INSIDE_FRACTION:
        reasons.append("positions_outside_mesh")

    # D. timing on a seeded sample of pairs
    rng = np.random.default_rng([seed, zlib.crc32(room.room_id.encode())])
    if pairs_per_room and len(pairs) > pairs_per_room:
        chosen = sorted(pairs[i] for i in rng.choice(len(pairs), size=pairs_per_room, replace=False))
    else:
        chosen = pairs
    sample_src = np.stack([sources[s] for s, _ in chosen])
    sample_rec = np.stack([receivers[r] for _, r in chosen])
    blocked = ag.line_of_sight_blocked(mesh, sample_src, sample_rec)
    direct, peaks, rates, lengths, maxima, dtypes = [], [], set(), [], [], set()
    figure_rir = None
    for i, pair in enumerate(chosen):
        rir = ar.load_rir(reader, room.rirs[pair])
        rates.add(rir.sample_rate_hz)
        dtypes.add(rir.native_dtype)
        lengths.append(len(rir.waveform))
        maxima.append(float(np.abs(rir.waveform).max()))
        direct.append(direct_path_samples(sample_src[i], sample_rec[i], rir.sample_rate_hz))
        peaks.append(first_strong_peak(rir.waveform))
        if keep_for_figure and (figure_rir is None or (figure_rir["blocked"] and not blocked[i])):
            figure_rir = {"pair": pair, "waveform": rir.waveform, "fs": rir.sample_rate_hz, "direct": direct[-1], "peak": peaks[-1], "blocked": bool(blocked[i])}
    direct_a, peaks_a = np.asarray(direct), np.asarray(peaks, dtype=np.float64)
    timing_all = summarize_timing(direct_a, peaks_a)
    timing_clear = summarize_timing(direct_a[~blocked], peaks_a[~blocked])
    record.update(
        rir_sample_rates_hz=sorted(rates),
        rir_native_dtypes=sorted(dtypes),
        rir_length_samples={"min": int(min(lengths)), "median": float(np.median(lengths)), "max": int(max(lengths))},
        rir_peak_abs={"min": min(maxima), "median": float(np.median(maxima)), "max": max(maxima)},
        timing={
            "n_pairs_sampled": len(chosen),
            "n_line_of_sight_blocked": int(blocked.sum()),
            "all_pairs": timing_all.to_dict(),
            "line_of_sight_pairs": timing_clear.to_dict(),
            "verdict": timing_clear.verdict if timing_clear.n_pairs >= MIN_CLEAR_PAIRS_FOR_TIMING else "insufficient",
        },
        pair_offset_columns=PAIR_OFFSET_COLUMNS,
        pair_offsets=[
            [s, r, round(float(d), 2), int(p), round(float(p - d), 2), int(b)]
            for (s, r), d, p, b in zip(chosen, direct_a, peaks_a, blocked)
        ],
    )
    verdict = record["timing"]["verdict"]
    if verdict != "constant_offset":
        reasons.append(f"timing_{verdict}")
    if len(rates) != 1:
        reasons.append("mixed_sample_rates")

    # F. eligibility, both rules, both canvases; receivers stand in for scanner poses
    poses_xy = receiver_xyz[:, axes]
    eligibility: dict[str, Any] = {"n_candidate_poses": int(len(poses_xy))}
    for label, canvas in CANVASES.items():
        covering = anchored_coverage_counts(footprint_xy, poses_xy, canvas)
        eligibility[label] = {
            "bbox": fits_bbox(footprint_xy, canvas),
            "anchor_poses_covering": int(covering),
            "anchor": covering > 0,
        }
    record["eligibility"] = eligibility

    # G. how the receivers are spread
    spans = poses_xy.max(axis=0) - poses_xy.min(axis=0)
    footprint_spans = footprint_xy.max(axis=0) - footprint_xy.min(axis=0)
    if len(poses_xy) > 1:
        gaps = np.linalg.norm(poses_xy[:, None, :] - poses_xy[None, :, :], axis=2)
        np.fill_diagonal(gaps, np.inf)
        nearest = float(np.median(gaps.min(axis=1)))
    else:
        nearest = None
    record["receiver_distribution"] = {
        "span_fraction_of_footprint": [float(v) for v in spans / footprint_spans],
        "median_nearest_neighbour_m": nearest,
        "height_min_m": float(receiver_xyz[:, up].min() - summary["floor"]),
        "height_max_m": float(receiver_xyz[:, up].max() - summary["floor"]),
        "distinct_horizontal_positions": int(len(np.unique(np.round(poses_xy, 2), axis=0))),
    }
    record.update(accepted=not reasons, reject_reasons=reasons)

    figure = None
    if keep_for_figure:
        figure = {
            "room_id": room.room_id,
            "outlines": cut.outlines,
            "sources": source_xyz[:, axes],
            "receivers": poses_xy,
            "bounds": np.asarray(mesh.bounds)[:, axes],
            "rir": figure_rir,
            "axes": axes,
        }
    return record, figure


def write_figure(figure: dict[str, Any], path: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, wave) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=70)
    for line in figure["outlines"]:
        top.plot(line[:, 0], line[:, 1], color="0.35", linewidth=0.8)
    top.scatter(*figure["receivers"].T, s=10, c="tab:blue", label=f"receivers ({len(figure['receivers'])})")
    top.scatter(*figure["sources"].T, s=22, c="tab:red", marker="^", label=f"sources ({len(figure['sources'])})")
    top.set_aspect("equal")
    names = "xyz"
    top.set_xlabel(f"source {names[figure['axes'][0]]} (m)")
    top.set_ylabel(f"source {names[figure['axes'][1]]} (m)")
    top.set_title(f"{figure['room_id']}: slice at {SLICE_HEIGHT_M} m", fontsize=9)
    top.legend(fontsize=7, loc="upper right")
    rir = figure["rir"]
    if rir is not None:
        stop = int(min(len(rir["waveform"]), max(rir["direct"], rir["peak"]) + 0.012 * rir["fs"]))
        time_ms = np.arange(stop) / rir["fs"] * 1000.0
        wave.plot(time_ms, rir["waveform"][:stop], color="0.2", linewidth=0.7)
        wave.axvline(rir["direct"] / rir["fs"] * 1000.0, color="tab:red", linestyle="--", linewidth=1.0, label="predicted direct path")
        wave.plot(rir["peak"] / rir["fs"] * 1000.0, rir["waveform"][rir["peak"]], "o", color="tab:green", markersize=5, fillstyle="none", label="first strong peak")
        blocked = " (line of sight blocked)" if rir["blocked"] else ""
        wave.set_title(f"S{rir['pair'][0]} R{rir['pair'][1]}: offset {rir['peak'] - rir['direct']:+.2f} samples{blocked}", fontsize=9)
        wave.set_xlabel("time (ms)")
        wave.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path.stat().st_size


def versions() -> dict[str, str]:
    import matplotlib
    import rtree
    import scipy
    import shapely
    import trimesh

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
        "shapely": shapely.__version__,
        "rtree": rtree.__version__,
        "matplotlib": matplotlib.__version__,
        "code_revision": commit,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Dataset-level counts. Independent rooms are counted per distinct geometry."""
    audited = [r for r in records if "geometry_signature" in r]
    accepted = [r for r in audited if r["accepted"]]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in audited:
        groups[record["geometry_signature"]].append(record)
    for record in audited:
        record["geometry_group_size"] = len(groups[record["geometry_signature"]])
        record["geometry_group_rooms"] = sorted((r["room_id"] for r in groups[record["geometry_signature"]]), key=natural_key)

    def independent(rows: list[dict[str, Any]]) -> int:
        return len({row["geometry_signature"] for row in rows})

    eligibility: dict[str, Any] = {}
    for label in CANVASES:
        for rule in ("bbox", "anchor"):
            rows = [r for r in accepted if r["eligibility"][label][rule]]
            eligibility[f"{rule}_{label}"] = {"rooms": len(rows), "independent_rooms": independent(rows)}
        both_fail_anchor = [r for r in accepted if r["eligibility"][label]["bbox"] and not r["eligibility"][label]["anchor"]]
        eligibility[f"bbox_pass_anchor_fail_{label}"] = len(both_fail_anchor)

    offsets = np.array([row[4] for r in audited for row in r["pair_offsets"]], dtype=np.float64)
    blocked = np.array([row[5] for r in audited for row in r["pair_offsets"]], dtype=bool)
    direct = np.array([row[2] for r in audited for row in r["pair_offsets"]], dtype=np.float64)
    timing = {
        "pairs_sampled": int(len(offsets)),
        "line_of_sight_blocked": int(blocked.sum()),
        "all_pairs": summarize_timing(direct, direct + offsets).to_dict() if len(offsets) else None,
        "line_of_sight_pairs": summarize_timing(direct[~blocked], (direct + offsets)[~blocked]).to_dict() if (~blocked).any() else None,
        "room_verdicts": dict(Counter(r["timing"]["verdict"] for r in audited)),
    }
    reasons = Counter(reason for r in records for reason in r["reject_reasons"])

    def quantiles(key: str) -> dict[str, float]:
        values = np.array([r[key] for r in audited], dtype=np.float64)
        return {name: float(np.quantile(values, q)) for name, q in (("min", 0), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("max", 1))} if len(values) else {}

    return {
        "rooms_examined": len(records),
        "rooms_with_mesh_and_pairs": len(audited),
        "rooms_accepted": len(accepted),
        "rooms_rejected": len(records) - len(accepted),
        "independent_geometries_audited": independent(audited),
        "independent_geometries_accepted": independent(accepted),
        "geometry_groups_larger_than_one": sorted(
            (sorted((r["room_id"] for r in rows), key=natural_key) for rows in groups.values() if len(rows) > 1),
            key=lambda names: natural_key(names[0]),
        ),
        "reject_reasons": dict(reasons),
        "eligibility": eligibility,
        "timing": timing,
        "pairs": {key: quantiles(key) for key in ("n_pairs", "n_sources", "n_receivers", "pairs_per_receiver_min")},
        "rooms_with_at_least_receivers": {str(k): sum(r["n_receivers"] >= k for r in accepted) for k in (4, 8, 16, 25)},
        "rooms_with_min_pairs_per_source_at_least": {str(k): sum(r["pairs_per_source_min"] >= k for r in accepted) for k in (4, 8)},
        "full_source_receiver_grid_rooms": sum(r["full_source_receiver_grid"] for r in audited),
        "sample_rates_hz": sorted({rate for r in audited for rate in r["rir_sample_rates_hz"]}),
        "native_dtypes": sorted({d for r in audited for d in r["rir_native_dtypes"]}),
        "watertight_rooms": sum(r["mesh"]["watertight"] for r in audited),
        "floor_at_zero_rooms": sum(r["mesh"]["floor_at_zero"] for r in audited),
        "single_closed_interior_rooms": sum(r["slice"][f"{SLICE_HEIGHT_M:.1f}"]["single_closed_interior"] for r in audited),
        "single_closed_interior_all_heights_rooms": sum(r["single_closed_interior_all_heights"] for r in audited),
        "rooms_with_interior_loops": sum(r["furniture"]["interior_loops_at_slice"] > 0 for r in audited),
        "clearance_met_fraction_median": float(np.median([r["containment"]["meets_documented_clearance_fraction"] for r in audited])) if audited else None,
        "surface_distance_min_m": float(min(r["containment"]["surface_distance_min_m"] for r in audited)) if audited else None,
        "inside_fraction_min": float(min(r["containment"]["inside_fraction"] or 0 for r in audited)) if audited else None,
        "simulation_info": {
            "rooms_with_file": sum(r.get("simulation_info") is not None for r in audited),
            "name_mismatches": sum(1 for r in audited if r.get("simulation_info") and not r["simulation_info"]["name_matches_folder"]),
            "receiver_positions_confirmed": sum(1 for r in audited if r.get("simulation_info") and (r["simulation_info"]["metadata_receivers_found_fraction"] or 0) >= 0.99),
        },
    }


def render_report(context: dict[str, Any]) -> str:
    s, identity, up, v = context["summary"], context["identity"], context["up_vote"], context["versions"]
    e, t = s["eligibility"], s["timing"]
    clear, everything = t["line_of_sight_pairs"] or {}, t["all_pairs"] or {}
    scope = "full dataset" if context["limit"] is None and not context["rooms_arg"] else f"TRIAL RUN on {s['rooms_examined']} rooms, not the dataset result"

    def number(value: Any, digits: int = 2) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# D0 audit: AcousticRooms geometry-acoustics pairing",
        "",
        f"Scope: **{scope}**. Generated by `scripts/audit_data.py` at code revision `{v['code_revision']}`; seed {context['seed']}, "
        f"{context['pairs_per_room'] or 'all'} RIR pairs sampled per room for timing.",
        "",
        "## Answers",
        "",
        f"1. **Independent eligible rooms, anchor rule:** {e['anchor_12.8']['independent_rooms']} at 12.8 m, "
        f"{e['anchor_25.6']['independent_rooms']} at 25.6 m "
        f"(of {s['independent_geometries_accepted']} accepted independent geometries; {s['rooms_accepted']} accepted room ids).",
        f"2. **Timing offset:** `{clear.get('verdict', 'n/a')}` on line-of-sight pairs: median {number(clear.get('offset_median'))} samples, "
        f"std {number(clear.get('offset_std'))}, {number(100 * clear.get('inlier_fraction', float('nan')), 1)} % within 2 samples, "
        f"{clear.get('n_early', 'n/a')} arrivals earlier than physically possible.",
        "",
        "## A. Identity",
        "",
        f"- Rooms indexed: {identity['rooms_total']}; with mesh {identity['rooms_with_mesh']}; with RIRs {identity['rooms_with_rirs']}; with metadata {identity['rooms_with_metadata']}.",
        f"- RIRs {identity['rirs_total']}, metadata records {identity['metadata_total']}, matched pairs {identity['pairs_matched']}. Duplicated ids: {len(identity['duplicates'])}.",
        f"- RIRs without metadata: {sum(len(p) for p in identity['rirs_without_metadata'].values())}. "
        f"Metadata without RIR: {sum(len(p) for p in identity['metadata_without_rirs'].values())} in "
        f"{len(identity['metadata_without_rirs'])} rooms ({', '.join(f'{k}: {len(p)}' for k, p in identity['metadata_without_rirs'].items()) or 'none'}).",
        f"- Rooms with RIRs but no mesh: {', '.join(identity['rooms_with_rirs_but_no_mesh']) or 'none'}. Meshes without RIRs: {', '.join(identity['meshes_without_rirs']) or 'none'}.",
        "- Ids are joined as integers: metadata names pad them as `S009_R0023`, RIR names as `S009_R023`.",
        f"- Same id, different position within a room: {sum(r.get('position_conflicts', 0) for r in context['records'])} records.",
        f"- `simulation_info` is shipped for {s['simulation_info']['rooms_with_file']} audited rooms. Its `name` field disagrees with its folder in "
        f"{s['simulation_info']['name_mismatches']} of them, yet its receiver list reproduces the metadata positions in "
        f"{s['simulation_info']['receiver_positions_confirmed']}. The folder is the reliable key; the `name` field is not.",
        f"- Git LFS pointers instead of data: {len(identity['lfs_pointers'])} files, all under `raw_dense_simulation/`. "
        "**The untrimmed 32 kHz cross-check could not be run.**",
        "",
        "## B. Units and axes",
        "",
        f"- Up axis: **{'xyz'[up['up_axis']]}** ({'unanimous' if up['unanimous'] else 'NOT unanimous'} over {up['rooms']} meshes). "
        f"Rooms whose minimum is 0 on x / y / z: {' / '.join(str(n) for n in up['rooms_with_min_at_zero_per_axis'])}.",
        f"- Extent per axis, min / median / max (m): "
        + "; ".join(f"{'xyz'[i]} {up['extent_min_per_axis'][i]:.2f} / {up['extent_median_per_axis'][i]:.2f} / {up['extent_max_per_axis'][i]:.2f}" for i in range(3))
        + ".",
        f"- Floor at 0 on the up axis in {s['floor_at_zero_rooms']} of {s['rooms_with_mesh_and_pairs']} audited rooms.",
        f"- Units are metres: with c = {SPEED_OF_SOUND_M_S:.0f} m/s the fitted slope of peak against predicted delay is {number(clear.get('slope'), 4)} "
        "(1.0 means metres; 0.01 would mean centimetres).",
        f"- RIR sample rate(s): {s['sample_rates_hz']} Hz; stored as {s['native_dtypes']}.",
    ]
    if up["up_axis"] == 2:
        lines += [
            "- Proposed `source_to_scene` (source x, y, z-up to scene X right, Y up, Z forward; rotation only, determinant +1, `source_units_to_meters` = 1):",
            "  `[[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]`. The horizontal heading is a free choice; nothing in the data fixes it.",
        ]
    lines += [
        "",
        "## C. Containment",
        "",
        f"- Lowest per-room share of source/receiver positions inside the room (slice outline and floor-to-ceiling): {number(s['inside_fraction_min'], 3)}.",
        f"- Closest any position comes to a mesh surface: {number(s['surface_distance_min_m'])} m. "
        f"Median share of a room's positions meeting the documented 0.5 m clearance against the OBJ mesh: {number(s['clearance_met_fraction_median'])}.",
        "",
        "## D. Timing convention",
        "",
        "`offset = first strong peak - ||src - mic|| / 343 * fs`, in samples at the native rate. Line of sight is decided by "
        "ray casting the mesh between the two metadata positions; it never looks at the audio.",
        "",
        "| Pairs | n | offset median | offset mean | offset std | within 2 samples | earlier than possible | slope | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, row in (("line of sight", clear), ("all sampled", everything)):
        if row:
            lines.append(
                f"| {label} | {row['n_pairs']} | {number(row['offset_median'])} | {number(row['offset_mean'])} | {number(row['offset_std'])} | "
                f"{number(100 * row['inlier_fraction'], 1)} % | {row['n_early']} | {number(row['slope'], 4)} | `{row['verdict']}` |"
            )
    lines += [
        "",
        f"- Sampled pairs with the direct path blocked by the mesh: {t['line_of_sight_blocked']} of {t['pairs_sampled']}.",
        f"- Per-room verdicts: {t['room_verdicts']}.",
        "- Per-pair offsets are in the manifest (`pair_offsets`). No per-pair correction is applied anywhere.",
        "",
        "## E. Geometry quality",
        "",
        f"- Watertight meshes: {s['watertight_rooms']} of {s['rooms_with_mesh_and_pairs']}. Volumes are therefore prism estimates (slice area x height), not exact.",
        f"- One closed connected interior at {SLICE_HEIGHT_M} m: {s['single_closed_interior_rooms']} rooms; at all of 1.0 / 1.1 / 1.2 m: {s['single_closed_interior_all_heights_rooms']}.",
        f"- Rooms with closed loops inside the outline at {SLICE_HEIGHT_M} m (furniture, columns, partitions): {s['rooms_with_interior_loops']}.",
        "- The OBJ files contain vertices and faces only: no groups, objects or materials. **Semantic labels cannot be used to drop furniture from a wall-only slice.** "
        "Layer names exist only in the Rhino `.3dm` files under `simulation_info/`.",
        f"- Distinct geometries among audited rooms: {s['independent_geometries_audited']}. Room ids sharing one geometry: "
        f"{len(s['geometry_groups_larger_than_one'])} groups"
        + (": " + "; ".join(", ".join(group) for group in s["geometry_groups_larger_than_one"][:40]) if s["geometry_groups_larger_than_one"] else "")
        + ".",
        "",
        "## F. Eligibility",
        "",
        "Accepted rooms only. The footprint is the outline of the slice; candidate scanner poses are the room's receiver positions; "
        "the canvas is centred on the pose and aligned with the mesh axes. A room passes the anchor rule if at least one pose covers the whole footprint.",
        "",
        "| Canvas | bbox rule: rooms (independent) | anchor rule: rooms (independent) | pass bbox, fail anchor |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in CANVASES:
        lines.append(
            f"| {label} m | {e[f'bbox_{label}']['rooms']} ({e[f'bbox_{label}']['independent_rooms']}) | "
            f"{e[f'anchor_{label}']['rooms']} ({e[f'anchor_{label}']['independent_rooms']}) | {e[f'bbox_pass_anchor_fail_{label}']} |"
        )
    lines += [
        "",
        "## G. Pair availability",
        "",
        "| Per room | min | p25 | median | p75 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("n_pairs", "paired RIRs"), ("n_sources", "sources"), ("n_receivers", "receivers"), ("pairs_per_receiver_min", "fewest pairs at any receiver")):
        q = s["pairs"][key]
        if q:
            lines.append(f"| {label} | {q['min']:.0f} | {q['p25']:.0f} | {q['median']:.0f} | {q['p75']:.0f} | {q['max']:.0f} |")
    lines += [
        "",
        f"- Rooms where every source is paired with every receiver: {s['full_source_receiver_grid_rooms']} of {s['rooms_with_mesh_and_pairs']}.",
        f"- Accepted rooms with at least 4 / 8 / 16 / 25 receivers: {' / '.join(str(s['rooms_with_at_least_receivers'][k]) for k in ('4', '8', '16', '25'))}.",
        f"- Accepted rooms where every source has at least 4 / 8 receivers: {' / '.join(str(s['rooms_with_min_pairs_per_source_at_least'][k]) for k in ('4', '8'))}.",
        "- Whether K receivers also lie inside the *observed* region depends on the scan masks of module 4; this audit only bounds it from above.",
        "",
        "## Accepted and rejected",
        "",
        f"- Examined {s['rooms_examined']}; accepted {s['rooms_accepted']}; rejected {s['rooms_rejected']}.",
        f"- Reject reasons (a room can have several): {s['reject_reasons'] or 'none'}.",
        "",
        "| Rejected room | reasons |",
        "| --- | --- |",
    ]
    rejected = [r for r in context["records"] if not r["accepted"]]
    lines += [f"| {r['room_id']} | {', '.join(r['reject_reasons'])} |" for r in rejected] or ["| none | |"]
    lines += [
        "",
        "## Method and thresholds",
        "",
        f"- Accept a room when: it has a mesh and paired RIRs; ids never map to two positions; the {SLICE_HEIGHT_M} m slice is one closed interior "
        f"(largest part >= 98 % of sliced area); >= {MIN_INSIDE_FRACTION:.0%} of positions are inside; and line-of-sight timing is `constant_offset` "
        f"(>= 95 % of pairs within 2 samples of the median, from >= {MIN_CLEAR_PAIRS_FOR_TIMING} line-of-sight pairs). Thresholds were fixed before the full run.",
        "- Eligibility is reported separately from acceptance and does not use ground-truth room bounds to place the canvas.",
        "- First strong peak: first sample reaching 50 % of the RIR's maximum magnitude, followed uphill to the crest of that peak.",
        f"- Versions: {', '.join(f'{k} {val}' for k, val in v.items() if k != 'code_revision')}.",
        "- Data: AcousticRooms, CC BY 4.0 (see `DATA_LICENSE.md`). Figures in `reports/figures/` are derived from it.",
        "",
    ]
    return "\n".join(lines)


def write_sample(reader: ar.ArchiveReader, index: ar.AcousticRoomsIndex, records: list[dict[str, Any]], out: Path, up: int) -> list[str]:
    """Copy 2 small accepted rooms with 4 line-of-sight pairs each, byte for byte."""
    candidates = sorted(
        (r for r in records if r["accepted"] and r["eligibility"]["12.8"]["anchor"] and r["mesh_bytes"] <= MAX_SAMPLE_MESH_BYTES),
        key=lambda r: (r["mesh_bytes"], natural_key(r["room_id"])),
    )
    chosen: list[dict[str, Any]] = []
    for record in candidates:
        if record["category"] not in {c["category"] for c in chosen}:
            chosen.append(record)
        if len(chosen) == 2:
            break
    listing = []
    for record in chosen:
        room = index.rooms[record["room_id"]]
        folder = out / room.category / room.room_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{room.room_id}.obj").write_bytes(reader.read(room.mesh))
        pairs = [(row[0], row[1]) for row in record["pair_offsets"] if not row[5]][:4]
        for pair in pairs:
            for ref in (room.rirs[pair], room.metadata[pair]):
                (folder / ref.chain[-1].rsplit("/", 1)[-1]).write_bytes(reader.read(ref))
        listing.append({"room_id": room.room_id, "category": room.category, "pairs": pairs, "rir_sample_rate_hz": record["rir_sample_rates_hz"][0], "up_axis": "xyz"[up], "units": "metres"})
    (out / "sample_index.json").write_text(json.dumps(jsonable(listing), indent=2) + "\n", encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(
        "# AcousticRooms sample\n\n"
        "These files are an unmodified excerpt of the AcousticRooms dataset\n"
        "(<https://github.com/facebookresearch/AcousticRooms>), (c) Meta Platforms, Inc.,\n"
        "licensed under CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>).\n"
        "Only the folder layout was changed: each room's mesh, RIRs and per-pair\n"
        "metadata sit together. `sample_index.json` was added by this project.\n\n"
        "Rooms: " + ", ".join(item["room_id"] for item in listing) + ".\n\n"
        "Cite: Liu et al., \"Hearing Anywhere in Any Environment\", CVPR 2025. See `DATA_LICENSE.md`.\n",
        encoding="utf-8",
    )
    return [item["room_id"] for item in listing]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help=f"dataset root (default: ${ar.ROOT_ENV_VAR} or data/raw)")
    parser.add_argument("--limit", type=int, help="audit only N rooms, spread across categories, into a scratch folder")
    parser.add_argument("--rooms", nargs="+", help="audit exactly these room ids")
    parser.add_argument("--pairs-per-room", type=int, default=64, help="RIR pairs sampled per room for timing; 0 = all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figures", type=int, default=12, help="number of rooms to plot")
    parser.add_argument("--out", help="output folder (default: repo root, or data/cache/audit_trial for trial runs)")
    parser.add_argument("--index-cache", default="data/cache/acousticrooms_index.json")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--write-sample", action="store_true", help="also write data/samples/acousticrooms")
    args = parser.parse_args()

    started = time.time()
    root = ar.resolve_root(args.root)
    trial = args.limit is not None or bool(args.rooms)
    out = Path(args.out) if args.out else (REPO / "data" / "cache" / "audit_trial" if trial else REPO)
    cache = REPO / args.index_cache
    index = None if args.rebuild_index else ar.load_cached_index(root, cache)
    if index is None:
        print("building index from archive directories ...", flush=True)
        index = ar.build_index(root)
        ar.save_index(index, cache)
    identity = ar.identity_report(index)
    print(f"index: {identity['rooms_total']} rooms, {identity['rirs_total']} RIRs, {identity['metadata_total']} metadata, "
          f"{identity['pairs_matched']} matched pairs, {identity['rooms_with_mesh']} meshes", flush=True)

    selected = select_rooms(index, args.limit, args.rooms)
    figure_rooms = set(select_rooms(index, min(args.figures, len(index.rooms)), None)) if not trial else set(selected)
    records: list[dict[str, Any]] = []
    figure_sizes: dict[str, int] = {}
    with ar.ArchiveReader(root) as reader:
        # B. the up axis is a dataset convention; vote over every mesh, not just the selected rooms
        bounds = [ag.obj_vertex_bounds(reader.read(room.mesh)) for _, room in sorted(index.rooms.items()) if room.mesh is not None]
        up_vote = ag.vote_up_axis(bounds)
        up = up_vote["up_axis"]
        print(f"up axis: {'xyz'[up]} (min at zero per axis {up_vote['rooms_with_min_at_zero_per_axis']} of {up_vote['rooms']} meshes; "
              f"unanimous={up_vote['unanimous']})", flush=True)
        for position, room_id in enumerate(selected, 1):
            tick = time.time()
            record, figure = audit_room(reader, index.rooms[room_id], up, args.pairs_per_room, args.seed, room_id in figure_rooms)
            records.append(record)
            if figure is not None:
                figure_sizes[room_id] = write_figure(figure, out / "reports" / "figures" / f"d0_{room_id}.png")
            if "timing" in record:
                clear = record["timing"]["line_of_sight_pairs"]
                elig = record["eligibility"]
                print(
                    f"[{position}/{len(selected)}] {room_id}: pairs {record['n_pairs']} (S{record['n_sources']} x R{record['n_receivers']}), "
                    f"extent {record['extent_m']['horizontal_0']:.1f} x {record['extent_m']['horizontal_1']:.1f} x {record['extent_m']['height']:.1f} m, "
                    f"slice parts {record['slice'][f'{SLICE_HEIGHT_M:.1f}']['n_parts']}, inside {record['containment']['inside_fraction']}, "
                    f"timing {record['timing']['verdict']} (LOS n={clear['n_pairs']} median {clear['offset_median']:+.2f} std {clear['offset_std']:.2f}; "
                    f"blocked {record['timing']['n_line_of_sight_blocked']}), "
                    f"anchor poses 12.8: {elig['12.8']['anchor_poses_covering']}/{elig['n_candidate_poses']}, 25.6: {elig['25.6']['anchor_poses_covering']}/{elig['n_candidate_poses']}, "
                    f"{'ACCEPT' if record['accepted'] else 'REJECT ' + ','.join(record['reject_reasons'])} [{time.time() - tick:.1f}s]",
                    flush=True,
                )
                if trial:
                    for row in record["pair_offsets"][:3]:
                        print(f"      S{row[0]} R{row[1]}: direct {row[2]:.2f} samples, peak {row[3]}, offset {row[4]:+.2f}, blocked={bool(row[5])}")
            else:
                print(f"[{position}/{len(selected)}] {room_id}: REJECT {','.join(record['reject_reasons'])}", flush=True)

        summary = summarize(records)
        sample_rooms = write_sample(reader, index, records, out / "data" / "samples" / "acousticrooms", up) if args.write_sample else []

    manifest = out / "data" / "manifests" / "acousticrooms_index.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(jsonable(record), separators=(",", ":"), allow_nan=False) + "\n")
    context = {
        "summary": summary, "identity": identity, "up_vote": up_vote, "versions": versions(), "records": records,
        "limit": args.limit, "rooms_arg": args.rooms, "seed": args.seed, "pairs_per_room": args.pairs_per_room,
    }
    report = out / "reports" / "d0_audit.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(context), encoding="utf-8", newline="\n")
    (out / "reports" / "d0_audit_summary.json").write_text(
        json.dumps(jsonable({"summary": summary, "identity": {k: val for k, val in identity.items() if k != "lfs_pointers"}, "up_axis": up_vote, "versions": context["versions"]}), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    oversized = {name: size for name, size in figure_sizes.items() if size > MAX_FIGURE_BYTES}
    e = summary["eligibility"]
    clear = summary["timing"]["line_of_sight_pairs"] or {}
    print("\n=== summary ===")
    print(f"examined {summary['rooms_examined']}, accepted {summary['rooms_accepted']}, rejected {summary['rooms_rejected']}; reasons {summary['reject_reasons']}")
    print(f"independent geometries (accepted): {summary['independent_geometries_accepted']}")
    for label in CANVASES:
        print(f"canvas {label} m: bbox {e[f'bbox_{label}']['rooms']} rooms ({e[f'bbox_{label}']['independent_rooms']} independent), "
              f"anchor {e[f'anchor_{label}']['rooms']} rooms ({e[f'anchor_{label}']['independent_rooms']} independent)")
    if clear:
        print(f"timing (line-of-sight pairs, n={clear['n_pairs']}): {clear['verdict']}, median offset {clear['offset_median']:+.2f}, "
              f"std {clear['offset_std']:.2f}, slope {clear['slope']:.4f}, early arrivals {clear['n_early']}")
    print(f"figures: {len(figure_sizes)} written" + (f"; OVER {MAX_FIGURE_BYTES} bytes: {oversized}" if oversized else ""))
    if sample_rooms:
        print(f"sample rooms: {sample_rooms}")
    print(f"outputs under {out.relative_to(REPO) if out.is_relative_to(REPO) else out} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
