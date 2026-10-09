"""Module 6 on synthetic rooms: room cache, folds, fixed eval masks, dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from shapely.geometry import Polygon  # noqa: E402

from echospace.acoustics import RirConfig  # noqa: E402
from echospace.acoustics.rir import PROCESSOR_VERSION  # noqa: E402
from echospace.contract_v0 import load_npz, validate_sample  # noqa: E402
from echospace.data import (  # noqa: E402
    K_MAX,
    EchoSpaceDataset,
    EvalMaskError,
    RoomCacheError,
    RoomInfo,
    SplitError,
    collate,
    content_sha256,
    load_room_cache,
    make_folds,
    masks_for_room,
    read_eval_sample,
    save_folds,
    save_room_cache,
    spatial_transform,
    validate_folds,
)
from echospace.data.evalmasks import EVAL_FORMAT, spec_checksum, write_sample  # noqa: E402
from echospace.io.audit_geometry import FloorSlice  # noqa: E402
from echospace.scans import RoomGeometry  # noqa: E402

CONFIG = RirConfig()
L_ROOM = Polygon([(0, 0), (8, 0), (8, 3), (3, 3), (3, 7), (0, 7)])
SHOEBOX = Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])


def geometry(outline: Polygon) -> RoomGeometry:
    lines = (np.asarray(outline.exterior.coords),)
    return RoomGeometry.from_slice(FloorSlice(1.1, Polygon(outline.exterior), 1, outline.area, 1.0, 0, 0.0, 0.0, lines))


def write_room(tmp: Path, room_id: str, group: int, outline: Polygon, spacing: float = 0.8) -> Path:
    """A cached room with a grid of sources and receivers over its floor, and fake peak-1 RIRs."""
    geo = geometry(outline)
    low_x, low_z, high_x, high_z = geo.free_region.buffer(-0.3).bounds
    points = [(x, z) for x in np.arange(low_x, high_x, spacing) for z in np.arange(low_z, high_z, spacing)
              if geo.free_region.buffer(-0.3).contains(__import__("shapely").Point(x, z))]
    sources, receivers = points[::3], points
    rng = np.random.default_rng(len(room_id))
    ids, positions, waveforms = [], [], []
    for s, (sx, sz) in enumerate(sources):
        for r, (rx, rz) in enumerate(receivers):
            if (sx, sz) == (rx, rz):
                continue
            ids.append(f"{room_id}/S{s}_R{r}")
            positions.append([[sx, 1.5, sz], [rx, 1.2, rz]])
            w = rng.normal(0, 0.05, CONFIG.window_samples)
            w[int(rng.integers(20, 200))] = 1.0
            waveforms.append(w)
    meta = {"audit_record_sha256": "0" * 64, "geometry_signature": f"sig-{group}", "rir_config": CONFIG.to_dict(),
            "processor_version": PROCESSOR_VERSION, "metadata_sha256": ["1" * 64] * len(ids),
            "wav_sha256": ["2" * 64] * len(ids), "footprint_m2": geo.footprint.area, "convex": geo.is_convex}
    path = tmp / "rooms" / f"{room_id}.npz"
    save_room_cache(path, room_id, group, geo, ids, np.asarray(positions), np.asarray(waveforms), meta)
    return path


# --- room cache ------------------------------------------------------------

def test_room_cache_round_trip_and_integrity(tmp_path: Path) -> None:
    path = write_room(tmp_path, "L_0", 0, L_ROOM)
    cache = load_room_cache(path, CONFIG)
    assert cache.geometry.footprint.area == pytest.approx(L_ROOM.area)
    assert len(cache.candidates()) == len(cache.rir_ids) > 50
    loaded = cache.loader(CONFIG)(cache.candidates()[3])
    assert loaded.waveform.dtype == np.float32 and loaded.metadata["config"] == CONFIG.to_dict()
    with np.load(path) as data:
        arrays = dict(data)
    arrays["waveforms"] = arrays["waveforms"].copy()
    arrays["waveforms"][0, 0] += np.float16(0.5)
    np.savez(path, **arrays)
    with pytest.raises(RoomCacheError, match="checksum"):
        load_room_cache(path)
    with pytest.raises(RoomCacheError):
        load_room_cache(write_room(tmp_path, "L_1", 1, L_ROOM), RirConfig(sample_rate_hz=24000))


# --- folds -------------------------------------------------------------------

def rooms_for_folds(n_groups: int = 40) -> list[RoomInfo]:
    rng = np.random.default_rng(0)
    rooms = []
    for g in range(n_groups):
        area, convex = float(rng.uniform(3, 150)), bool(g % 3)
        rooms += [RoomInfo(f"room_{g}_{k}", g, area, convex) for k in range(1 + g % 2)]  # some groups are twins
    return rooms


def test_folds_keep_groups_together_and_test_each_once(tmp_path: Path) -> None:
    folds = make_folds(rooms_for_folds())
    tested = [g for f in folds["folds"] for g in f["test_groups"]]
    assert sorted(tested) == list(range(40))
    for fold in folds["folds"]:
        test, val, train = (set(fold[f"{p}_rooms"]) for p in ("test", "val", "train"))
        assert not (test & val or test & train or val & train)
        groups = {p: {folds["rooms"][r]["room_group"] for r in fold[f"{p}_rooms"]} for p in ("test", "val", "train")}
        assert not (groups["test"] & groups["train"] or groups["val"] & groups["train"])
        assert any(not folds["rooms"][r]["convex"] for r in fold["test_rooms"])  # every fold has non-convex rooms
        assert 0.10 <= len(fold["val_groups"]) / (len(fold["val_groups"]) + len(fold["train_groups"])) <= 0.25
    assert make_folds(rooms_for_folds())["checksum"] == folds["checksum"]  # deterministic
    save_folds(folds, tmp_path / "folds.json")
    edited = json.loads((tmp_path / "folds.json").read_text())
    edited["folds"][0]["train_groups"].append(edited["folds"][0]["test_groups"][0])
    with pytest.raises(SplitError):
        validate_folds(edited)


# --- fixed evaluation masks ----------------------------------------------------

@pytest.fixture(scope="module")
def eval_setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict, dict]:
    tmp = tmp_path_factory.mktemp("p6")
    rooms = {"L_a": (0, L_ROOM), "L_b": (1, L_ROOM), "box_a": (2, SHOEBOX), "box_b": (3, SHOEBOX)}
    entries, feasibility, records = [], {}, []
    for room_id, (group, outline) in rooms.items():
        cache = load_room_cache(write_room(tmp, room_id, group, outline), CONFIG)
        samples, feasibility[room_id] = masks_for_room(cache, CONFIG, per_bin=1, max_seeds=4)
        for sample in samples:
            entries.append(write_sample(sample, tmp / "eval"))
            records.append(sample.record)
    with (tmp / "eval" / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    spec = {"format": EVAL_FORMAT, "samples": entries, "feasibility": feasibility}
    spec["checksum"] = spec_checksum(spec)
    infos = [RoomInfo(r, g, o.area, o.equals(SHOEBOX)) for r, (g, o) in rooms.items()]
    folds = make_folds(infos, n_folds=2, inner_val_fraction=0.5)
    return tmp, spec, folds


def test_eval_masks_are_contract_samples_with_eight_nested_pairs(eval_setup: tuple[Path, dict, dict]) -> None:
    tmp, spec, _ = eval_setup
    assert spec["samples"], "no evaluation masks generated"
    kinds = {e["occlusion_type"] for e in spec["samples"]}
    assert {"viewpoint", "doorway"} <= kinds and "l_wing" in kinds
    assert all(not spec["feasibility"][r]["l_wing"]["high"]["applicable"] for r in ("box_a", "box_b"))
    manifest = {json.loads(line)["sample_id"]: json.loads(line) for line in (tmp / "eval" / "manifest.jsonl").open()}
    for entry in spec["samples"]:
        record = manifest[entry["sample_id"]]
        arrays = load_npz(tmp / "eval" / record["sample_path"])
        validate_sample(record, arrays)
        assert arrays["rir_valid"].sum() == K_MAX and record["rir_sample_rate_hz"] == 16000
        assert content_sha256(record, arrays) == entry["sha256"]


def test_eval_masks_regenerate_identically_and_tampering_is_refused(eval_setup: tuple[Path, dict, dict]) -> None:
    tmp, spec, _ = eval_setup
    cache = load_room_cache(tmp / "rooms" / "L_a.npz", CONFIG)
    again, _ = masks_for_room(cache, CONFIG, per_bin=1, max_seeds=4)
    first = {e["sample_id"]: e["sha256"] for e in spec["samples"] if e["room_id"] == "L_a"}
    assert {s.record["sample_id"]: content_sha256(s.record, s.arrays) for s in again} == first
    entry = next(e for e in spec["samples"] if e["room_id"] == "L_a")
    record = json.loads(next(line for line in (tmp / "eval" / "manifest.jsonl").open() if entry["sample_id"] in line))
    path = tmp / "eval" / record["sample_path"]
    arrays = load_npz(path)
    arrays["observed_free"] = arrays["observed_free"].copy()
    arrays["observed_free"][0, 0] = 1 - arrays["observed_free"][0, 0]
    original = path.read_bytes()
    np.savez_compressed(path, **arrays)
    try:
        with pytest.raises(EvalMaskError):
            read_eval_sample(tmp / "eval", record, entry["sha256"])
    finally:
        path.write_bytes(original)


# --- dataset -------------------------------------------------------------------

def _cell_free(item: dict, which: str) -> bool:
    size = 0.20
    for row, ok in zip(item[which], item["rir_valid"]):
        if ok:
            col, r = int(np.floor(row[0] / size)), int(np.floor(row[1] / size))
            if not item["observed_free"][r, col]:
                return False
    return True


def test_train_items_are_valid_deterministic_and_vary_by_epoch(eval_setup: tuple[Path, dict, dict]) -> None:
    tmp, spec, folds = eval_setup
    data = EchoSpaceDataset(tmp / "rooms", folds, 0, "train", eval_spec=spec, samples_per_room=3, seed=5)
    assert len(data) == 3 * len(data.rooms) and data.rooms
    item = data[0]
    assert item["observed_free"].shape == (64, 64) and item["rir"].shape == (K_MAX, CONFIG.window_samples)
    assert 1 <= item["k"] <= K_MAX and item["rir_valid"].sum() == item["k"]
    assert not np.any(item["rir"][item["k"]:]) and not np.any(item["src_pos"][item["k"]:])
    for i in range(len(data)):
        it = data[i]
        assert _cell_free(it, "src_pos") and _cell_free(it, "mic_pos"), "sensor outside observed free space"
        assert not (it["observed_free"].astype(bool) & ~it["target_occupancy"].astype(bool)).any()
    assert np.array_equal(data[1]["rir"], data[1]["rir"])
    first = data[1]["observed_cells"].copy()
    data.set_epoch(1)
    assert not np.array_equal(first, data[1]["observed_cells"])
    batch = collate([data[0], data[1]])
    assert batch["rir"].shape == (2, K_MAX, CONFIG.window_samples) and len(batch["room_id"]) == 2


def test_eval_items_use_fixed_masks_and_nested_k(eval_setup: tuple[Path, dict, dict]) -> None:
    tmp, spec, folds = eval_setup
    rooms_with_masks = {e["room_id"] for e in spec["samples"]}
    split = next(s for s in ("test", "val") if set(folds["folds"][0][f"{s}_rooms"]) & rooms_with_masks)
    full = EchoSpaceDataset(tmp / "rooms", folds, 0, split, eval_dir=tmp / "eval", eval_spec=spec, k_eval=8)
    four = EchoSpaceDataset(tmp / "rooms", folds, 0, split, eval_dir=tmp / "eval", eval_spec=spec, k_eval=4)
    assert len(full) == len(four) > 0
    a, b = full[0], four[0]
    assert a["sample_id"] == b["sample_id"] and a["k"] == 8 and b["k"] == 4
    assert np.array_equal(a["rir"][:4], b["rir"][:4]) and not np.any(b["rir"][4:])
    assert np.array_equal(a["observed_cells"], b["observed_cells"])
    assert _cell_free(a, "src_pos") and _cell_free(a, "mic_pos")
    assert a["spatial_transform"] == {"quarter_turns": 0, "flip_columns": False}
    full.set_epoch(7)
    assert np.array_equal(full[0]["observed_cells"], a["observed_cells"])  # fixed across epochs


def test_spatial_transform_moves_positions_with_the_grids() -> None:
    rng = np.random.default_rng(0)
    for turns in range(4):
        for flip in (False, True):
            grid = np.zeros((64, 64), dtype=np.uint8)
            uvh = np.column_stack((rng.uniform(0, 12.8, 30), rng.uniform(0, 12.8, 30), np.full(30, 1.2)))
            for u, v, _ in uvh:
                grid[int(v / 0.2), int(u / 0.2)] = 1
            arrays = {name: grid for name in ("observed_cells", "observed_free", "observed_wall", "valid_cells",
                                              "target_occupancy", "target_boundary")}
            moved, new = spatial_transform(arrays, uvh, turns, flip, 12.8)
            for u, v, h in new:
                assert moved["observed_free"][int(np.floor(v / 0.2)), int(np.floor(u / 0.2))] == 1
                assert h == pytest.approx(1.2)


def test_p7_fixed_freeze_uses_fold_rooms_and_verifies_p6_content(eval_setup):
    pytest.importorskip("torch")
    from echospace.training.frozen import FrozenDataset, freeze_fixed_p6
    tmp, spec, folds = eval_setup
    # Four synthetic groups are too small for stratified rounding to guarantee
    # an inner validation group. Make one explicit without changing test folds.
    from echospace.data.splits import folds_checksum
    folds = json.loads(json.dumps(folds))
    f = folds["folds"][0]
    validation_group = f["train_groups"].pop()
    f["val_groups"] = [validation_group]
    f["val_rooms"] = [r for r, info in folds["rooms"].items() if info["room_group"] == validation_group]
    f["train_rooms"] = [r for r in f["train_rooms"] if r not in f["val_rooms"]]
    folds["checksum"] = folds_checksum(folds)
    spec_path = tmp / "p7_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    folds_path = tmp / "p7_folds.json"
    save_folds(folds, folds_path)
    frozen = freeze_fixed_p6(tmp / "p7_fixed", tmp / "eval", folds_path, spec_path, fold=0, k=2)
    counts = 0
    for split in ("train", "val", "test"):
        dataset = FrozenDataset(frozen, split)
        counts += len(dataset)
        assert {e["metadata"]["room_id"] for e in dataset.entries} <= set(folds["folds"][0][f"{split}_rooms"])
        sample = dataset[0]
        assert sample["rir_valid"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
        assert not sample["src_pos"][2:].any() and not sample["mic_pos"][2:].any()
    assert counts == len(spec["samples"])
    assert dataset.manifest["source"]["kind"] == "P6_fixed_masks"
    assert dataset.manifest["source"]["training_augmentation"] == "none"
    manifest = {r["sample_id"]: r for r in map(json.loads, (tmp / "eval/manifest.jsonl").read_text().splitlines())}
    entry = spec["samples"][0]
    path = tmp / "eval" / manifest[entry["sample_id"]]["sample_path"]
    arrays = load_npz(path)
    arrays["rir_waveforms"][0, 0] += 0.1
    np.savez_compressed(path, **arrays)
    with pytest.raises(EvalMaskError, match="content differs"):
        freeze_fixed_p6(tmp / "p7_edited", tmp / "eval", folds_path, spec_path)
