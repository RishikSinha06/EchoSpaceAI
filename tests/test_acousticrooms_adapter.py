"""AcousticRooms index and loaders against a synthetic dataset built in a temp folder.

The fake dataset mimics the real archive nesting (a deflated wrapper zip that
holds meshes and a nested ``metadata.zip``; a separate zip of stored
per-category RIR zips) but contains only a generated shoebox and impulses.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
wavfile = pytest.importorskip("scipy.io.wavfile")

from echospace.io.adapters import acousticrooms as ar  # noqa: E402
from echospace.io.audit import direct_path_samples, first_strong_peak  # noqa: E402

FS = 22050
ROOM_SIZE = (4.0, 5.0, 2.5)  # Z up, floor at z = 0, like the source dataset
SOURCES = {1: (1.0, 1.0, 1.5), 2: (3.0, 4.0, 1.2)}
RECEIVERS = {3: (2.0, 2.5, 1.4), 12: (3.2, 1.1, 1.0)}
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 5033164800\n"


def _zip_bytes(members: dict[str, bytes], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _shoebox_obj() -> bytes:
    box = trimesh.creation.box(extents=ROOM_SIZE)
    box.apply_translation(np.asarray(ROOM_SIZE) / 2.0)
    return box.export(file_type="obj").encode()


def _wav_bytes(source: tuple[float, ...], receiver: tuple[float, ...]) -> bytes:
    samples = np.zeros(2000, dtype=np.int16)
    samples[int(round(direct_path_samples(source, receiver, FS)))] = 20000
    buffer = io.BytesIO()
    wavfile.write(buffer, FS, samples)
    return buffer.getvalue()


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    metadata: dict[str, bytes] = {}
    rirs: dict[str, bytes] = {}
    for source_id, source in SOURCES.items():
        for receiver_id, receiver in RECEIVERS.items():
            record = json.dumps({"src_loc": list(source), "rec_loc": list(receiver)}).encode()
            # ids are padded differently on the two sides, as in the real dataset
            metadata[f"metadata/Office/Office_idx_0/S00{source_id}_R00{receiver_id}.json"] = record
            rirs[f"Office/Office_idx_0/S{source_id:03d}_R{receiver_id:03d}_hybrid_IR.wav"] = _wav_bytes(source, receiver)
    # orphans on both sides, and a room that has RIRs but no mesh
    metadata["metadata/Office/Office_idx_0/S009_R009.json"] = b'{"src_loc": [1, 1, 1], "rec_loc": [2, 2, 1]}'
    rirs["Office/Office_idx_0/S008_R008_hybrid_IR.wav"] = _wav_bytes((1, 1, 1), (2, 2, 1))
    metadata["metadata/Cafe/Cafe_idx_7/S001_R001.json"] = b'{"src_loc": [1, 1, 1], "rec_loc": [2, 2, 1]}'
    cafe = {"Cafe/Cafe_idx_7/S001_R001_hybrid_IR.wav": _wav_bytes((1, 1, 1), (2, 2, 1))}

    wrapper = {
        "AcousticRooms/README.md": b"synthetic",
        "AcousticRooms/room_mesh_obj_format/Office/Office_idx_0.obj": _shoebox_obj(),
        "AcousticRooms/metadata.zip": _zip_bytes(metadata),
        "AcousticRooms/simulation_info/Office/Office_idx_0/simulation.json": b'{"name": "Office_idx_0"}',
        "AcousticRooms/raw_dense_simulation/dense.zip.part-001": LFS_POINTER,
        "AcousticRooms/raw_dense_simulation/room_mesh_obj_format/Office/Office_idx_0.obj": b"not the same room",
    }
    (tmp_path / "AcousticRooms-export-001.zip").write_bytes(_zip_bytes(wrapper))
    outer = {
        "single_channel_ir_1/Office.zip": _zip_bytes(rirs),
        "single_channel_ir_1/Cafe.zip": _zip_bytes(cafe),
    }
    (tmp_path / "single_channel_ir-001.zip").write_bytes(_zip_bytes(outer, zipfile.ZIP_STORED))
    (tmp_path / "RAF-unrelated.zip").write_bytes(_zip_bytes({"Office/Office_idx_0/S001_R001_hybrid_IR.wav": b"x"}))
    return tmp_path


def test_classify_roles_and_integer_ids() -> None:
    assert ar.classify("metadata/Office/Office_idx_0/S009_R0023.json") == ("metadata", "Office", "Office_idx_0", (9, 23))
    assert ar.classify("Office/Office_idx_0/S009_R023_hybrid_IR.wav") == ("rir", "Office", "Office_idx_0", (9, 23))
    assert ar.classify("AcousticRooms/room_mesh_obj_format/Cafe/Cafe_idx_1.obj") == ("mesh", "Cafe", "Cafe_idx_1", None)
    assert ar.classify("AcousticRooms/raw_dense_simulation/room_mesh_obj_format/Cafe/Cafe_idx_1.obj") is None
    assert ar.classify("AcousticRooms/README.md") is None


def test_index_joins_by_room_and_pair_and_reports_orphans(dataset_root: Path) -> None:
    index = ar.build_index(dataset_root)
    office = index.rooms["Office_idx_0"]
    assert office.category == "Office"
    assert office.mesh is not None and office.simulation is not None
    assert office.paired() == [(1, 3), (1, 12), (2, 3), (2, 12)]

    report = ar.identity_report(index)
    assert report["rooms_total"] == 2
    assert report["pairs_matched"] == 5
    assert report["rirs_without_metadata"] == {"Office_idx_0": [(8, 8)]}
    assert report["metadata_without_rirs"] == {"Office_idx_0": [(9, 9)]}
    assert report["rooms_with_rirs_but_no_mesh"] == ["Cafe_idx_7"]
    assert report["duplicates"] == []
    assert len(report["lfs_pointers"]) == 1 and "part-001" in report["lfs_pointers"][0]


def test_loaders_read_nested_members_without_extraction(dataset_root: Path) -> None:
    before = sorted(path.name for path in dataset_root.iterdir())
    index = ar.build_index(dataset_root)
    office = index.rooms["Office_idx_0"]
    with ar.ArchiveReader(dataset_root) as reader:
        mesh = ar.load_mesh(reader, office.mesh)
        assert np.allclose(mesh.bounds, [[0, 0, 0], ROOM_SIZE])
        assert ar.load_simulation(reader, office.simulation) == {"name": "Office_idx_0"}
        for pair in office.paired():
            metadata = ar.load_pair_metadata(reader, office.metadata[pair])
            assert metadata.source_xyz == SOURCES[pair[0]]
            assert metadata.receiver_xyz == RECEIVERS[pair[1]]
            rir = ar.load_rir(reader, office.rirs[pair])
            assert rir.sample_rate_hz == FS and rir.native_dtype == "int16"
            assert float(np.abs(rir.waveform).max()) <= 1.0
            predicted = direct_path_samples(metadata.source_xyz, metadata.receiver_xyz, rir.sample_rate_hz)
            assert abs(first_strong_peak(rir.waveform) - predicted) <= 0.5
    assert sorted(path.name for path in dataset_root.iterdir()) == before  # nothing written beside the archives


def test_index_cache_round_trip_and_invalidation(dataset_root: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    cache = tmp_path_factory.mktemp("cache") / "index.json"
    index = ar.build_index(dataset_root)
    assert ar.load_cached_index(dataset_root, cache) is None
    ar.save_index(index, cache)
    cached = ar.load_cached_index(dataset_root, cache)
    assert cached is not None and cached.to_json() == index.to_json()
    (dataset_root / "metadata-extra.zip").write_bytes(_zip_bytes({"metadata/Cafe/Cafe_idx_7/S002_R002.json": b"{}"}))
    assert ar.load_cached_index(dataset_root, cache) is None


def test_duplicate_members_are_reported_not_overwritten(dataset_root: Path) -> None:
    duplicate = {"metadata/Office/Office_idx_0/S01_R03.json": b'{"src_loc": [0, 0, 0], "rec_loc": [9, 9, 9]}'}
    (dataset_root / "metadata-extra.zip").write_bytes(_zip_bytes(duplicate))
    index = ar.build_index(dataset_root)
    assert any("Office_idx_0 metadata S1_R3" in entry for entry in index.duplicates)


def test_official_checkout_layout_is_indexed(tmp_path: Path) -> None:
    mesh_dir = tmp_path / "room_mesh_obj_format" / "Office"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "Office_idx_0.obj").write_bytes(_shoebox_obj())
    record = b'{"src_loc": [1, 1, 1], "rec_loc": [2, 2, 1]}'
    (tmp_path / "metadata.zip").write_bytes(_zip_bytes({"metadata/Office/Office_idx_0/S001_R001.json": record}))
    category = _zip_bytes({"Office/Office_idx_0/S001_R001_hybrid_IR.wav": _wav_bytes((1, 1, 1), (2, 2, 1))})
    (tmp_path / "single_channel_ir.zip").write_bytes(_zip_bytes({"single_channel_ir/Office.zip": category}))
    room = ar.build_index(tmp_path).rooms["Office_idx_0"]
    assert room.mesh is not None and room.paired() == [(1, 1)]


def test_missing_root_and_bad_metadata_raise(tmp_path: Path, dataset_root: Path) -> None:
    with pytest.raises(ar.AcousticRoomsError):
        ar.resolve_root(tmp_path / "absent")
    with pytest.raises(ar.AcousticRoomsError):
        ar.build_index(tmp_path / "..")  # a folder with no AcousticRooms sources
    (dataset_root / "metadata-extra.zip").write_bytes(_zip_bytes({"metadata/Cafe/Cafe_idx_7/S002_R002.json": b'{"x": 1}'}))
    index = ar.build_index(dataset_root)
    with ar.ArchiveReader(dataset_root) as reader, pytest.raises(ar.AcousticRoomsError):
        ar.load_pair_metadata(reader, index.rooms["Cafe_idx_7"].metadata[(2, 2)])
