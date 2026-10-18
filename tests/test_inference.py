"""P10 inference: registry, verified samples, reconstruction, export and API, all synthetic.

A 64 x 64 shoebox sample is written in the P6 fixed-sample layout and a tiny,
randomly initialised EchoFusion D is saved as a p8 checkpoint, so these tests
need neither the dataset nor real weights.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
trimesh = pytest.importorskip("trimesh")

from fastapi.testclient import TestClient  # noqa: E402

from echospace.data.evalmasks import EVAL_FORMAT, content_sha256, spec_checksum  # noqa: E402
from echospace.inference import export  # noqa: E402
from echospace.inference.api import REQUEST_SCHEMA, Settings, create_app  # noqa: E402
from echospace.inference.predict import compose_interior, masked_iou, reconstruct  # noqa: E402
from echospace.inference.registry import ModelRegistry, RegistryError, file_sha256  # noqa: E402
from echospace.inference.samples import SampleError, SampleStore, model_inputs  # noqa: E402
from echospace.models.echofusion import build  # noqa: E402
from echospace.training.audio_frozen import fixed_items  # noqa: E402
from echospace.training.fusion_runner import masked_iou as torch_masked_iou  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "grid_v2_agreement.json"
CELL, SIZE = 0.2, 64
ORIGIN = (-6.4, 1.1, -6.4)
TINY = {"width": 16, "channels": [8, 8, 16, 16], "audio_channels": [8, 8, 16, 16], "fusion_layers": 1,
        "heads": 2, "feedforward": 32, "fourier_bands": 2}


def _centre(row: int, col: int, height: float = 1.2) -> list[float]:
    return [ORIGIN[0] + CELL * (col + 0.5), height, ORIGIN[2] + CELL * (row + 0.5)]


def _sample(room_id: str, group: int, seed: int) -> tuple[dict, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    occupancy = np.zeros((SIZE, SIZE), np.uint8)
    occupancy[16:48, 16:48] = 1
    boundary = occupancy.copy()
    boundary[17:47, 17:47] = 0
    free = np.zeros_like(occupancy)
    free[16:48, 16:32] = 1  # the scanner saw the left half of the room
    wall = np.zeros_like(occupancy)
    wall[16:48, 15] = 1  # wall hits just outside the left wall
    cells = [(int(r), int(c)) for r, c in zip(rng.integers(18, 46, 16), rng.integers(18, 30, 16))]
    positions = np.array([[_centre(*cells[i]), _centre(*cells[8 + i], 1.5)] for i in range(8)], np.float32)
    sample_id = f"{room_id}__viewpoint__mid__{seed}"
    record = {
        "schema_version": "0.1.0", "sample_id": sample_id, "room_id": room_id, "dataset": "synthetic",
        "geometry_id": room_id, "rir_ids": [f"S{i}_R{i}" for i in range(8)], "rir_sample_rate_hz": 16000,
        "sample_path": f"rooms/{room_id}/{sample_id}.npz", "split": "unassigned",
        "frame": {"description": "synthetic scene metres", "source_units_to_meters": 1.0,
                  "source_to_scene": np.eye(4).tolist()},
        "grid": {"width": SIZE, "height": SIZE, "cell_size_m": CELL, "origin_scene_m": list(ORIGIN),
                 "axis_u_scene": [1.0, 0.0, 0.0], "axis_v_scene": [0.0, 0.0, 1.0]},
        "coverage": 0.5, "coverage_bin": "mid", "occlusion_type": "viewpoint", "room_group": group,
        "scan_seed": seed, "eligible_pairs": 8,
    }
    arrays = {
        "observed_points_scene_m": np.array([_centre(r, 15) for r in range(16, 48)], np.float32),
        "observed_cells": (free | wall).astype(np.uint8), "observed_free": free, "observed_wall": wall,
        "valid_cells": np.ones_like(occupancy), "target_occupancy": occupancy, "target_boundary": boundary,
        "rir_waveforms": rng.standard_normal((8, 1280)).astype(np.float32),
        "rir_positions_scene_m": positions, "rir_valid": np.ones(8, np.uint8),
    }
    return record, arrays


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A miniature repository: eval samples, spec, folds, registry and a tiny checkpoint."""
    rooms = {"RoomTest_idx_0": ("test", 0), "RoomTrain_idx_0": ("train", 1)}
    entries, records = [], []
    for seed, (room_id, (_, group)) in enumerate(rooms.items(), 1):
        record, arrays = _sample(room_id, group, seed)
        path = tmp_path / "eval" / record["sample_path"]
        path.parent.mkdir(parents=True)
        np.savez_compressed(path, **arrays)
        records.append(record)
        entries.append({key: record[key] for key in ("sample_id", "room_id", "occlusion_type", "coverage_bin",
                                                     "coverage", "scan_seed", "eligible_pairs")}
                       | {"sha256": content_sha256(record, arrays)})
    (tmp_path / "eval" / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    spec = {"format": EVAL_FORMAT, "samples": entries}
    spec["checksum"] = spec_checksum(spec)
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    folds = {"folds": [{"fold": 0, "train_rooms": ["RoomTrain_idx_0"], "val_rooms": [], "test_rooms": ["RoomTest_idx_0"]}]}
    (tmp_path / "folds.json").write_text(json.dumps(folds), encoding="utf-8")

    torch.manual_seed(0)
    model = build("D", TINY)
    checkpoint = tmp_path / "artifacts" / "tiny.pt"
    checkpoint.parent.mkdir()
    torch.save({"format": "p8.checkpoint.1", "model": model.specification(), "state_dict": model.state_dict(),
                "frozen_checksum": "f" * 64}, checkpoint)
    registry = {"format": "echospace.models.1", "models": [{
        "model_id": "tiny-d", "family": "EchoFusion", "variant": "D", "checkpoint": "artifacts/tiny.pt",
        "checkpoint_sha256": file_sha256(checkpoint), "checkpoint_format": "p8.checkpoint.1",
        "frozen_checksum": "f" * 64, "fold": 0, "seed": 0, "threshold": 0.5,
        "folds_file_sha256": file_sha256(tmp_path / "folds.json"), "eval_spec_checksum": spec["checksum"],
        "obtain": "generated by the test"}]}
    (tmp_path / "models.json").write_text(json.dumps(registry), encoding="utf-8")
    return tmp_path


def _store(repo: Path) -> SampleStore:
    return SampleStore(repo / "eval", repo / "spec.json", repo / "folds.json")


def _registry(repo: Path) -> ModelRegistry:
    return ModelRegistry(repo / "models.json", repo, repo / "folds.json", repo / "spec.json")


def _app(repo: Path) -> TestClient:
    settings = Settings(repo_root=repo, models="models.json", eval_dir="eval", eval_spec="spec.json", folds="folds.json")
    return TestClient(create_app(settings))


def _request(**overrides: object) -> dict:
    return {"schema_version": REQUEST_SCHEMA, "model_id": "tiny-d", "sample_id": "RoomTest_idx_0__viewpoint__mid__1"} | overrides


def test_model_inputs_are_exactly_the_p9_evaluation_inputs(repo: Path) -> None:
    store = _store(repo)
    sample = store.load("RoomTest_idx_0__viewpoint__mid__1")
    reference = next(fixed_items(repo / "eval", store.folds, store.spec, 0, "test"))
    ours = model_inputs(sample, 8)
    for key in ("observed_cells", "observed_free", "observed_wall", "rir", "rir_valid", "src_pos", "mic_pos"):
        np.testing.assert_array_equal(ours[key][0], reference[key], err_msg=key)
    three = model_inputs(sample, 3)
    assert int(three["k_used"][0]) == 3 and three["rir_valid"][0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert not three["rir"][0, 3:].any() and not three["src_pos"][0, 3:].any()
    np.testing.assert_array_equal(three["rir"][0, :3], reference["rir"][:3])
    with pytest.raises(ValueError):
        model_inputs(sample, 0)


def test_sample_store_refuses_unknown_unprepared_and_tampered_samples(repo: Path) -> None:
    store = _store(repo)
    with pytest.raises(SampleError, match="unknown"):
        store.load("nope")
    assert store.split_of("RoomTest_idx_0", 0) == "test" and store.split_of("RoomTrain_idx_0", 0) == "train"
    path = repo / "eval" / "rooms" / "RoomTest_idx_0" / "RoomTest_idx_0__viewpoint__mid__1.npz"
    with np.load(path) as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays["rir_waveforms"][0, 0] += 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(SampleError, match="verification"):
        _store(repo).load("RoomTest_idx_0__viewpoint__mid__1")


def test_registry_serves_only_verified_checkpoints(repo: Path) -> None:
    registry = _registry(repo)
    assert registry.problem("tiny-d") is None
    assert not registry.load("tiny-d").training
    with pytest.raises(RegistryError, match="unknown"):
        registry.get("other")
    (repo / "artifacts" / "tiny.pt").write_bytes(b"tampered")
    assert "SHA-256" in _registry(repo).problem("tiny-d")
    (repo / "artifacts" / "tiny.pt").unlink()
    assert "missing" in _registry(repo).problem("tiny-d")
    (repo / "folds.json").write_text('{"folds": []}', encoding="utf-8")
    assert "folds.json" in _registry(repo).problem("tiny-d")


def test_reconstruction_is_deterministic_and_scored_like_p9(repo: Path) -> None:
    store, registry = _store(repo), _registry(repo)
    sample = store.load("RoomTest_idx_0__viewpoint__mid__1")
    model, entry = registry.load("tiny-d"), registry.get("tiny-d")
    first = reconstruct(model, entry, sample, 8, split="test", revision="test")
    second = reconstruct(model, entry, sample, 8, split="test", revision="test")
    assert first.reconstruction_id == second.reconstruction_id
    np.testing.assert_array_equal(first.occupancy_probability, second.occupancy_probability)
    assert reconstruct(model, entry, sample, 4, split="test", revision="test").reconstruction_id != first.reconstruction_id

    arrays = sample.arrays
    mask = (arrays["observed_cells"] == 0) & (arrays["valid_cells"] == 1)
    for head in ("occupancy", "boundary"):
        probability = getattr(first, f"{head}_probability")
        expected = torch_masked_iou(torch.from_numpy(probability)[None], torch.from_numpy(arrays[f"target_{head}"])[None],
                                    torch.from_numpy(mask)[None])
        assert first.metrics[f"{head}_iou"] == pytest.approx(float(expected[0]))
    assert first.provenance["sample"]["sha256"] == sample.sha256
    assert first.provenance["model"]["checkpoint_sha256"] == entry.checkpoint_sha256
    assert "checkpoint" not in first.provenance["model"]  # no local paths leave the machine


def test_display_composition_pastes_only_measured_free_cells() -> None:
    probability = np.array([[0.9, 0.1, 0.9], [0.2, 0.8, 0.4]], np.float32)
    free = np.array([[0, 1, 0], [0, 0, 0]], np.uint8)
    wall = np.array([[1, 0, 0], [0, 0, 0]], np.uint8)
    assert compose_interior(probability, free, wall, 0.5).tolist() == [[0, 1, 1], [0, 1, 0]]
    assert masked_iou(probability, np.zeros((2, 3)), np.zeros((2, 3), bool)) == 1.0


def test_api_endpoints_and_rejections(repo: Path) -> None:
    client = _app(repo)
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["models_servable"] == 1 and health["samples_prepared"] == 2
    models = client.get("/v1/models").json()["models"]
    assert models[0]["available"] and "checkpoint" not in models[0]
    listed = client.get("/v1/samples").json()
    assert [s["room_id"] for s in listed["samples"]] == ["RoomTest_idx_0"]

    created = client.post("/v1/reconstructions", json=_request(k=4))
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["k_used"] == 4 and body["split_for_model_fold"] == "test"
    for link in ("self", "grid", "glb", "provenance", "prediction"):
        assert client.get(body["links"][link]).status_code == 200, link
    assert client.get(body["links"]["glb"]).content[:4] == b"glTF"
    assert client.get(body["links"]["grid"]).json()["format"] == "echospace.grid.v2"

    assert client.post("/v1/reconstructions", json=_request(model_id="nope")).status_code == 404
    assert client.post("/v1/reconstructions", json=_request(sample_id="nope")).status_code == 404
    assert client.post("/v1/reconstructions", json=_request(schema_version="v0")).status_code == 422
    assert client.post("/v1/reconstructions", json=_request(mesh="raw.obj")).status_code == 422
    assert client.post("/v1/reconstructions", json=_request(k=9)).status_code == 422
    seen = _request(sample_id="RoomTrain_idx_0__viewpoint__mid__2")
    assert client.post("/v1/reconstructions", json=seen).status_code == 409
    assert client.post("/v1/reconstructions", json=seen | {"allow_seen_rooms": True}).status_code == 201
    assert client.get("/v1/reconstructions/doesnotexist").status_code == 404


def test_api_reports_unservable_checkpoint(repo: Path) -> None:
    (repo / "artifacts" / "tiny.pt").unlink()
    client = _app(repo)
    assert client.get("/health").json()["status"] == "degraded"
    assert "missing" in client.get("/v1/models").json()["models"][0]["unavailable_reason"]
    assert client.post("/v1/reconstructions", json=_request()).status_code == 503


def _grid(heading_rad: float = 0.0) -> dict:
    interior = np.zeros((5, 6), int)
    interior[1:4, 1:5] = 1
    interior[3, 4] = 0  # an L-shape
    wall = np.zeros_like(interior)
    wall[1:4, 0] = 1  # hits beside the left wall
    u = [math.cos(heading_rad), 0.0, math.sin(heading_rad)]
    v = [-math.sin(heading_rad), 0.0, math.cos(heading_rad)]
    return {"format": "echospace.grid.v2", "name": "agreement", "cell_size_m": 0.2, "width": 6, "height": 5,
            "origin_scene_m": [1.0, 1.1, -2.0], "axis_u_scene": u, "axis_v_scene": v, "wall_height_m": 2.5,
            "wall_height_assumed": True, "interior": interior.tolist(), "observed": (interior | wall).tolist(),
            "observed_wall": wall.tolist()}


def test_wall_rule_and_glb_round_trip() -> None:
    document = _grid()
    stats = export.geometry_stats(document)
    # 11 interior cells; cutting a corner off a 3 x 4 block keeps its 14-edge perimeter,
    # 3 of which face wall hits
    assert stats["floor_triangles"] == 22
    assert stats["observed_wall_triangles"] == 6 and stats["inferred_wall_triangles"] == 22
    assert stats["observed_wall_m"] == pytest.approx(0.6) and stats["area_m2"] == pytest.approx(0.44)
    assert stats["bounds_min"][1] == 0.0 and stats["bounds_max"][1] == 2.5  # floor at Y = 0, assumed height
    scene = trimesh.load(trimesh.util.wrap_as_stream(export.glb_bytes(document)), file_type="glb")
    assert set(scene.geometry) == {"floor", "observed_walls", "inferred_walls"}
    np.testing.assert_allclose(scene.bounds, [stats["bounds_min"], stats["bounds_max"]], atol=1e-5)
    bad = copy.deepcopy(document)
    bad["axis_v_scene"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="orthonormal"):
        export.validate_grid_v2(bad)


def test_committed_agreement_fixture_matches_python_geometry() -> None:
    """The browser renderer is checked against the same fixture by apps/web/scripts/check-grid.mjs."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["document"] == _grid(math.radians(30))
    assert export.geometry_stats(fixture["document"]) == fixture["expected"]
    digest = hashlib.sha256(json.dumps(fixture["document"], sort_keys=True).encode()).hexdigest()
    assert fixture["document_sha256"] == digest


def test_room_scene_schema_and_prediction_independent_measured_layer(repo: Path) -> None:
    from dataclasses import replace
    from jsonschema import Draft202012Validator
    from echospace.inference.room_scene import room_scene_document
    store, registry = _store(repo), _registry(repo)
    result = reconstruct(registry.load('tiny-d'),registry.get('tiny-d'),
                         store.load('RoomTest_idx_0__viewpoint__mid__1'),4,split='test',revision='test')
    scene = room_scene_document(result)
    schema = Path(__file__).resolve().parents[1]/'contracts/v0.1.0/room-scene.schema.json'
    Draft202012Validator(json.loads(schema.read_text(encoding='utf-8'))).validate(scene)
    changed = room_scene_document(replace(result,interior=1-result.interior))
    assert scene['observed_geometry'] and scene['observed_geometry']==changed['observed_geometry']
    assert scene['inferred_geometry']!=changed['inferred_geometry']
    assert all(w['provenance']=='measured' for w in scene['observed_geometry'])
    assert all(w['provenance']=='model' for w in scene['inferred_geometry'])
    assert len(scene['acoustic_measurements'])==4
    assert scene['acoustic_measurements'][0]['source_scene_m']==result.sample.arrays['rir_positions_scene_m'][0,0].tolist()
    assert scene['observations'][0]['sensor_type']=='simulated_partial_scan'


def test_room_scene_and_observation_downloads_and_restart_recovery(repo: Path) -> None:
    import io
    client = _app(repo)
    created = client.post('/v1/reconstructions',json=_request(k=4)).json()
    original = client.get(created['links']['room_scene'])
    assert original.status_code==200
    scene = original.json()
    observation = client.get(scene['observations'][0]['payload_ref'])
    assert observation.status_code==200
    with np.load(io.BytesIO(observation.content),allow_pickle=False) as data:
        assert 'target_occupancy' not in data and 'target_boundary' not in data
        assert 'observed_points_scene_m' in data and 'rir_waveforms' in data
    restarted = _app(repo)
    assert restarted.get('/health').json()['persistent_requests'] is True
    restored = restarted.get(created['links']['grid'])
    assert restored.status_code==200
    assert restored.json()==client.get(created['links']['grid']).json()
    path = repo/'artifacts/inference/results'/f"{created['reconstruction_id']}.json"
    path.write_text(json.dumps(_request(k=1)),encoding='utf-8')
    assert _app(repo).get(created['links']['grid']).status_code==409


def test_archived_mesh_store_and_api_reuse_p2_without_extraction(repo: Path) -> None:
    import io
    import zipfile
    from echospace.inference.meshes import MeshStore
    root = repo/'raw'; root.mkdir()
    body = b'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n'
    inner = io.BytesIO()
    member = 'room_mesh_obj_format/Test/Room_idx_0.obj'
    with zipfile.ZipFile(inner,'w') as z: z.writestr(member,body)
    with zipfile.ZipFile(root/'outer.zip','w') as z: z.writestr('meshes.zip',inner.getvalue())
    manifest = repo/'meshes.jsonl'
    manifest.write_text(json.dumps({'room_id':'Room_idx_0','category':'Test','mesh_bytes':len(body),
                                    'mesh_path':'outer.zip :: meshes.zip :: '+member})+'\n',encoding='utf-8')
    store = MeshStore(root,manifest)
    assert store.catalogue()['rooms'][0]['storage']=='archive'
    assert store.read('Test/Room_idx_0')[0]==body
    with pytest.raises(LookupError): store.read('../outer.zip')
    settings = Settings(repo_root=repo,models='models.json',eval_dir='eval',eval_spec='spec.json',folds='folds.json',
                        mesh_root='raw',mesh_manifest='meshes.jsonl')
    client = TestClient(create_app(settings))
    assert client.get('/v1/acousticrooms').json()['available']
    response = client.get('/v1/acousticrooms/mesh',params={'id':'Test/Room_idx_0'})
    assert response.status_code==200 and response.content==body
    assert response.headers['etag']=='"'+hashlib.sha256(body).hexdigest()+'"'
    assert client.get('/v1/acousticrooms/mesh',params={'id':'../outer.zip'}).status_code==404
    assert sorted(p.name for p in root.iterdir())==['outer.zip']
