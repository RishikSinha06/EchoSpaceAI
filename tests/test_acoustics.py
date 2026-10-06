"""P5 timing, resampling, identity gates, nested bundles and training isolation."""

from __future__ import annotations

import io
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.io import wavfile

pytest.importorskip("trimesh")
pytest.importorskip("shapely")

from echospace.acoustics import (  # noqa: E402
    AcousticError, AugmentConfig, RirCache, RirCandidate, RirConfig, TimingEvidence,
    augment_bundle, build_bundle, load_audited_candidates, preprocess_wav,
)
from echospace.contract_v0 import validate_sample  # noqa: E402
from echospace.geometry.frames import GridFrame  # noqa: E402
from echospace.io.adapters import acousticrooms as ar  # noqa: E402
from echospace.io.audit_geometry import geometry_signature  # noqa: E402

TIMING = TimingEvidence(emission_time_verified=True, upstream_trimmed=False, evidence="synthetic test")
CONFIG = RirConfig()
GRID = GridFrame((0, 1.1, 0), 0)
FREE = np.ones((64, 64), dtype=np.uint8)


def wav(samples: np.ndarray, fs: int = 22050) -> bytes:
    buffer = io.BytesIO()
    wavfile.write(buffer, fs, samples)
    return buffer.getvalue()


def impulse_bytes(gain: float = 1.0, fs: int = 22050) -> bytes:
    samples = np.zeros(fs // 10, dtype=np.float32)
    samples[round(fs * 0.02)] = gain * 0.2
    samples[round(fs * 0.05)] = gain * 0.7  # the reflection is stronger than the direct arrival
    return wav(samples, fs)


def candidate(i: int, verified: bool = True) -> RirCandidate:
    return RirCandidate("room", (i, 0), np.array([[0.1, 1.1, 0.1], [1.1, 1.1, 1.1]], dtype=np.float32),
                        ar.MemberRef((f"{i}.wav",)), ar.MemberRef((f"{i}.json",)), "metadata", "audit", verified=verified)


def loader(item: RirCandidate):
    return preprocess_wav(impulse_bytes(1 + item.pair[0] / 100), CONFIG, TIMING)


def bundle(k=8, candidates=None, free=FREE, pad_missing=False):
    return build_bundle(candidates if candidates is not None else [candidate(i) for i in range(10)], GRID, free,
                        loader, room_id="room", selection_key="scan-7", seed=42, k=k, pad_missing=pad_missing)


def contract_record(ids):
    return {"schema_version": "0.1.0", "sample_id": "sample", "room_id": "room", "dataset": "synthetic",
            "geometry_id": "room", "sample_path": "sample.npz", "split": "train", "rir_ids": list(ids),
            "rir_sample_rate_hz": 16000, "frame": {"description": "scene metres", "source_units_to_meters": 1,
            "source_to_scene": np.eye(4).tolist()}, "grid": GRID.manifest_grid()}


def contract_arrays(result, free=FREE):
    arrays = result.contract_arrays()
    arrays.update(observed_points_scene_m=np.zeros((0, 3)), observed_cells=free, observed_free=free,
                  observed_wall=np.zeros_like(free), valid_cells=np.ones_like(free),
                  target_occupancy=np.ones_like(free), target_boundary=np.zeros_like(free))
    return arrays


def test_preserves_emission_delay_and_reflection_spacing_at_16khz():
    result = preprocess_wav(impulse_bytes(), CONFIG, TIMING)
    assert result.waveform.shape == (1280,) and result.waveform.dtype == np.float32
    assert np.argmax(np.abs(result.waveform[310:330])) + 310 == 320
    assert np.argmax(np.abs(result.waveform)) == 800  # no strongest-peak alignment
    assert result.metadata["window_start_native_sample"] == 0
    assert result.metadata["native_sample_rate_hz"] == 22050


def test_preserve_gain_and_optional_explicit_peak_normalization():
    first = preprocess_wav(impulse_bytes(), CONFIG, TIMING)
    twice = preprocess_wav(impulse_bytes(2), CONFIG, TIMING)
    assert np.allclose(twice.waveform, 2 * first.waveform)
    normalized = preprocess_wav(impulse_bytes(), replace(CONFIG, normalization="peak"), TIMING)
    assert abs(normalized.waveform).max() == pytest.approx(1)
    assert normalized.metadata["normalization_divisor"] != 1


def test_short_waveform_is_right_padded_and_pcm_is_scaled():
    pcm = np.zeros(80, dtype=np.int16)
    pcm[10] = -32768
    result = preprocess_wav(wav(pcm, 16000), CONFIG, TIMING)
    assert result.waveform[10] == -1 and not result.waveform[80:].any()
    assert result.metadata["valid_output_samples"] == 80
    assert result.metadata["right_padding_samples"] == 1200
    unsigned = np.full(80, 128, dtype=np.uint8)
    unsigned[10] = 192
    assert preprocess_wav(wav(unsigned, 16000), CONFIG, TIMING).waveform[10] == 0.5


def test_downsampling_suppresses_above_nyquist_tone():
    fs = 32000
    time = np.arange(fs // 10) / fs
    low = preprocess_wav(wav(np.sin(2 * np.pi * 1000 * time).astype(np.float32), fs), CONFIG, TIMING)
    high = preprocess_wav(wav(np.sin(2 * np.pi * 12000 * time).astype(np.float32), fs), CONFIG, TIMING)
    assert np.linalg.norm(high.waveform[100:-100]) < 0.01 * np.linalg.norm(low.waveform[100:-100])


@pytest.mark.parametrize("samples", [np.zeros(20), np.full(20, np.nan), np.ones((20, 2))])
def test_rejects_silent_nonfinite_and_stereo(samples):
    with pytest.raises(AcousticError):
        preprocess_wav(wav(samples), CONFIG, TIMING)


def test_alignment_requires_the_corresponding_evidence():
    with pytest.raises(AcousticError, match="verified"):
        preprocess_wav(impulse_bytes(), CONFIG, TimingEvidence())
    with pytest.raises(AcousticError, match="direct-onset"):
        preprocess_wav(impulse_bytes(), replace(CONFIG, alignment="verified_direct"),
                       TimingEvidence(onset_native_sample=441))
    verified = TimingEvidence(onset_native_sample=441, direct_onset_verified=True)
    direct = preprocess_wav(impulse_bytes(), replace(CONFIG, alignment="verified_direct"), verified)
    assert direct.metadata["window_start_seconds"] == pytest.approx(0.02)
    assert abs(direct.waveform[0]) > 0
    relative = preprocess_wav(impulse_bytes(), replace(CONFIG, alignment="relative_onset"),
                             TimingEvidence(onset_native_sample=441, upstream_trimmed=True))
    assert not relative.metadata["timing"]["emission_time_verified"]


def test_cache_round_trip_config_invalidation_and_corruption(tmp_path):
    cache = RirCache(tmp_path)
    clean = cache.get(impulse_bytes(), CONFIG, TIMING)
    again = cache.get(impulse_bytes(), CONFIG, TIMING)
    assert np.array_equal(clean.waveform, again.waveform) and clean.metadata == again.metadata
    changed = cache.get(impulse_bytes(), replace(CONFIG, window_ms=40), TIMING)
    assert changed.metadata["cache_key"] != clean.metadata["cache_key"]
    path = tmp_path / f"{clean.metadata['cache_key']}.npz"
    path.write_bytes(b"corrupt")
    with pytest.raises(AcousticError, match="corrupt"):
        cache.get(impulse_bytes(), CONFIG, TIMING)


def test_k_bundles_are_nested_and_order_independent():
    maximum = bundle()
    for k in (1, 2, 4, 8):
        small = bundle(k, list(reversed([candidate(i) for i in range(10)])))
        assert small.rir_ids == maximum.rir_ids[:k]
        assert np.array_equal(small.waveforms, maximum.waveforms[:k])
        validate_sample(contract_record(small.rir_ids), contract_arrays(small))


def test_unverified_and_hidden_pose_pairs_never_reach_loader():
    hidden = candidate(2)
    hidden.positions_scene_m[1] = [6.3, 1.1, 6.3]
    free = FREE.copy()
    free[63, 63] = 0
    items = [candidate(0), candidate(1, verified=False), hidden]
    result = bundle(4, items, free, pad_missing=True)
    assert result.valid.tolist() == [1, 0, 0, 0]
    assert not result.waveforms[1:].any() and not result.positions_scene_m[1:].any()
    assert {x["reason"] for x in result.selection["rejected"]} == {"unverified_pair", "pose_not_in_observed_free"}
    validate_sample(contract_record(result.rir_ids), contract_arrays(result, free))
    with pytest.raises(AcousticError, match="need K"):
        bundle(4, items, free)
    with pytest.raises(AcousticError, match="duplicate"):
        bundle(1, [candidate(0), candidate(0)])


def test_training_augmentation_is_reproducible_prefix_stable_and_does_not_mutate_clean():
    clean = bundle()
    original = clean.waveforms.copy()
    args = dict(split="train", seed=123, sample_key="scan-7")
    first = augment_bundle(clean, GRID, FREE, **args)
    again = augment_bundle(clean, GRID, FREE, **args)
    smaller = augment_bundle(clean.prefix(2), GRID, FREE, **args)
    assert np.array_equal(first.waveforms, again.waveforms)
    assert np.array_equal(first.waveforms[:2], smaller.waveforms)
    assert np.array_equal(first.positions_scene_m[:2], smaller.positions_scene_m)
    assert np.array_equal(clean.waveforms, original) and not np.array_equal(original, first.waveforms)
    assert np.array_equal(clean.positions_scene_m[:, :, 1], first.positions_scene_m[:, :, 1])
    validate_sample(contract_record(first.rir_ids), contract_arrays(first))


@pytest.mark.parametrize("split", ["val", "test"])
def test_evaluation_never_augments(split):
    clean = bundle()
    result = augment_bundle(clean, GRID, FREE, split=split, seed=123, sample_key="scan-7")
    assert np.array_equal(clean.waveforms, result.waveforms)
    assert np.array_equal(clean.positions_scene_m, result.positions_scene_m)
    assert result.provenance == clean.provenance


def test_invalid_padding_stays_zero_and_large_pose_noise_never_leaves_observation():
    clean = bundle(4, [candidate(0)], pad_missing=True)
    free = np.zeros_like(FREE)
    free[32, 32] = free[37, 37] = 1
    result = augment_bundle(clean, GRID, free, split="train", seed=0, sample_key="narrow",
                            config=AugmentConfig(pose_std_m=20, pose_attempts=1))
    assert not result.waveforms[1:].any() and not result.positions_scene_m[1:].any()
    assert np.array_equal(result.positions_scene_m, clean.positions_scene_m)
    validate_sample(contract_record(result.rir_ids), contract_arrays(result, free))
    with pytest.raises(AcousticError, match="assigned"):
        augment_bundle(clean, GRID, FREE, split="unassigned", seed=0, sample_key="x")


def test_loader_requires_matching_accepted_audit_and_detects_pose_conflicts(tmp_path):
    import trimesh

    mesh = trimesh.creation.box()
    mesh_bytes = mesh.export(file_type="obj").encode("utf-8")
    (tmp_path / "room.obj").write_bytes(mesh_bytes)
    room = ar.RoomEntry("room", "synthetic", ar.MemberRef(("room.obj",)))
    for rec in range(2):
        path = f"{rec}.json"
        (tmp_path / path).write_text(json.dumps({"src_loc": [rec, 0, 1], "rec_loc": [0, rec, 1]}))
        room.metadata[(0, rec)] = ar.MemberRef((path,))
        room.rirs[(0, rec)] = ar.MemberRef((f"{rec}.wav",))
    audit = {"dataset": "AcousticRooms", "room_id": "room", "accepted": True, "up_axis": 2,
             "units": "metres", "position_conflicts": 0, "timing": {"verdict": "constant_offset"},
             "geometry_signature": geometry_signature(mesh.vertices), "mesh_bytes": len(mesh_bytes),
             "mesh": {"bounds_min": mesh.bounds[0].tolist(), "bounds_max": mesh.bounds[1].tolist()}}
    with ar.ArchiveReader(tmp_path) as reader:
        with pytest.raises(AcousticError, match="accepted"):
            load_audited_candidates(reader, room, {**audit, "accepted": False})
        with pytest.raises(AcousticError, match="signature"):
            load_audited_candidates(reader, room, {**audit, "geometry_signature": "wrong"})
        moved = {"bounds_min": (mesh.bounds[0] + [1, 0, 0]).tolist(),
                 "bounds_max": (mesh.bounds[1] + [1, 0, 0]).tolist()}
        with pytest.raises(AcousticError, match="signature"):
            load_audited_candidates(reader, room, {**audit, "mesh": moved})
        with pytest.raises(AcousticError, match="conflicting"):
            load_audited_candidates(reader, room, audit)
    (tmp_path / "1.json").write_text(json.dumps({"src_loc": [0, 0, 1], "rec_loc": [0, 1, 1]}))
    (tmp_path / "room.obj").write_bytes(mesh_bytes.replace(b"\n", b"\r\n"))
    with ar.ArchiveReader(tmp_path) as reader:
        candidates = load_audited_candidates(reader, room, audit)
        assert len(candidates) == 2 and all(c.verified for c in candidates)
        with pytest.raises(AcousticError, match="signature"):
            load_audited_candidates(reader, room, {**audit, "mesh_bytes": 1})
