"""Direct-path arithmetic, timing verdicts and canvas eligibility on a synthetic shoebox."""

from __future__ import annotations

import numpy as np
import pytest

from echospace.io.audit import (
    CANVAS_12_8_M,
    CANVAS_25_6_M,
    anchored_coverage,
    anchored_coverage_counts,
    direct_path_samples,
    first_strong_peak,
    fits_bbox,
    summarize_timing,
)

FS = 22050
ROOM_XYZ = np.array([8.0, 3.0, 6.0])  # metres; Y up, as in the scene contract
TRIM_OFFSET = 40  # samples a fixed upstream trim would leave before the direct sound


def _positions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Points at least 0.5 m from every shoebox surface."""
    return rng.uniform(0.5, ROOM_XYZ - 0.5, size=(n, 3))


def _synthetic_rir(direct: float, length: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rir = np.zeros(length)
    arrival = int(round(direct)) + TRIM_OFFSET
    rir[arrival] = 1.0
    tail = np.arange(length - arrival - 5)
    rir[arrival + 5 :] = 0.3 * rng.standard_normal(tail.size) * np.exp(-tail / 600.0)
    return rir


def test_direct_path_samples_matches_hand_computation() -> None:
    # 3-4-5 triangle: 5 m at 343 m/s and 22.05 kHz
    assert direct_path_samples([0, 0, 0], [3, 4, 0], FS) == pytest.approx(5 / 343 * FS)
    assert direct_path_samples([1, 1, 1], [1, 1, 1], FS) == 0.0


def test_first_strong_peak_finds_direct_sound_not_reflections() -> None:
    rir = _synthetic_rir(direct=300.0)
    assert first_strong_peak(rir) == 300 + TRIM_OFFSET
    with pytest.raises(ValueError):
        first_strong_peak(np.zeros(10))


def test_constant_offset_detected_for_correct_coordinates() -> None:
    rng = np.random.default_rng(1)
    src, mic = _positions(rng, 30), _positions(rng, 30)
    direct = [direct_path_samples(s, m, FS) for s, m in zip(src, mic)]
    peaks = [first_strong_peak(_synthetic_rir(d, seed=i)) for i, d in enumerate(direct)]
    summary = summarize_timing(direct, peaks)
    assert summary.verdict == "constant_offset"
    assert summary.offset_mean == pytest.approx(TRIM_OFFSET, abs=1.0)
    assert summary.slope == pytest.approx(1.0, abs=0.01)


def test_axis_swap_is_flagged_as_inconsistent() -> None:
    rng = np.random.default_rng(2)
    src, mic = _positions(rng, 30), _positions(rng, 30)
    true_direct = [direct_path_samples(s, m, FS) for s, m in zip(src, mic)]
    peaks = [first_strong_peak(_synthetic_rir(d, seed=i)) for i, d in enumerate(true_direct)]
    swap = [0, 2, 1]  # metadata read with Y and Z exchanged for receivers only
    wrong_direct = [direct_path_samples(s, m[swap], FS) for s, m in zip(src, mic)]
    assert summarize_timing(wrong_direct, peaks).verdict == "inconsistent"


def test_centimetre_units_are_flagged_as_inconsistent() -> None:
    rng = np.random.default_rng(3)
    src, mic = _positions(rng, 30), _positions(rng, 30)
    true_direct = [direct_path_samples(s, m, FS) for s, m in zip(src, mic)]
    peaks = [first_strong_peak(_synthetic_rir(d, seed=i)) for i, d in enumerate(true_direct)]
    wrong_direct = [direct_path_samples(s * 100, m * 100, FS) for s, m in zip(src, mic)]
    assert summarize_timing(wrong_direct, peaks).verdict == "inconsistent"


def test_per_pair_onset_trim_is_distinguished_from_coordinate_error() -> None:
    rng = np.random.default_rng(4)
    src, mic = _positions(rng, 30), _positions(rng, 30)
    direct = [direct_path_samples(s, m, FS) for s, m in zip(src, mic)]
    summary = summarize_timing(direct, [10.0] * len(direct))
    assert summary.verdict == "per_pair_onset_trim"
    assert summarize_timing([100.0], [140.0]).verdict == "insufficient"


def _rectangle(width: float, depth: float) -> np.ndarray:
    return np.array([[0, 0], [width, 0], [width, depth], [0, depth]], dtype=float)


def test_bbox_rule_by_canvas_size() -> None:
    assert fits_bbox(_rectangle(8.0, 6.0), CANVAS_12_8_M)
    assert fits_bbox(_rectangle(12.8, 12.8), CANVAS_12_8_M)
    assert not fits_bbox(_rectangle(14.0, 6.0), CANVAS_12_8_M)
    assert fits_bbox(_rectangle(14.0, 6.0), CANVAS_25_6_M)


def test_anchor_rule_can_fail_where_bbox_passes() -> None:
    room = _rectangle(10.0, 6.0)
    assert fits_bbox(room, CANVAS_12_8_M)
    assert anchored_coverage(room, [5.0, 3.0], CANVAS_12_8_M)  # centred pose: 5 m reach < 6.4 m
    assert not anchored_coverage(room, [1.0, 3.0], CANVAS_12_8_M)  # near a wall: 9 m reach
    assert anchored_coverage(room, [1.0, 3.0], CANVAS_25_6_M)
    poses = np.array([[5.0, 3.0], [1.0, 3.0], [9.0, 1.0], [4.0, 4.0]])
    assert anchored_coverage_counts(room, poses, CANVAS_12_8_M) == 2
