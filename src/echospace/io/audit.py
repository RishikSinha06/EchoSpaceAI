"""Pure, dataset-agnostic checks used by the D0 data audit.

Nothing here reads files or mutates state. Inputs are metric coordinates,
waveforms and 2D footprints; outputs are plain numbers and dicts so the audit
script can serialize them directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

SPEED_OF_SOUND_M_S = 343.0
CANVAS_12_8_M = 12.8
CANVAS_25_6_M = 25.6


def direct_path_samples(
    source_xyz: Sequence[float],
    receiver_xyz: Sequence[float],
    sample_rate_hz: float,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> float:
    """Direct-path propagation delay in (fractional) samples."""
    distance = float(np.linalg.norm(np.asarray(source_xyz, float) - np.asarray(receiver_xyz, float)))
    return distance / speed_of_sound_m_s * sample_rate_hz


def first_strong_peak(
    waveform: np.ndarray,
    relative_threshold: float = 0.5,
    refine_samples: int = 16,
) -> int:
    """Index of the first peak whose magnitude reaches a fraction of the maximum.

    The first sample crossing ``relative_threshold * max|x|`` is found, then
    refined to the local magnitude maximum within ``refine_samples`` after it,
    so a rising edge of the direct sound resolves to its crest.
    """
    magnitude = np.abs(np.asarray(waveform, dtype=np.float64).ravel())
    if magnitude.size == 0 or not np.isfinite(magnitude).all():
        raise ValueError("waveform must be non-empty and finite")
    peak = float(magnitude.max())
    if peak <= 0:
        raise ValueError("waveform is silent")
    first = int(np.argmax(magnitude >= relative_threshold * peak))
    window = magnitude[first : first + refine_samples + 1]
    return first + int(np.argmax(window))


@dataclass(frozen=True)
class TimingSummary:
    """How measured peak positions relate to predicted direct-path delays.

    ``offset = peak - direct``. A constant offset (small ``offset_std``) means
    a fixed onset was trimmed. A slope near 0 means each RIR was trimmed at its
    own arrival, so absolute delay carries no geometry. Anything else points
    to a coordinate, unit or frame error.
    """

    n_pairs: int
    offset_mean: float
    offset_std: float
    offset_min: float
    offset_max: float
    peak_std: float
    slope: float
    intercept: float
    residual_std: float
    verdict: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def summarize_timing(
    direct_samples: Iterable[float],
    peak_samples: Iterable[float],
    constant_tolerance_samples: float = 2.0,
) -> TimingSummary:
    """Classify the timing convention across many pairs.

    Verdicts: ``constant_offset`` (expected after a fixed trim),
    ``per_pair_onset_trim`` (peak fixed regardless of distance),
    ``inconsistent`` (likely coordinate/unit error; must not be patched per
    pair) and ``insufficient`` (fewer than 2 pairs).
    """
    direct = np.asarray(list(direct_samples), dtype=np.float64)
    peaks = np.asarray(list(peak_samples), dtype=np.float64)
    if direct.shape != peaks.shape:
        raise ValueError("direct and peak arrays must match")
    offsets = peaks - direct
    n = int(direct.size)
    if n == 0:
        nan = float("nan")
        return TimingSummary(0, nan, nan, nan, nan, nan, nan, nan, nan, "insufficient")
    std = float(offsets.std(ddof=1)) if n > 1 else 0.0
    peak_std = float(peaks.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and float(direct.std()) > 0:
        slope, intercept = (float(v) for v in np.polyfit(direct, peaks, 1))
        residual_std = float((peaks - (slope * direct + intercept)).std(ddof=1))
    else:
        slope, intercept, residual_std = float("nan"), float(offsets.mean()), float("nan")

    if n < 2:
        verdict = "insufficient"
    elif std <= constant_tolerance_samples:
        verdict = "constant_offset"
    elif peak_std <= constant_tolerance_samples:
        # Peaks do not move with distance at all. A unit error (e.g. cm read
        # as m) also gives a near-zero slope, but its peaks still spread.
        verdict = "per_pair_onset_trim"
    else:
        verdict = "inconsistent"
    return TimingSummary(
        n_pairs=n,
        offset_mean=float(offsets.mean()),
        offset_std=std,
        offset_min=float(offsets.min()),
        offset_max=float(offsets.max()),
        peak_std=peak_std,
        slope=slope,
        intercept=intercept,
        residual_std=residual_std,
        verdict=verdict,
    )


def fits_bbox(footprint_xy: np.ndarray, canvas_m: float) -> bool:
    """Axis-aligned footprint extent fits inside a square canvas."""
    points = _as_xy(footprint_xy)
    extent = points.max(axis=0) - points.min(axis=0)
    return bool((extent <= canvas_m + 1e-9).all())


def anchored_coverage(footprint_xy: np.ndarray, pose_xy: Sequence[float], canvas_m: float) -> bool:
    """A square canvas centred on ``pose_xy`` (heading = room axes) covers the footprint.

    The canvas is convex, so checking the footprint vertices is sufficient.
    The anchor is an observed pose, never the footprint's own bounds.
    """
    points = _as_xy(footprint_xy)
    half = canvas_m / 2.0
    relative = np.abs(points - np.asarray(pose_xy, dtype=np.float64).reshape(1, 2))
    return bool((relative <= half + 1e-9).all())


def anchored_coverage_counts(
    footprint_xy: np.ndarray,
    poses_xy: np.ndarray,
    canvas_m: float,
) -> int:
    """Number of candidate poses whose anchored canvas covers the whole footprint."""
    poses = _as_xy(poses_xy)
    return sum(anchored_coverage(footprint_xy, pose, canvas_m) for pose in poses)


def _as_xy(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise ValueError("expected a non-empty Nx2 array")
    return array
