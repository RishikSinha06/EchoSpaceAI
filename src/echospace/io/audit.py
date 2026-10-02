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


def first_strong_peak(waveform: np.ndarray, relative_threshold: float = 0.5) -> int:
    """Index of the crest of the first peak reaching a fraction of the maximum.

    The first sample crossing ``relative_threshold * max|x|`` is found, then
    followed uphill while the magnitude keeps rising. Searching a fixed window
    for its maximum instead would jump to a stronger reflection that arrives a
    few samples after the direct sound, which happens whenever a microphone or
    source sits close to a surface.
    """
    magnitude = np.abs(np.asarray(waveform, dtype=np.float64).ravel())
    if magnitude.size == 0 or not np.isfinite(magnitude).all():
        raise ValueError("waveform must be non-empty and finite")
    peak = float(magnitude.max())
    if peak <= 0:
        raise ValueError("waveform is silent")
    index = int(np.argmax(magnitude >= relative_threshold * peak))
    while index + 1 < magnitude.size and magnitude[index + 1] > magnitude[index]:
        index += 1
    return index


@dataclass(frozen=True)
class TimingSummary:
    """How measured peak positions relate to predicted direct-path delays.

    ``offset = peak - direct``. A constant offset means a fixed onset was
    trimmed (zero: nothing was trimmed). Peaks that do not move with distance
    mean each RIR was trimmed at its own arrival, so absolute delay carries no
    geometry. Anything else points to a coordinate, unit or frame error.

    ``offset_std`` is the raw spread the audit must report. The verdict uses
    the inlier fraction around the median instead, because one occluded pair
    whose strongest early arrival is a reflection would otherwise condemn a
    correct room. ``n_early`` counts arrivals before the direct path could
    physically reach the receiver; occlusion can never produce those.
    """

    n_pairs: int
    offset_mean: float
    offset_std: float
    offset_median: float
    offset_min: float
    offset_max: float
    inlier_fraction: float
    n_early: int
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
    min_inlier_fraction: float = 0.95,
) -> TimingSummary:
    """Classify the timing convention across many pairs.

    Verdicts: ``constant_offset`` (at least ``min_inlier_fraction`` of pairs
    lie within the tolerance of the median offset), ``per_pair_onset_trim``
    (peak fixed regardless of distance), ``inconsistent`` (likely
    coordinate/unit error; must not be patched per pair) and ``insufficient``
    (fewer than 2 pairs).
    """
    direct = np.asarray(list(direct_samples), dtype=np.float64)
    peaks = np.asarray(list(peak_samples), dtype=np.float64)
    if direct.shape != peaks.shape:
        raise ValueError("direct and peak arrays must match")
    offsets = peaks - direct
    n = int(direct.size)
    nan = float("nan")
    if n == 0:
        return TimingSummary(0, nan, nan, nan, nan, nan, nan, 0, nan, nan, nan, nan, "insufficient")
    std = float(offsets.std(ddof=1)) if n > 1 else 0.0
    peak_std = float(peaks.std(ddof=1)) if n > 1 else 0.0
    median = float(np.median(offsets))
    inlier_fraction = float((np.abs(offsets - median) <= constant_tolerance_samples).mean())
    if n > 1 and float(direct.std()) > 0:
        slope, intercept = (float(v) for v in np.polyfit(direct, peaks, 1))
        residual_std = float((peaks - (slope * direct + intercept)).std(ddof=1))
    else:
        slope, intercept, residual_std = nan, float(offsets.mean()), nan

    if n < 2:
        verdict = "insufficient"
    elif peak_std <= constant_tolerance_samples:
        # Peaks do not move with distance at all. A unit error (e.g. cm read
        # as m) also gives a near-zero slope, but its peaks still spread.
        verdict = "per_pair_onset_trim"
    elif inlier_fraction >= min_inlier_fraction:
        verdict = "constant_offset"
    else:
        verdict = "inconsistent"
    return TimingSummary(
        n_pairs=n,
        offset_mean=float(offsets.mean()),
        offset_std=std,
        offset_median=median,
        offset_min=float(offsets.min()),
        offset_max=float(offsets.max()),
        inlier_fraction=inlier_fraction,
        n_early=int((offsets < -constant_tolerance_samples).sum()),
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


# First-reflection check (plan, Stage 0 gate). Fixed before the check was run
# on real data: a +-2 sample window, 16 control offsets, a 90th-percentile
# bar, so a hit happens by chance about 10 % of the time.
REFLECTION_WINDOW_SAMPLES = 2
REFLECTION_MIN_GAP_SAMPLES = 5
REFLECTION_CONTROL_SHIFTS = tuple(s for k in range(8, 40, 4) for s in (-k, k))
REFLECTION_CONTROL_QUANTILE = 0.9


def mirror_across_plane(point: Sequence[float], plane_point: Sequence[float], plane_normal: Sequence[float]) -> np.ndarray:
    """Image of ``point`` in the plane through ``plane_point`` with normal ``plane_normal``."""
    p = np.asarray(point, dtype=np.float64)
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return p - 2.0 * float(np.dot(p - np.asarray(plane_point, dtype=np.float64), n)) * n


def window_peak(waveform: np.ndarray, centre: float, half_width: int = REFLECTION_WINDOW_SAMPLES) -> float | None:
    """Largest magnitude within ``half_width`` samples of ``centre``; None if off the end."""
    start, stop = int(round(centre)) - half_width, int(round(centre)) + half_width + 1
    if start < 0 or stop > len(waveform):
        return None
    return float(np.abs(waveform[start:stop]).max())


def reflection_hit(waveform: np.ndarray, reflection_samples: float, direct_samples: float) -> bool | None:
    """Is there a peak at the predicted reflection time, stronger than at control times?

    Returns None when the pair cannot be judged: the reflection would merge
    with the direct sound, or too few control windows fit away from it.
    Control windows are the same window shifted by ``REFLECTION_CONTROL_SHIFTS``,
    skipping any that come near the direct sound.
    """
    if reflection_samples - direct_samples < REFLECTION_MIN_GAP_SAMPLES:
        return None
    value = window_peak(waveform, reflection_samples)
    if value is None:
        return None
    controls = []
    for shift in REFLECTION_CONTROL_SHIFTS:
        centre = reflection_samples + shift
        if abs(centre - direct_samples) < REFLECTION_MIN_GAP_SAMPLES + REFLECTION_WINDOW_SAMPLES:
            continue
        control = window_peak(waveform, centre)
        if control is not None:
            controls.append(control)
    if len(controls) < len(REFLECTION_CONTROL_SHIFTS) // 2:
        return None
    return bool(value > np.quantile(controls, REFLECTION_CONTROL_QUANTILE))


def wall_specular_point(
    source_xyz: Sequence[float],
    receiver_xyz: Sequence[float],
    wall_start_xy: Sequence[float],
    wall_end_xy: Sequence[float],
    horizontal: Sequence[int] = (0, 1),
) -> tuple[float, float] | None:
    """Where the first-order reflection off a vertical wall hits it.

    Returns ``(t, height)``: ``t`` in [0, 1] along the wall segment and the
    height of the hit on the up axis, or None if the reflection path misses the
    segment or the two points sit on opposite sides of the wall.
    """
    h0, h1 = horizontal
    up = 3 - h0 - h1
    s, r = np.asarray(source_xyz, float), np.asarray(receiver_xyz, float)
    a, b = np.asarray(wall_start_xy, float), np.asarray(wall_end_xy, float)
    direction = b - a
    normal = np.array([-direction[1], direction[0]])
    side_s, side_r = float(np.dot(s[[h0, h1]] - a, normal)), float(np.dot(r[[h0, h1]] - a, normal))
    if side_s * side_r <= 0:
        return None
    image = s[[h0, h1]] - 2.0 * side_s / float(np.dot(normal, normal)) * normal
    path = r[[h0, h1]] - image
    matrix = np.array([direction, -path]).T
    if abs(np.linalg.det(matrix)) < 1e-12:
        return None
    t, u = np.linalg.solve(matrix, image - a)
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None
    return float(t), float(s[up] + (r[up] - s[up]) * u)
