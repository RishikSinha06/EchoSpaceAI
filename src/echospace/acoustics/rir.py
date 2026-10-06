"""P5 mono WAV decoding and explicit, versioned RIR preprocessing."""

from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass
from math import gcd
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

PROCESSOR_VERSION = "p5.1"


class AcousticError(ValueError):
    """Audio or its timing evidence cannot support a valid acoustic input."""


@dataclass(frozen=True)
class RirConfig:
    sample_rate_hz: int = 16000
    window_ms: float = 80.0
    alignment: str = "emission"
    normalization: str = "preserve"

    def __post_init__(self) -> None:
        if type(self.sample_rate_hz) is not int or self.sample_rate_hz <= 0:
            raise AcousticError("sample_rate_hz must be a positive integer")
        if not np.isfinite(self.window_ms) or self.window_ms <= 0:
            raise AcousticError("window_ms must be positive and finite")
        if self.window_samples < 1:
            raise AcousticError("window must contain at least one sample")
        if self.alignment not in ("emission", "verified_direct", "relative_onset"):
            raise AcousticError("unknown alignment")
        if self.normalization not in ("preserve", "peak"):
            raise AcousticError("unknown normalization")

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.window_ms / 1000))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimingEvidence:
    """Timing claims must be supplied explicitly, rather than inferred from a peak.

    ``relative_onset`` needs a caller-supplied onset but makes no direct-path
    or absolute travel-delay claim. ``verified_direct`` requires separate
    evidence that the supplied onset really is the direct arrival.
    """

    emission_time_verified: bool = False
    onset_native_sample: int | None = None
    direct_onset_verified: bool = False
    upstream_trimmed: bool | None = None
    upstream_normalization: str = "unverified"
    evidence: str = "unverified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessedRir:
    waveform: np.ndarray  # finite float32 T
    metadata: dict[str, Any]


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<f4").tobytes()).hexdigest()


def preprocess_wav(wav_bytes: bytes, config: RirConfig, timing: TimingEvidence) -> ProcessedRir:
    """Decode mono PCM/float WAV, anti-alias resample, then crop/right-pad.

    PCM scaling is a format conversion, not peak normalization. No measured
    gain or absolute emission delay is removed by the default configuration.
    """
    try:
        native_rate, samples = wavfile.read(io.BytesIO(wav_bytes))
    except (ValueError, OSError, EOFError) as exc:
        raise AcousticError("unreadable WAV") from exc
    if samples.ndim != 1 or len(samples) == 0:
        raise AcousticError("RIR must be nonempty mono audio")
    if native_rate <= 0:
        raise AcousticError("invalid native sample rate")
    dtype = samples.dtype
    waveform = samples.astype(np.float64)
    if np.issubdtype(dtype, np.signedinteger):
        waveform /= float(-np.iinfo(dtype).min)
    elif dtype == np.dtype("uint8"):
        waveform = (waveform - 128.0) / 128.0
    elif not np.issubdtype(dtype, np.floating):
        raise AcousticError(f"unsupported WAV dtype {dtype}")
    if not np.isfinite(waveform).all() or not np.any(waveform):
        raise AcousticError("RIR is nonfinite or silent")

    start = 0
    if config.alignment == "emission":
        if not timing.emission_time_verified or timing.upstream_trimmed is not False:
            raise AcousticError("emission alignment requires verified, untrimmed timing")
    else:
        start = timing.onset_native_sample
        if type(start) is not int or not 0 <= start < len(waveform):
            raise AcousticError("alignment requires a valid native onset sample")
        if config.alignment == "verified_direct" and not timing.direct_onset_verified:
            raise AcousticError("direct alignment requires verified direct-onset evidence")
    factor = gcd(int(native_rate), config.sample_rate_hz)
    converted = resample_poly(waveform[start:], config.sample_rate_hz // factor, int(native_rate) // factor)
    count = min(config.window_samples, len(converted))
    output = np.zeros(config.window_samples, dtype=np.float64)
    output[:count] = converted[:count]
    peak = float(np.max(np.abs(output)))
    if peak <= 0:
        raise AcousticError("selected RIR window is silent")
    scale = peak if config.normalization == "peak" else 1.0
    output = (output / scale).astype(np.float32)
    if not np.isfinite(output).all():
        raise AcousticError("processed RIR overflows float32")
    metadata = {
        "processor_version": PROCESSOR_VERSION,
        "source_wav_sha256": hashlib.sha256(wav_bytes).hexdigest(),
        "native_sample_rate_hz": int(native_rate),
        "native_dtype": str(dtype),
        "native_length_samples": len(samples),
        "native_peak_abs": float(np.max(np.abs(waveform))),
        "config": config.to_dict(),
        "timing": timing.to_dict(),
        "window_start_native_sample": start,
        "window_start_seconds": start / int(native_rate),
        "valid_output_samples": count,
        "right_padding_samples": config.window_samples - count,
        "normalization_divisor": scale,
        "output_sha256": array_sha256(output),
    }
    return ProcessedRir(output, metadata)
