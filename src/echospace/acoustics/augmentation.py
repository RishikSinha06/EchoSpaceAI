"""Seeded training-only gain, noise, microphone filtering and pose perturbation."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

from echospace.geometry.frames import GridFrame

from .bundles import AcousticBundle, poses_in_observed_free
from .rir import AcousticError


@dataclass(frozen=True)
class AugmentConfig:
    gain_db: tuple[float, float] = (-3.0, 3.0)
    snr_db: tuple[float, float] | None = (25.0, 40.0)
    lowpass_hz: tuple[float, float] | None = (3000.0, 7000.0)
    pose_std_m: float = 0.02
    pose_attempts: int = 8

    def __post_init__(self) -> None:
        if self.gain_db is None:
            raise AcousticError("gain_db must be a range")
        for name in ("gain_db", "snr_db", "lowpass_hz"):
            value = getattr(self, name)
            if value is not None and (len(value) != 2 or not np.isfinite(value).all() or value[0] > value[1]):
                raise AcousticError(f"invalid {name} range")
        if self.lowpass_hz is not None and self.lowpass_hz[0] <= 0:
            raise AcousticError("lowpass cutoff must be positive")
        if not np.isfinite(self.pose_std_m) or self.pose_std_m < 0 or type(self.pose_attempts) is not int or self.pose_attempts < 1:
            raise AcousticError("invalid pose augmentation settings")


def augment_bundle(
    bundle: AcousticBundle, grid: GridFrame, observed_free: np.ndarray, *,
    split: str, seed: int, sample_key: str, config: AugmentConfig = AugmentConfig(),
    sample_rate_hz: int = 16000,
) -> AcousticBundle:
    """Return a copy; val/test are exact no-ops and invalid rows stay zero.

    A separate stream per RIR ID makes K prefixes and row permutations agree.
    Horizontal pose jitter never leaves observed-free cells; after bounded
    retries the original pose is retained. Height is kept unchanged.
    """
    if split not in ("train", "val", "test"):
        raise AcousticError("augmentation requires an assigned train/val/test split")
    if type(seed) is not int or seed < 0 or type(sample_rate_hz) is not int or sample_rate_hz <= 0:
        raise AcousticError("seed and sample rate must be valid")
    result = bundle.prefix(len(bundle.rir_ids))
    if split != "train":
        return result
    if config.lowpass_hz is not None and config.lowpass_hz[1] >= sample_rate_hz / 2:
        raise AcousticError("lowpass cutoff must be below Nyquist")
    provenance = list(copy.deepcopy(result.provenance))
    for i, rir_id in enumerate(result.rir_ids):
        if not result.valid[i]:
            continue
        if result.provenance[i].get("config", {}).get("sample_rate_hz") != sample_rate_hz:
            raise AcousticError("augmentation sample rate differs from waveform provenance")
        if not poses_in_observed_free(result.positions_scene_m[i], grid, observed_free):
            raise AcousticError("input bundle pose is not in observed free space")
        identity = json.dumps([seed, sample_key, rir_id]).encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity).digest()[:8], "little"))
        gain_db = float(rng.uniform(*config.gain_db))
        waveform = result.waveforms[i].astype(np.float64) * 10 ** (gain_db / 20)
        cutoff = None
        if config.lowpass_hz is not None:
            cutoff = float(rng.uniform(*config.lowpass_hz))
            waveform = sosfilt(butter(2, cutoff, fs=sample_rate_hz, output="sos"), waveform)
        snr = None
        if config.snr_db is not None:
            snr = float(rng.uniform(*config.snr_db))
            noise = rng.normal(size=len(waveform))
            rms = float(np.sqrt(np.mean(waveform**2)))
            noise *= rms / (10 ** (snr / 20) * np.sqrt(np.mean(noise**2)))
            waveform += noise
        result.waveforms[i] = waveform.astype(np.float32)
        if not np.isfinite(result.waveforms[i]).all():
            raise AcousticError("augmentation overflows float32")
        original = result.positions_scene_m[i].copy()
        accepted = config.pose_std_m == 0
        if config.pose_std_m:
            for _ in range(config.pose_attempts):
                proposal = original.copy()
                proposal[:, [0, 2]] += rng.normal(0, config.pose_std_m, size=(2, 2))
                if poses_in_observed_free(proposal, grid, observed_free):
                    result.positions_scene_m[i] = proposal
                    accepted = True
                    break
        provenance[i]["augmentation"] = {
            "split": "train", "seed": seed, "sample_key": sample_key,
            "gain_db": gain_db, "snr_db": snr, "causal_lowpass_hz": cutoff,
            "filter_order": 2 if cutoff else None,
            "pose_delta_scene_m": (result.positions_scene_m[i] - original).tolist(),
            "pose_jitter_accepted": accepted,
            "note": "causal filter changes phase; pose jitter is input uncertainty, not resimulation",
        }
    return AcousticBundle(result.rir_ids, result.waveforms, result.positions_scene_m, result.valid,
                          tuple(provenance), result.selection)
