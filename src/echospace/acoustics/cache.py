"""Content-addressed clean waveform cache; augmented audio is never written here."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .rir import PROCESSOR_VERSION, AcousticError, ProcessedRir, RirConfig, TimingEvidence, array_sha256, preprocess_wav


class RirCache:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    @staticmethod
    def key(wav_bytes: bytes, config: RirConfig, timing: TimingEvidence) -> str:
        identity = {
            "processor_version": PROCESSOR_VERSION,
            "source_wav_sha256": hashlib.sha256(wav_bytes).hexdigest(),
            "config": config.to_dict(),
            "timing": timing.to_dict(),
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def get(self, wav_bytes: bytes, config: RirConfig, timing: TimingEvidence) -> ProcessedRir:
        key = self.key(wav_bytes, config, timing)
        target = self.directory / f"{key}.npz"
        if target.exists():
            try:
                with np.load(target, allow_pickle=False) as data:
                    waveform = data["waveform"].copy()
                    metadata = json.loads(data["metadata_utf8"].tobytes().decode("utf-8"))
                if (waveform.dtype != np.float32 or waveform.shape != (config.window_samples,)
                        or not np.isfinite(waveform).all() or metadata["cache_key"] != key
                        or metadata["output_sha256"] != array_sha256(waveform)
                        or metadata["processor_version"] != PROCESSOR_VERSION
                        or metadata["config"] != config.to_dict() or metadata["timing"] != timing.to_dict()
                        or metadata["source_wav_sha256"] != hashlib.sha256(wav_bytes).hexdigest()):
                    raise ValueError("cache integrity mismatch")
                return ProcessedRir(waveform, metadata)
            except (OSError, ValueError, KeyError, UnicodeError) as exc:
                raise AcousticError(f"corrupt RIR cache entry {key}") from exc
        result = preprocess_wav(wav_bytes, config, timing)
        result.metadata["cache_key"] = key
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".npz", delete=False) as handle:
                temporary = handle.name
                meta = json.dumps(result.metadata, sort_keys=True, separators=(",", ":")).encode()
                np.savez_compressed(handle, waveform=result.waveform, metadata_utf8=np.frombuffer(meta, dtype=np.uint8))
            os.replace(temporary, target)
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()
        return result
