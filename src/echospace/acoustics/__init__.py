"""P5 acoustic preprocessing; requires the repository's existing audit extras."""

from .augmentation import AugmentConfig, augment_bundle
from .bundles import AcousticBundle, RirCandidate, build_bundle, load_audited_candidates, process_candidate
from .cache import RirCache
from .rir import AcousticError, ProcessedRir, RirConfig, TimingEvidence, preprocess_wav

__all__ = [
    "AcousticError", "AcousticBundle", "AugmentConfig", "ProcessedRir", "RirCache",
    "RirCandidate", "RirConfig", "TimingEvidence", "augment_bundle", "build_bundle",
    "load_audited_candidates", "preprocess_wav", "process_candidate",
]
