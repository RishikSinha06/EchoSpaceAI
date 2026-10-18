"""Versioned model registry: which checkpoints may be served, and proof they are the right ones.

``configs/models.json`` is committed; the weights are not. A model is served
only when its file exists, its SHA-256 equals the registered value, its
checkpoint format and architecture match the entry, and the folds file and
evaluation spec on disk are the exact versions it was trained and scored
against. Anything else is reported as unavailable with the reason, never
loaded "close enough".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_FORMAT = "echospace.models.1"


class RegistryError(LookupError):
    """Unknown model, or a registered model that cannot be served."""


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    family: str
    variant: str
    checkpoint: str
    checkpoint_sha256: str
    checkpoint_format: str
    fold: int
    seed: int
    threshold: float
    folds_file_sha256: str
    eval_spec_checksum: str
    raw: dict[str, Any] = field(repr=False, compare=False)

    def public(self) -> dict[str, Any]:
        """Registry fields safe to show a client (no local paths)."""
        return {key: value for key, value in self.raw.items() if key not in ("checkpoint", "obtain")}


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: str | Path) -> dict[str, ModelEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != REGISTRY_FORMAT:
        raise RegistryError(f"{path}: expected registry format {REGISTRY_FORMAT}")
    entries: dict[str, ModelEntry] = {}
    for raw in payload["models"]:
        entry = ModelEntry(
            model_id=raw["model_id"], family=raw["family"], variant=raw["variant"], checkpoint=raw["checkpoint"],
            checkpoint_sha256=raw["checkpoint_sha256"], checkpoint_format=raw["checkpoint_format"],
            fold=int(raw["fold"]), seed=int(raw["seed"]), threshold=float(raw["threshold"]),
            folds_file_sha256=raw["folds_file_sha256"], eval_spec_checksum=raw["eval_spec_checksum"], raw=raw,
        )
        if Path(entry.checkpoint).is_absolute() or ".." in Path(entry.checkpoint).parts:
            raise RegistryError(f"{entry.model_id}: checkpoint path must be relative to the repository")
        if entry.model_id in entries:
            raise RegistryError(f"duplicate model_id {entry.model_id}")
        entries[entry.model_id] = entry
    return entries


class ModelRegistry:
    """Loads registered checkpoints on demand and caches the verified models."""

    def __init__(self, registry_path: str | Path, repo_root: str | Path, folds_path: str | Path, spec_path: str | Path) -> None:
        self.root = Path(repo_root)
        self.entries = load_registry(registry_path)
        self.folds_sha256 = file_sha256(Path(folds_path))
        self.spec_checksum = json.loads(Path(spec_path).read_text(encoding="utf-8"))["checksum"]
        self._models: dict[str, Any] = {}
        self._problems: dict[str, str | None] = {}

    def get(self, model_id: str) -> ModelEntry:
        entry = self.entries.get(model_id)
        if entry is None:
            raise RegistryError(f"unknown model_id {model_id!r}")
        return entry

    def problem(self, model_id: str) -> str | None:
        """Why a registered model cannot be served, or ``None`` when it can (checks are cached)."""
        if model_id not in self._problems:
            entry = self.get(model_id)
            path = self.root / entry.checkpoint
            if entry.folds_file_sha256 != self.folds_sha256:
                reason = "data/splits/folds.json differs from the split this model was trained on"
            elif entry.eval_spec_checksum != self.spec_checksum:
                reason = "data/splits/eval_masks.json differs from the spec this model was scored on"
            elif not path.is_file():
                reason = f"checkpoint file missing ({entry.raw.get('obtain', 'see configs/models.json')})"
            elif file_sha256(path) != entry.checkpoint_sha256:
                reason = "checkpoint SHA-256 does not match the registry"
            else:
                reason = None
            self._problems[model_id] = reason
        return self._problems[model_id]

    def load(self, model_id: str) -> Any:
        """The verified model in eval mode on CPU; raises with the reason if it cannot be served."""
        if model_id in self._models:
            return self._models[model_id]
        entry = self.get(model_id)
        reason = self.problem(model_id)
        if reason is not None:
            raise RegistryError(f"{model_id} unavailable: {reason}")
        from echospace.training.fusion_runner import load_fusion_checkpoint

        model, payload = load_fusion_checkpoint(self.root / entry.checkpoint)
        spec = payload["model"]
        if payload.get("format") != entry.checkpoint_format or spec.get("variant") != entry.variant or spec.get("family") != entry.family:
            raise RegistryError(f"{model_id}: checkpoint is {spec.get('family')} {spec.get('variant')}, registry says {entry.family} {entry.variant}")
        if entry.raw.get("frozen_checksum") and payload.get("frozen_checksum") != entry.raw["frozen_checksum"]:
            raise RegistryError(f"{model_id}: checkpoint belongs to another frozen dataset")
        model.eval()
        self._models[model_id] = model
        return model
