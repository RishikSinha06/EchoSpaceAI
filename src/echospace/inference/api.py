"""Local EchoSpace inference API (FastAPI).

    uvicorn echospace.inference.api:app --host 127.0.0.1 --port 8000

Serves only registered, hash-verified checkpoints on committed, hash-verified
evaluation samples. By default it refuses samples from rooms the model saw in
training or validation, so the viewer cannot present a memorised room as a
reconstruction. Results live in memory, keyed by a deterministic id.

Configuration (environment variables, paths relative to the repository):
``ECHOSPACE_MODELS`` (configs/models.json), ``ECHOSPACE_EVAL_DIR``
(data/processed/eval_samples), ``ECHOSPACE_EVAL_SPEC``
(data/splits/eval_masks.json), ``ECHOSPACE_FOLDS`` (data/splits/folds.json),
``ECHOSPACE_DEVICE`` (cpu).
"""

from __future__ import annotations

import io
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from echospace.contract_v0 import SCHEMA_VERSION
from echospace.inference import export
from echospace.inference.predict import INFERENCE_VERSION, Reconstruction, code_revision, reconstruct
from echospace.inference.registry import ModelRegistry, RegistryError
from echospace.inference.samples import K_MAX, SampleError, SampleStore

REQUEST_SCHEMA = "echospace.reconstruction_request.v1"
REPO = Path(__file__).resolve().parents[3]
MAX_STORED = 128


@dataclass(frozen=True)
class Settings:
    repo_root: Path = REPO
    models: str = "configs/models.json"
    eval_dir: str = "data/processed/eval_samples"
    eval_spec: str = "data/splits/eval_masks.json"
    folds: str = "data/splits/folds.json"
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            models=os.environ.get("ECHOSPACE_MODELS", cls.models),
            eval_dir=os.environ.get("ECHOSPACE_EVAL_DIR", cls.eval_dir),
            eval_spec=os.environ.get("ECHOSPACE_EVAL_SPEC", cls.eval_spec),
            folds=os.environ.get("ECHOSPACE_FOLDS", cls.folds),
            device=os.environ.get("ECHOSPACE_DEVICE", cls.device),
        )

    def path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repo_root / path


class ReconstructionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    schema_version: Literal["echospace.reconstruction_request.v1"]
    model_id: str = Field(min_length=1, max_length=128)
    sample_id: str = Field(min_length=1, max_length=256)
    k: int = Field(default=K_MAX, ge=1, le=K_MAX)
    allow_seen_rooms: bool = False


def _summary(result: Reconstruction) -> dict[str, Any]:
    base = f"/v1/reconstructions/{result.reconstruction_id}"
    sample = result.provenance["sample"]
    return {
        "reconstruction_id": result.reconstruction_id,
        "model_id": result.model.model_id,
        "sample_id": result.sample.sample_id,
        "room_id": sample["room_id"],
        "split_for_model_fold": sample["split_for_model_fold"],
        "occlusion_type": sample["occlusion_type"],
        "coverage": sample["coverage"],
        "k_requested": result.k_requested,
        "k_used": result.k_used,
        "metrics": result.metrics,
        "links": {
            "self": base,
            "grid": f"{base}/grid.json",
            "glb": f"{base}/scene.glb",
            "provenance": f"{base}/provenance.json",
            "prediction": f"{base}/prediction.npz",
        },
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = SampleStore(settings.path(settings.eval_dir), settings.path(settings.eval_spec), settings.path(settings.folds))
    registry = ModelRegistry(settings.path(settings.models), settings.repo_root, settings.path(settings.folds), settings.path(settings.eval_spec))
    revision = code_revision(settings.repo_root)
    results: OrderedDict[str, Reconstruction] = OrderedDict()
    lock = threading.Lock()

    app = FastAPI(title="EchoSpace inference", version=INFERENCE_VERSION,
                  description="2D room footprint completion from a partial scan and RIRs. Local research service.")

    def stored(reconstruction_id: str) -> Reconstruction:
        with lock:
            result = results.get(reconstruction_id)
        if result is None:
            raise HTTPException(404, f"unknown reconstruction_id {reconstruction_id!r}")
        return result

    @app.get("/health")
    def health() -> dict[str, Any]:
        servable = [model_id for model_id in registry.entries if registry.problem(model_id) is None]
        return {
            "status": "ok" if servable else "degraded",
            "inference_version": INFERENCE_VERSION,
            "contract_schema_version": SCHEMA_VERSION,
            "code_revision": revision,
            "device": settings.device,
            "models_registered": len(registry.entries),
            "models_servable": len(servable),
            "samples_in_spec": len(store.spec_samples()),
            "samples_prepared": len(store.available()),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        rows = []
        for model_id, entry in registry.entries.items():
            problem = registry.problem(model_id)
            rows.append(entry.public() | {"available": problem is None, "unavailable_reason": problem})
        return {"models": rows}

    @app.get("/v1/samples")
    def samples(
        model_id: str | None = None,
        split: Literal["train", "val", "test", "any"] = "test",
        room_id: str | None = None,
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> dict[str, Any]:
        fold = registry.get(model_id).fold if model_id else next(iter(registry.entries.values())).fold
        rows = []
        for entry in store.available():
            room_split = store.split_of(entry["room_id"], fold)
            if split != "any" and room_split != split or room_id and entry["room_id"] != room_id:
                continue
            rows.append({key: entry[key] for key in ("sample_id", "room_id", "occlusion_type", "coverage_bin", "coverage")}
                        | {"split_for_fold": room_split})
        return {"fold": fold, "count": len(rows), "samples": rows[:limit]}

    @app.post("/v1/reconstructions", status_code=201)
    def create(request: ReconstructionRequest) -> dict[str, Any]:
        try:
            entry = registry.get(request.model_id)
        except RegistryError as exc:
            raise HTTPException(404, str(exc)) from exc
        try:
            sample = store.load(request.sample_id)
        except SampleError as exc:
            status = 404 if "unknown" in str(exc) or "not prepared" in str(exc) else 422
            raise HTTPException(status, str(exc)) from exc
        split = store.split_of(sample.record["room_id"], entry.fold)
        if split in ("train", "val") and not request.allow_seen_rooms:
            raise HTTPException(409, f"room {sample.record['room_id']} is in the {split} partition of fold {entry.fold} "
                                     f"for {entry.model_id}; set allow_seen_rooms to reconstruct it anyway")
        try:
            model = registry.load(entry.model_id)
        except RegistryError as exc:
            raise HTTPException(503, str(exc)) from exc
        result = reconstruct(model, entry, sample, request.k, split=split, revision=revision, device=settings.device)
        with lock:
            results[result.reconstruction_id] = result
            results.move_to_end(result.reconstruction_id)
            while len(results) > MAX_STORED:
                results.popitem(last=False)
        return _summary(result)

    @app.get("/v1/reconstructions/{reconstruction_id}")
    def get(reconstruction_id: str) -> dict[str, Any]:
        result = stored(reconstruction_id)
        return _summary(result) | {"provenance": result.provenance}

    @app.get("/v1/reconstructions/{reconstruction_id}/grid.json")
    def grid(reconstruction_id: str, truth: bool = True) -> JSONResponse:
        document = export.grid_document(stored(reconstruction_id), include_truth=truth)
        return JSONResponse(document, headers={"Content-Disposition": f'attachment; filename="{reconstruction_id}.grid.json"'})

    @app.get("/v1/reconstructions/{reconstruction_id}/scene.glb")
    def glb(reconstruction_id: str) -> Response:
        result = stored(reconstruction_id)
        data = export.glb_bytes(export.grid_document(result, include_truth=False), result.provenance)
        return Response(data, media_type="model/gltf-binary",
                        headers={"Content-Disposition": f'attachment; filename="{reconstruction_id}.glb"'})

    @app.get("/v1/reconstructions/{reconstruction_id}/provenance.json")
    def provenance(reconstruction_id: str) -> JSONResponse:
        return JSONResponse(stored(reconstruction_id).provenance,
                            headers={"Content-Disposition": f'attachment; filename="{reconstruction_id}.provenance.json"'})

    @app.get("/v1/reconstructions/{reconstruction_id}/prediction.npz")
    def prediction(reconstruction_id: str) -> Response:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **stored(reconstruction_id).prediction_arrays())
        return Response(buffer.getvalue(), media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{reconstruction_id}.prediction.npz"'})

    app.state.registry, app.state.store, app.state.settings = registry, store, settings
    return app


def __getattr__(name: str) -> Any:
    """``echospace.inference.api:app`` builds the app lazily so importing this module has no side effects."""
    if name == "app":
        return create_app()
    raise AttributeError(name)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("ECHOSPACE_PORT", "8000")))
