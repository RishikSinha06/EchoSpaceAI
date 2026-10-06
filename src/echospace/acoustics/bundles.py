"""Audited pair loading and nested RIR selection on P4's observed-free mask.

P6 owns dataset assembly and chooses K. This module supplies the same ranked
prefix for K=1/2/4/8 without using audio amplitudes or hidden target geometry.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from echospace.geometry.frames import GridFrame, source_to_scene
from echospace.io.adapters import acousticrooms as ar
from echospace.io.audit_geometry import geometry_signature

from .cache import RirCache
from .rir import AcousticError, ProcessedRir, RirConfig, TimingEvidence, preprocess_wav


@dataclass(frozen=True)
class RirCandidate:
    room_id: str
    pair: tuple[int, int]
    positions_scene_m: np.ndarray
    rir_ref: ar.MemberRef
    metadata_ref: ar.MemberRef
    metadata_sha256: str
    audit_sha256: str
    ir_norm: Any = None
    verified: bool = False

    @property
    def rir_id(self) -> str:
        return f"{self.room_id}/S{self.pair[0]}_R{self.pair[1]}"


def load_audited_candidates(
    reader: ar.ArchiveReader, room: ar.RoomEntry, audit: dict[str, Any],
) -> tuple[RirCandidate, ...]:
    """Require a matching accepted D0 record and mesh before admitting pairs.

    The attributed integration subset is permitted; it need not contain every
    pair in the release. Conflicting positions for one source/receiver ID are
    refused. The audit is sampled evidence, not a per-waveform timing proof.
    """
    if (audit.get("dataset") != "AcousticRooms" or audit.get("room_id") != room.room_id
            or audit.get("accepted") is not True or audit.get("up_axis") != 2
            or audit.get("units") != "metres" or audit.get("position_conflicts") != 0
            or audit.get("timing", {}).get("verdict") != "constant_offset" or room.mesh is None):
        raise AcousticError("room needs a matching accepted AcousticRooms D0 audit")
    mesh = ar.load_mesh(reader, room.mesh)
    try:
        expected_bounds = np.asarray([audit.get("mesh", {}).get("bounds_min"),
                                      audit.get("mesh", {}).get("bounds_max")], dtype=float)
    except (ValueError, TypeError) as exc:
        raise AcousticError("audit mesh bounds are malformed") from exc
    # Git may check out the attributed OBJ text as CRLF. Only that byte-level
    # difference is allowed; geometry signature and absolute bounds still match.
    mesh_bytes = reader.read(room.mesh)
    if (geometry_signature(mesh.vertices) != audit.get("geometry_signature")
            or expected_bounds.shape != (2, 3) or not np.isfinite(expected_bounds).all()
            or not np.allclose(mesh.bounds, expected_bounds, rtol=0, atol=1e-6)
            or audit.get("mesh_bytes") not in (len(mesh_bytes), len(mesh_bytes.replace(b"\r\n", b"\n")))):
        raise AcousticError("mesh does not match the D0 geometry signature")
    audit_sha = hashlib.sha256(json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    candidates = []
    sources: dict[int, tuple[float, float, float]] = {}
    receivers: dict[int, tuple[float, float, float]] = {}
    for pair in room.paired():
        raw = reader.read(room.metadata[pair])
        metadata = ar.load_pair_metadata(reader, room.metadata[pair])
        positions = np.asarray([metadata.source_xyz, metadata.receiver_xyz], dtype=np.float64)
        if not np.isfinite(positions).all():
            raise AcousticError("metadata poses must be finite")
        for table, number, point in ((sources, pair[0], metadata.source_xyz), (receivers, pair[1], metadata.receiver_xyz)):
            if number in table and table[number] != point:
                raise AcousticError("one pair ID maps to conflicting poses")
            table[number] = point
        candidates.append(RirCandidate(
            room.room_id, pair, source_to_scene(positions).astype(np.float32), room.rirs[pair],
            room.metadata[pair], hashlib.sha256(raw).hexdigest(), audit_sha,
            ir_norm=json.loads(raw).get("IR_norm"), verified=True,
        ))
    return tuple(candidates)


def poses_in_observed_free(positions: np.ndarray, grid: GridFrame, observed_free: np.ndarray) -> bool:
    """Use the contract's half-open cell projection for both source and mic."""
    positions = np.asarray(positions)
    if positions.shape != (2, 3) or not np.isfinite(positions).all():
        return False
    free = np.asarray(observed_free)
    if free.shape != (grid.height, grid.width) or not np.isin(free, [0, 1]).all():
        raise AcousticError("observed_free must be a binary mask on this grid")
    columns, rows = np.floor(grid.scene_to_cell(positions)).astype(int).T
    if np.any((rows < 0) | (rows >= grid.height) | (columns < 0) | (columns >= grid.width)):
        return False
    return bool(free[rows, columns].all())


def process_candidate(
    reader: ar.ArchiveReader, candidate: RirCandidate, config: RirConfig,
    cache: RirCache | None = None,
) -> ProcessedRir:
    if not candidate.verified:
        raise AcousticError("unverified RIR pair cannot be processed as training input")
    if config.alignment != "emission":
        raise AcousticError("D0 supplies emission timing; other alignments need separate per-pair evidence")
    timing = TimingEvidence(
        emission_time_verified=True, upstream_trimmed=False,
        upstream_normalization="IR_norm meaning unverified; no level correction",
        evidence=f"D0:{candidate.audit_sha256}",
    )
    wav_bytes = reader.read(candidate.rir_ref)
    processed = cache.get(wav_bytes, config, timing) if cache else preprocess_wav(wav_bytes, config, timing)
    if processed.metadata["native_sample_rate_hz"] != 22050:
        raise AcousticError("waveform rate differs from the audited 22050 Hz release")
    metadata = copy.deepcopy(processed.metadata)
    metadata.update({
        "rir_id": candidate.rir_id,
        "source_metadata_sha256": candidate.metadata_sha256,
        "audit_record_sha256": candidate.audit_sha256,
        "IR_norm_uninterpreted": candidate.ir_norm,
        "positions_scene_m": candidate.positions_scene_m.tolist(),
    })
    return ProcessedRir(processed.waveform, metadata)


@dataclass(frozen=True)
class AcousticBundle:
    rir_ids: tuple[str, ...]
    waveforms: np.ndarray  # K,T float32
    positions_scene_m: np.ndarray  # K,2,3 float32
    valid: np.ndarray  # K uint8
    provenance: tuple[dict[str, Any], ...]
    selection: dict[str, Any]

    def contract_arrays(self) -> dict[str, np.ndarray]:
        return {"rir_waveforms": self.waveforms.copy(), "rir_positions_scene_m": self.positions_scene_m.copy(),
                "rir_valid": self.valid.copy()}

    def prefix(self, k: int) -> AcousticBundle:
        if type(k) is not int or not 1 <= k <= len(self.rir_ids):
            raise AcousticError("prefix K must be within the bundle")
        return AcousticBundle(self.rir_ids[:k], self.waveforms[:k].copy(), self.positions_scene_m[:k].copy(),
                              self.valid[:k].copy(), copy.deepcopy(self.provenance[:k]), copy.deepcopy(self.selection))


def build_bundle(
    candidates: Sequence[RirCandidate], grid: GridFrame, observed_free: np.ndarray,
    loader: Callable[[RirCandidate], ProcessedRir], *, room_id: str, selection_key: str,
    seed: int = 0, k: int = 8, config: RirConfig = RirConfig(), pad_missing: bool = False,
) -> AcousticBundle:
    """Rank by IDs/seed, filter poses, load only needed audio, return K rows.

    By default a missing K raises. Explicit padding adds zero waveforms and
    poses with ``valid=0``; P6 must still enforce its primary usable K gate.
    """
    if type(k) is not int or k < 1 or type(seed) is not int or seed < 0:
        raise AcousticError("K must be positive and seed nonnegative")
    free = np.asarray(observed_free)
    if free.shape != (grid.height, grid.width) or not np.isin(free, [0, 1]).all():
        raise AcousticError("observed_free must be a binary mask on this grid")
    rejected = []
    eligible = []
    ids = set()
    for candidate in candidates:
        if candidate.room_id != room_id or candidate.rir_id in ids:
            raise AcousticError("mixed rooms or duplicate RIR IDs in candidate list")
        ids.add(candidate.rir_id)
        reason = None
        if not candidate.verified:
            reason = "unverified_pair"
        elif not poses_in_observed_free(candidate.positions_scene_m, grid, observed_free):
            reason = "pose_not_in_observed_free"
        if reason:
            rejected.append({"rir_id": candidate.rir_id, "reason": reason})
        else:
            eligible.append(candidate)

    def rank(candidate: RirCandidate) -> bytes:
        return hashlib.sha256(json.dumps([seed, room_id, selection_key, candidate.rir_id]).encode()).digest()

    eligible.sort(key=lambda candidate: (rank(candidate), candidate.rir_id))
    waveforms = np.zeros((k, config.window_samples), dtype=np.float32)
    positions = np.zeros((k, 2, 3), dtype=np.float32)
    valid = np.zeros(k, dtype=np.uint8)
    chosen_ids: list[str] = []
    provenance: list[dict[str, Any]] = []
    for candidate in eligible:
        if len(chosen_ids) == k:
            break
        try:
            processed = loader(candidate)
            if (processed.waveform.shape != (config.window_samples,) or not np.isfinite(processed.waveform).all()
                    or not np.any(processed.waveform) or processed.metadata.get("config") != config.to_dict()):
                raise AcousticError("processed waveform does not match this bundle config")
        except AcousticError as exc:
            rejected.append({"rir_id": candidate.rir_id, "reason": str(exc)})
            continue
        i = len(chosen_ids)
        waveforms[i] = processed.waveform
        positions[i] = candidate.positions_scene_m
        valid[i] = 1
        chosen_ids.append(candidate.rir_id)
        provenance.append(copy.deepcopy(processed.metadata))
    if len(chosen_ids) < k and not pad_missing:
        raise AcousticError(f"need K={k} usable pairs; found {len(chosen_ids)}")
    for i in range(len(chosen_ids), k):
        chosen_ids.append(f"{room_id}/invalid_padding_{i}")
        provenance.append({"valid": False, "reason": "insufficient_usable_pairs"})
    selection = {"seed": seed, "selection_key": selection_key, "eligible_pose_pairs": len(eligible),
                 "rejected": rejected, "ranking": "sha256(seed,room,selection_key,rir_id); independent of K"}
    return AcousticBundle(tuple(chosen_ids), waveforms, positions, valid, tuple(provenance), selection)
