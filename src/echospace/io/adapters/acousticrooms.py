"""Read-only adapter for facebookresearch/AcousticRooms (CC BY 4.0).

Meshes, per-pair metadata and RIRs are indexed by ``(room_id, source_id,
receiver_id)`` and read straight out of the distributed archives, including
zips nested inside zips. Nothing is extracted to disk and the archives are
only ever opened for reading.

Two on-disk layouts are understood, because the dataset reaches people both
ways:

* the official repository checkout: ``room_mesh_obj_format/`` as a folder,
  ``metadata.zip`` and ``single_channel_ir.zip`` beside it;
* a cloud-drive export: one wrapper zip holding the small files (meshes,
  ``metadata.zip``, ``simulation_info/``) plus separate large zips.

Member roles are recognised from their path *inside* the dataset, never from
the name of the container they arrived in. Positions and meshes are returned
in the dataset's own frame and units: axis convention and scale are for the
audit to discover, not for this module to assume.

Source and receiver ids are compared as integers. The dataset pads them
differently in metadata names (``S009_R0023.json``) and RIR names
(``S009_R023_hybrid_IR.wav``), so string equality would report false orphans.
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:  # heavy optional dependency, imported lazily in load_mesh
    import trimesh

DATASET_NAME = "AcousticRooms"
ROOT_ENV_VAR = "ECHOSPACE_ACOUSTICROOMS_ROOT"
INDEX_VERSION = 1

_PAIR = r"S(?P<source>\d+)_R(?P<receiver>\d+)"
_ROOM = r"(?P<category>[^/]+)/(?P<room>[^/]+)"
_RIR_RE = re.compile(rf"(?:^|/){_ROOM}/{_PAIR}_hybrid_IR\.wav$")
_METADATA_RE = re.compile(rf"(?:^|/)metadata/{_ROOM}/{_PAIR}\.json$")
_MESH_RE = re.compile(rf"(?:^|/)room_mesh_obj_format/{_ROOM}\.obj$")
_SIMULATION_RE = re.compile(rf"(?:^|/)simulation_info/{_ROOM}/simulation(?:_info)?\.json$")
# Unprocessed 32 kHz subset: its room ids differ from the preprocessed set, so
# it must never be joined to the main index by name.
_EXCLUDED_RE = re.compile(r"(?:^|/)raw_dense_simulation/")

_LFS_MAGIC = b"version https://git-lfs.github.com/spec/"
_LFS_MAX_BYTES = 1024
# Deflated nested zips have to be inflated to memory to get random access.
_MAX_IN_MEMORY_ZIP_BYTES = 1 << 30

PairId = tuple[int, int]
Chain = tuple[str, ...]


class AcousticRoomsError(RuntimeError):
    """The dataset is missing, unreadable or not in a recognised layout."""


@dataclass(frozen=True)
class MemberRef:
    """Where one file lives: a path under the root, then nested zip members."""

    chain: Chain

    @property
    def display(self) -> str:
        return " :: ".join(self.chain)


@dataclass(frozen=True)
class PairMetadata:
    """Source and receiver positions exactly as stored (source frame, source units)."""

    source_xyz: tuple[float, float, float]
    receiver_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class Rir:
    """One impulse response at its native rate; integer PCM is scaled to [-1, 1]."""

    waveform: np.ndarray
    sample_rate_hz: int
    native_dtype: str


@dataclass
class RoomEntry:
    room_id: str
    category: str
    mesh: MemberRef | None = None
    simulation: MemberRef | None = None
    rirs: dict[PairId, MemberRef] = field(default_factory=dict)
    metadata: dict[PairId, MemberRef] = field(default_factory=dict)

    def paired(self) -> list[PairId]:
        """Pairs that have both an RIR and a metadata record, in sorted order."""
        return sorted(self.rirs.keys() & self.metadata.keys())


@dataclass
class AcousticRoomsIndex:
    fingerprint: list[list[Any]]
    rooms: dict[str, RoomEntry] = field(default_factory=dict)
    duplicates: list[str] = field(default_factory=list)
    lfs_pointers: list[str] = field(default_factory=list)
    skipped_archives: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        containers: dict[Chain, int] = {}

        def pack(ref: MemberRef | None) -> list[Any] | None:
            if ref is None:
                return None
            parent = ref.chain[:-1]
            return [containers.setdefault(parent, len(containers)), ref.chain[-1]]

        rooms = {
            room_id: {
                "category": room.category,
                "mesh": pack(room.mesh),
                "simulation": pack(room.simulation),
                "rirs": [[s, r, *pack(ref)] for (s, r), ref in sorted(room.rirs.items())],
                "metadata": [[s, r, *pack(ref)] for (s, r), ref in sorted(room.metadata.items())],
            }
            for room_id, room in sorted(self.rooms.items())
        }
        return {
            "index_version": INDEX_VERSION,
            "dataset": DATASET_NAME,
            "fingerprint": self.fingerprint,
            "containers": [list(chain) for chain in containers],
            "rooms": rooms,
            "duplicates": self.duplicates,
            "lfs_pointers": self.lfs_pointers,
            "skipped_archives": self.skipped_archives,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "AcousticRoomsIndex":
        if payload.get("index_version") != INDEX_VERSION:
            raise AcousticRoomsError("index cache has an unsupported version")
        containers = [tuple(chain) for chain in payload["containers"]]

        def unpack(packed: list[Any] | None) -> MemberRef | None:
            return None if packed is None else MemberRef((*containers[packed[0]], packed[1]))

        index = cls(
            fingerprint=payload["fingerprint"],
            duplicates=list(payload["duplicates"]),
            lfs_pointers=list(payload["lfs_pointers"]),
            skipped_archives=list(payload["skipped_archives"]),
        )
        for room_id, room in payload["rooms"].items():
            index.rooms[room_id] = RoomEntry(
                room_id=room_id,
                category=room["category"],
                mesh=unpack(room["mesh"]),
                simulation=unpack(room["simulation"]),
                rirs={(s, r): MemberRef((*containers[c], name)) for s, r, c, name in room["rirs"]},
                metadata={(s, r): MemberRef((*containers[c], name)) for s, r, c, name in room["metadata"]},
            )
        return index


def resolve_root(explicit: str | os.PathLike[str] | None = None, default: str = "data/raw") -> Path:
    """Dataset root from an explicit argument, then the env var, then a relative default."""
    chosen = explicit or os.environ.get(ROOT_ENV_VAR) or default
    root = Path(chosen)
    if not root.is_dir():
        raise AcousticRoomsError(f"dataset root not found: {root} (set {ROOT_ENV_VAR} or pass --root)")
    return root


def classify(member_path: str) -> tuple[str, str, str, PairId | None] | None:
    """Role of a path inside the dataset: ``(kind, category, room_id, pair)``.

    ``kind`` is one of ``rir``, ``metadata``, ``mesh``, ``simulation``.
    Returns ``None`` for anything else, including the raw dense subset.
    """
    path = member_path.replace("\\", "/")
    if _EXCLUDED_RE.search(path):
        return None
    for kind, pattern in (("metadata", _METADATA_RE), ("rir", _RIR_RE), ("mesh", _MESH_RE), ("simulation", _SIMULATION_RE)):
        match = pattern.search(path)
        if match:
            groups = match.groupdict()
            pair = (int(groups["source"]), int(groups["receiver"])) if "source" in groups else None
            return kind, groups["category"], groups["room"], pair
    return None


class ArchiveReader:
    """Opens (nested) zip members read-only and keeps the handles for reuse."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self._zips: dict[Chain, zipfile.ZipFile] = {}

    def __enter__(self) -> "ArchiveReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for handle in reversed(list(self._zips.values())):
            handle.close()
        self._zips.clear()

    def open_zip(self, chain: Chain) -> zipfile.ZipFile:
        """Zip at ``chain``: a file under the root, then zips nested inside it."""
        cached = self._zips.get(chain)
        if cached is not None:
            return cached
        if len(chain) == 1:
            handle = zipfile.ZipFile(self.root / chain[0], mode="r")
        else:
            parent = self.open_zip(chain[:-1])
            info = parent.getinfo(chain[-1])
            if info.compress_type == zipfile.ZIP_STORED:
                handle = zipfile.ZipFile(parent.open(info), mode="r")  # seekable without inflating
            elif info.file_size <= _MAX_IN_MEMORY_ZIP_BYTES:
                handle = zipfile.ZipFile(io.BytesIO(parent.read(info)), mode="r")
            else:
                raise AcousticRoomsError(f"compressed nested zip too large to read in place: {' :: '.join(chain)}")
        self._zips[chain] = handle
        return handle

    def read(self, ref: MemberRef) -> bytes:
        if len(ref.chain) == 1:
            return (self.root / ref.chain[0]).read_bytes()
        return self.open_zip(ref.chain[:-1]).read(ref.chain[-1])


def _is_lfs_pointer(size: int, head: bytes) -> bool:
    return size <= _LFS_MAX_BYTES and head.startswith(_LFS_MAGIC)


def _walk_zip(reader: ArchiveReader, chain: Chain, index: AcousticRoomsIndex) -> Iterator[tuple[Chain, str]]:
    """Yield ``(container_chain, member_name)`` for every file, descending into nested zips."""
    archive = reader.open_zip(chain)
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if info.file_size <= _LFS_MAX_BYTES and _is_lfs_pointer(info.file_size, archive.read(info)):
            index.lfs_pointers.append(" :: ".join((*chain, name)))
            continue
        if name.lower().endswith(".zip"):
            if _EXCLUDED_RE.search(name):
                continue
            try:
                yield from _walk_zip(reader, (*chain, name), index)
            except (AcousticRoomsError, zipfile.BadZipFile) as exc:
                index.skipped_archives.append(f"{' :: '.join((*chain, name))}: {exc}")
            continue
        yield chain, name


def _top_level_sources(root: Path) -> tuple[list[str], list[str]]:
    """Zip files and loose dataset files directly under the root that belong to AcousticRooms."""
    zips: list[str] = []
    loose: list[str] = []
    for entry in sorted(root.iterdir()):
        lowered = entry.name.lower()
        if entry.is_file() and lowered.endswith(".zip"):
            if lowered.startswith(("acousticrooms", "single_channel_ir", "metadata")):
                zips.append(entry.name)
        elif entry.is_dir() and entry.name in ("AcousticRooms", "room_mesh_obj_format", "simulation_info"):
            for path in sorted(entry.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(root).as_posix()
                    (zips if relative.lower().endswith(".zip") and not _EXCLUDED_RE.search(relative) else loose).append(relative)
    return zips, loose


def source_fingerprint(root: Path) -> list[list[Any]]:
    """Name, size and mtime of every top-level source, to invalidate a stale cache."""
    zips, loose = _top_level_sources(root)
    return [[name, (root / name).stat().st_size, int((root / name).stat().st_mtime)] for name in (*zips, *loose)]


def build_index(root: str | os.PathLike[str]) -> AcousticRoomsIndex:
    """Scan the archives' directories and group every member by room and pair id."""
    root = Path(root)
    zips, loose = _top_level_sources(root)
    if not zips and not loose:
        raise AcousticRoomsError(f"no AcousticRooms archives or folders found under {root}")
    index = AcousticRoomsIndex(fingerprint=source_fingerprint(root))

    def register(ref: MemberRef) -> None:
        role = classify(ref.chain[-1])
        if role is None:
            return
        kind, category, room_id, pair = role
        room = index.rooms.setdefault(room_id, RoomEntry(room_id=room_id, category=category))
        if room.category != category:
            index.duplicates.append(f"{room_id}: categories {room.category} and {category}")
        if kind in ("rir", "metadata"):
            table = room.rirs if kind == "rir" else room.metadata
            assert pair is not None
            if pair in table:
                index.duplicates.append(f"{room_id} {kind} S{pair[0]}_R{pair[1]}: {table[pair].display} and {ref.display}")
            else:
                table[pair] = ref
        else:
            existing = getattr(room, kind)
            if existing is not None:
                index.duplicates.append(f"{room_id} {kind}: {existing.display} and {ref.display}")
            else:
                setattr(room, kind, ref)

    with ArchiveReader(root) as reader:
        for name in loose:
            head = b""
            path = root / name
            if path.stat().st_size <= _LFS_MAX_BYTES:
                head = path.read_bytes()
            if _is_lfs_pointer(path.stat().st_size, head):
                index.lfs_pointers.append(name)
            else:
                register(MemberRef((name,)))
        for name in zips:
            for chain, member in _walk_zip(reader, (name,), index):
                register(MemberRef((*chain, member)))
    return index


def save_index(index: AcousticRoomsIndex, path: str | os.PathLike[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index.to_json(), separators=(",", ":")), encoding="utf-8")


def load_cached_index(root: str | os.PathLike[str], path: str | os.PathLike[str]) -> AcousticRoomsIndex | None:
    """Cached index if it exists and still matches the archives on disk."""
    cache = Path(path)
    if not cache.is_file():
        return None
    try:
        index = AcousticRoomsIndex.from_json(json.loads(cache.read_text(encoding="utf-8")))
    except (AcousticRoomsError, KeyError, ValueError):
        return None
    return index if index.fingerprint == source_fingerprint(Path(root)) else None


def identity_report(index: AcousticRoomsIndex) -> dict[str, Any]:
    """Orphans on every side of the mesh / metadata / RIR join."""
    rooms = index.rooms.values()
    rir_only = {room.room_id: sorted(room.rirs.keys() - room.metadata.keys()) for room in rooms}
    metadata_only = {room.room_id: sorted(room.metadata.keys() - room.rirs.keys()) for room in rooms}
    return {
        "rooms_total": len(index.rooms),
        "rooms_with_mesh": sum(room.mesh is not None for room in rooms),
        "rooms_with_rirs": sum(bool(room.rirs) for room in rooms),
        "rooms_with_metadata": sum(bool(room.metadata) for room in rooms),
        "rirs_total": sum(len(room.rirs) for room in rooms),
        "metadata_total": sum(len(room.metadata) for room in rooms),
        "pairs_matched": sum(len(room.paired()) for room in rooms),
        "rooms_with_rirs_but_no_mesh": sorted(room.room_id for room in rooms if room.rirs and room.mesh is None),
        "meshes_without_rirs": sorted(room.room_id for room in rooms if room.mesh is not None and not room.rirs),
        "rirs_without_metadata": {room_id: pairs for room_id, pairs in sorted(rir_only.items()) if pairs},
        "metadata_without_rirs": {room_id: pairs for room_id, pairs in sorted(metadata_only.items()) if pairs},
        "duplicates": list(index.duplicates),
        "lfs_pointers": list(index.lfs_pointers),
        "skipped_archives": list(index.skipped_archives),
    }


def _xyz(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AcousticRoomsError(f"{name} is not an [x, y, z] triple: {value!r}")
    x, y, z = (float(component) for component in value)
    return x, y, z


def load_pair_metadata(reader: ArchiveReader, ref: MemberRef) -> PairMetadata:
    record = json.loads(reader.read(ref))
    if not isinstance(record, dict) or not {"src_loc", "rec_loc"} <= record.keys():
        keys = sorted(record) if isinstance(record, dict) else type(record).__name__
        raise AcousticRoomsError(f"unexpected metadata fields {keys} in {ref.display}")
    return PairMetadata(_xyz(record["src_loc"], "src_loc"), _xyz(record["rec_loc"], "rec_loc"))


def load_rir(reader: ArchiveReader, ref: MemberRef) -> Rir:
    from scipy.io import wavfile

    sample_rate, samples = wavfile.read(io.BytesIO(reader.read(ref)))
    native = samples.dtype
    waveform = samples.astype(np.float64)
    if np.issubdtype(native, np.integer):
        waveform /= float(np.iinfo(native).max)
    return Rir(waveform=waveform, sample_rate_hz=int(sample_rate), native_dtype=str(native))


def load_mesh(reader: ArchiveReader, ref: MemberRef) -> "trimesh.Trimesh":
    """Room mesh with vertices and faces untouched (no merging, no reorientation)."""
    import trimesh

    mesh = trimesh.load(io.BytesIO(reader.read(ref)), file_type="obj", force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise AcousticRoomsError(f"no triangle mesh in {ref.display}")
    return mesh


def load_simulation(reader: ArchiveReader, ref: MemberRef) -> dict[str, Any]:
    return json.loads(reader.read(ref))
