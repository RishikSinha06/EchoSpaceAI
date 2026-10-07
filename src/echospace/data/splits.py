"""Room-group folds and inner validation (plan section 1.5).

The unit of splitting is the D0 duplicate group (``room_group`` in
``reports/d0_duplicates.json``), never a room id: rooms sharing a mesh or an
identical footprint are one physical room and must stay on one side.

Groups are stratified by footprint size (tertiles) and by convex vs.
non-convex, so every fold receives L-shaped rooms, then dealt to folds in a
seeded order. Within each fold's training groups ~15 % are held out as inner
validation, again stratified.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPLITS_FORMAT = "p6.folds.1"


class SplitError(ValueError):
    """A folds file violates the grouping rules."""


@dataclass(frozen=True)
class RoomInfo:
    room_id: str
    room_group: int
    footprint_m2: float
    convex: bool


def _stratum(rooms: list[RoomInfo], edges: np.ndarray) -> str:
    area = float(np.median([r.footprint_m2 for r in rooms]))
    size = int(np.searchsorted(edges, area, side="right"))
    convex = all(r.convex for r in rooms)
    return f"size{size}_{'convex' if convex else 'nonconvex'}"


def make_folds(rooms: list[RoomInfo], n_folds: int = 5, inner_val_fraction: float = 0.15, seed: int = 0) -> dict[str, Any]:
    """Assign every group to one test fold and pick inner-validation groups per fold."""
    if n_folds < 2:
        raise SplitError("need at least two folds")
    groups: dict[int, list[RoomInfo]] = defaultdict(list)
    for room in rooms:
        groups[room.room_group].append(room)
    group_ids = sorted(groups)
    if len(group_ids) < n_folds:
        raise SplitError("fewer room groups than folds")
    areas = np.array([np.median([r.footprint_m2 for r in groups[g]]) for g in group_ids])
    edges = np.quantile(areas, [1 / 3, 2 / 3])
    strata: dict[str, list[int]] = defaultdict(list)
    for g in group_ids:
        strata[_stratum(groups[g], edges)].append(g)

    rng = np.random.default_rng(seed)
    fold_of: dict[int, int] = {}
    offset = 0
    for name in sorted(strata):
        members = [strata[name][i] for i in rng.permutation(len(strata[name]))]
        for k, g in enumerate(members):
            fold_of[g] = (offset + k) % n_folds
        offset += len(members)  # continue dealing where the last stratum stopped, to balance sizes

    folds = []
    for f in range(n_folds):
        test = sorted(g for g in group_ids if fold_of[g] == f)
        train_pool = [g for g in group_ids if fold_of[g] != f]
        by_stratum: dict[str, list[int]] = defaultdict(list)
        for g in train_pool:
            by_stratum[_stratum(groups[g], edges)].append(g)
        val: list[int] = []
        fold_rng = np.random.default_rng([seed, f])
        for name in sorted(by_stratum):
            members = sorted(by_stratum[name])
            take = int(round(inner_val_fraction * len(members)))
            val += [members[i] for i in fold_rng.permutation(len(members))[:take]]
        val = sorted(val)
        train = sorted(set(train_pool) - set(val))
        folds.append({
            "fold": f,
            "test_groups": test,
            "val_groups": val,
            "train_groups": train,
            "test_rooms": _rooms(groups, test),
            "val_rooms": _rooms(groups, val),
            "train_rooms": _rooms(groups, train),
        })
    result = {
        "format": SPLITS_FORMAT,
        "seed": seed,
        "n_folds": n_folds,
        "inner_val_fraction": inner_val_fraction,
        "size_tertile_edges_m2": [round(float(e), 4) for e in edges],
        "rooms": {r.room_id: {"room_group": r.room_group, "footprint_m2": round(r.footprint_m2, 4), "convex": r.convex,
                              "stratum": _stratum(groups[r.room_group], edges), "test_fold": fold_of[r.room_group]}
                  for r in sorted(rooms, key=lambda r: r.room_id)},
        "folds": folds,
    }
    result["checksum"] = folds_checksum(result)
    validate_folds(result)
    return result


def _rooms(groups: dict[int, list[RoomInfo]], ids: list[int]) -> list[str]:
    return sorted(r.room_id for g in ids for r in groups[g])


def folds_checksum(folds: dict[str, Any]) -> str:
    body = {k: v for k, v in folds.items() if k != "checksum"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_folds(folds: dict[str, Any]) -> None:
    """No group or room in two places; every group tested exactly once; checksum intact."""
    if folds.get("format") != SPLITS_FORMAT:
        raise SplitError("unknown folds format")
    if folds.get("checksum") != folds_checksum(folds):
        raise SplitError("folds checksum mismatch: the file was edited")
    group_of = {room: info["room_group"] for room, info in folds["rooms"].items()}
    tested: dict[int, int] = {}
    for fold in folds["folds"]:
        parts = [set(fold[f"{p}_groups"]) for p in ("test", "val", "train")]
        if parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]:
            raise SplitError(f"fold {fold['fold']}: a group is in two parts")
        if set().union(*parts) != set(group_of.values()):
            raise SplitError(f"fold {fold['fold']}: groups missing")
        for part in ("test", "val", "train"):
            rooms = fold[f"{part}_rooms"]
            if {group_of[r] for r in rooms} != set(fold[f"{part}_groups"]):
                raise SplitError(f"fold {fold['fold']}: {part} rooms and groups disagree")
        for g in fold["test_groups"]:
            if g in tested:
                raise SplitError(f"group {g} is tested in folds {tested[g]} and {fold['fold']}")
            tested[g] = fold["fold"]
    if set(tested) != set(group_of.values()):
        raise SplitError("some group is never tested")


def save_folds(folds: dict[str, Any], path: str | Path) -> None:
    validate_folds(folds)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(folds, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load_folds(path: str | Path) -> dict[str, Any]:
    folds = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_folds(folds)
    return folds
