"""P9 small-data study: deterministic room-group selection and paired summaries."""
from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audio_frozen import fixed_items, write_freeze
from .frozen import json_sha


def select_groups(folds: dict, fold: int, counts: dict[str, int], seed: int = 0) -> tuple[dict, dict]:
    """Select groups by seeded hash, before loading waveforms or target masks.

    Keep all rooms and masks of chosen groups in their original fold partition.
    The returned folds object is an internal view, never a replacement folds file.
    """
    if seed < 0 or not 0 <= fold < len(folds['folds']):
        raise ValueError('invalid seed or fold')
    view = copy.deepcopy(folds)
    selection = {}
    for split in ('train', 'val', 'test'):
        groups = folds['folds'][fold][f'{split}_groups']
        count = counts[split]
        if not isinstance(count, int) or not 1 <= count <= len(groups):
            raise ValueError(f'{split} count must be within 1..{len(groups)}')
        ranked = sorted(groups, key=lambda g: hashlib.sha256(f'p9-demo|{seed}|{split}|{g}'.encode()).digest())
        chosen = sorted(ranked[:count])
        rooms = sorted(r for r in folds['folds'][fold][f'{split}_rooms']
                       if folds['rooms'][r]['room_group'] in chosen)
        if not rooms:
            raise ValueError(f'{split} has no rooms')
        view['folds'][fold][f'{split}_groups'] = chosen
        view['folds'][fold][f'{split}_rooms'] = rooms
        selection[split] = {'groups': chosen, 'rooms': rooms}
    owners = {}
    for split, part in selection.items():
        for key in [f'group:{g}' for g in part['groups']] + part['rooms']:
            if owners.setdefault(key, split) != split:
                raise ValueError('selected room/group crosses splits')
    return view, selection


def freeze_demo(destination, eval_dir, folds_path, spec_path, counts, seed=0, fold=0):
    from echospace.data import load_eval_spec, load_folds
    folds, spec = load_folds(folds_path), load_eval_spec(spec_path)
    view, selection = select_groups(folds, fold, counts, seed)
    source = {'kind': 'P9_demo_fixed_masks', 'demonstration_only': True,
              'fold': fold, 'selection_seed': seed, 'selection': selection,
              'selection_method': 'seeded SHA256 rank of original split group IDs; all selected-group masks',
              'folds_sha256': hashlib.sha256(Path(folds_path).read_bytes()).hexdigest(),
              'eval_spec_checksum': spec['checksum'], 'training_augmentation': 'none',
              'train_epochs': 1, 'available_k': 8}
    source['selection_checksum'] = json_sha(selection)
    return write_freeze(destination, {s: fixed_items(eval_dir, view, spec, fold, s)
                                     for s in ('train', 'val', 'test')}, source)


def room_metrics(rows: list[dict]) -> dict:
    """Equal weight per room, after averaging its scan masks."""
    rooms = defaultdict(list)
    for row in rows:
        rooms[row['room_id']].append(row)
    if not rooms:
        raise ValueError('empty metric rows')
    values = {r: {key: float(np.mean([row[key] for row in rs]))
                  for key in ('occupancy_iou', 'boundary_iou', 'loss')}
              for r, rs in sorted(rooms.items())}
    return {'rooms': len(rooms), 'samples': len(rows),
            **{key: float(np.mean([v[key] for v in values.values()]))
               for key in ('occupancy_iou', 'boundary_iou', 'loss')}, 'per_room': values}


def paired_group_delta(reference: list[dict], candidate: list[dict], seed=0, repeats=2000) -> dict:
    """Exploratory D-A mean room delta; bootstrap whole physical-room groups."""
    left = {r['sample_id']: r for r in reference}
    right = {r['sample_id']: r for r in candidate}
    if len(left) != len(reference) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError('paired sample IDs differ or repeat')
    room_deltas = defaultdict(list)
    group_by_room = {}
    for sid, a in left.items():
        d = right[sid]
        if any(a[k] != d[k] for k in ('room_id', 'room_group', 'k')):
            raise ValueError('paired identities or K differ')
        room_deltas[a['room_id']].append(d['occupancy_iou'] - a['occupancy_iou'])
        group_by_room[a['room_id']] = a['room_group']
    grouped = defaultdict(list)
    for room, vals in room_deltas.items():
        grouped[group_by_room[room]].append(float(np.mean(vals)))
    groups = sorted(grouped)
    rng = np.random.default_rng(seed)
    draws = [float(np.mean([v for i in rng.integers(0, len(groups), len(groups)) for v in grouped[groups[i]]]))
             for _ in range(repeats)]
    return {'mean_room_delta': float(np.mean([v for vals in grouped.values() for v in vals])),
            'group_bootstrap_95_percentile_interval': np.quantile(draws, [0.025, 0.975]).tolist(),
            'groups': len(groups), 'bootstrap_seed': seed, 'repeats': repeats,
            'scope': 'exploratory demo only; few groups and one training seed do not establish acoustic gain'}
