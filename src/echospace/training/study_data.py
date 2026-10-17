"""Resumable study freezes; verify and journal each materialized sample.

Recovery resumes the deterministic online generator at the committed sample
offset. A partial final journal line is discarded; any complete changed entry
or sample fails closed. Existing frozen arrays are never silently relabelled.
"""
import copy
import hashlib
import itertools
import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .audio_frozen import (ARRAY_KEYS, META_KEYS, FORMAT, _arrays, _online_worker,
                           fixed_items, sample_sha, validate_audio_item)
from .frozen import json_sha
from .packed import file_sha
from .runner import atomic_json


def online_resume(cache, folds, spec, fold, seed, epochs, samples_per_room, workers, start):
    """Bounded ordered prefetch; skipped jobs are not re-simulated."""
    from echospace.data import EchoSpaceDataset
    ds = EchoSpaceDataset(cache, folds, fold, 'train', eval_spec=spec, seed=seed,
                         samples_per_room=samples_per_room, k_range=(8, 8), max_cached_rooms=2)
    expected = set(folds['folds'][fold]['train_rooms'])
    if set(ds.rooms) != expected:
        raise ValueError('some study training rooms are infeasible or unavailable')
    total = len(ds)
    if not 0 <= start <= total*epochs:
        raise ValueError('invalid resume sample count')
    shared = dict(folds, _samples_per_room=samples_per_room)
    jobs = []
    for epoch in range(epochs):
        first = max(0, start-epoch*total)
        for i in range(first, total, 64):
            jobs.append((str(cache), shared, spec, fold, seed, epoch, list(range(i, min(i+64, total)))))
    if workers == 1:
        for job in jobs:
            yield from _online_worker(job)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = iter(jobs)
        pending = deque(pool.submit(_online_worker, j) for j in itertools.islice(iterator, workers))
        while pending:
            yield from pending.popleft().result()
            next_job = next(iterator, None)
            if next_job is not None:
                pending.append(pool.submit(_online_worker, next_job))


def verified_journal(root, emit=None):
    path = root/'entries.jsonl'
    if not path.exists():
        return []
    data = path.read_bytes()
    rows, repaired = [], False
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not line.endswith(b'\n'):
            if i != len(lines)-1:
                raise ValueError('malformed nonfinal journal line')
            repaired = True
            break
        row = json.loads(line)
        sample = (root/row['path']).resolve()
        if not sample.is_relative_to(root.resolve()):
            raise ValueError('journal path escapes freeze')
        with np.load(sample, allow_pickle=False) as z:
            arrays = {k: z[k] for k in ARRAY_KEYS}
        if sample_sha(row['metadata'], arrays) != row['sha256']:
            raise ValueError('journal sample integrity mismatch')
        validate_audio_item(arrays, row['metadata'])
        rows.append(row)
        if emit and len(rows) % 2048 == 0:
            emit({'stage': 'verify_preparation_resume', 'samples': len(rows)})
    if repaired:
        temporary = path.with_suffix('.repair')
        temporary.write_bytes(b''.join(lines[:len(rows)]))
        os.replace(temporary, path)
    return rows


def prepare_fold(destination, *, cache, eval_dir, folds, spec, fold, settings, provenance, emit=None):
    root = Path(destination)
    source = {'kind': 'dual_study_online', 'fold': fold, 'settings': settings,
              'folds_checksum': folds['checksum'] if 'checksum' in folds else json_sha(folds),
              'eval_spec_checksum': spec['checksum'], 'preflight_checksum': provenance,
              'train_epochs': settings['epochs'],
              'val_test': 'committed fixed masks; clean; held out by physical room group'}
    plan = {'format': 'dual.prepare.1', 'source': source}
    root.mkdir(parents=True, exist_ok=True)
    if (root/'preparation.json').exists():
        if json.loads((root/'preparation.json').read_text(encoding='utf-8')) != plan:
            raise ValueError('preparation identity changed')
    else:
        if any(root.iterdir()):
            raise ValueError('unknown partial preparation directory')
        atomic_json(root/'preparation.json', plan)
    if (root/'manifest.json').exists():
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['source'] != source or manifest['checksum'] != json_sha({k:v for k,v in manifest.items() if k!='checksum'}):
            raise ValueError('completed preparation identity changed')
        return root/'manifest.json'
    entries = verified_journal(root, emit)
    counts = {s: sum(e['split'] == s for e in entries) for s in ('train', 'val', 'test')}
    owners, seen = {}, set()
    def identity(entry):
        meta = entry['metadata']
        unique = (meta['sample_id'], entry['epoch'])
        if unique in seen:
            raise ValueError('duplicate prepared sample')
        seen.add(unique)
        for name in (meta['room_id'], f"group:{meta['room_group']}"):
            if owners.setdefault(name, entry['split']) != entry['split']:
                raise ValueError('prepared room group crosses splits')
    for e in entries:
        identity(e)
    (root/'samples').mkdir(exist_ok=True)
    with (root/'entries.jsonl').open('a', encoding='utf-8', newline='\n') as journal:
        for split in ('train', 'val', 'test'):
            if split == 'train':
                items = online_resume(cache, folds, spec, fold, settings['seed'], settings['epochs'],
                                      settings['samples_per_room'], settings['workers'], counts[split])
            else:
                items = itertools.islice(fixed_items(eval_dir, folds, spec, fold, split), counts[split], None)
            for item in items:
                meta = {key: item[key] for key in META_KEYS}
                meta['room_group'] = int(meta['room_group'])
                arrays = _arrays(item)
                validate_audio_item(arrays, meta)
                name = f'samples/{split}_{counts[split]:06d}.npz'
                row = {'split': split, 'epoch': int(item.get('epoch', 0)), 'path': name,
                       'metadata': meta, 'sha256': sample_sha(meta, arrays)}
                identity(row)
                temporary = (root/name).with_suffix('.tmp.npz')
                np.savez_compressed(temporary, **arrays)
                os.replace(temporary, root/name)
                journal.write(json.dumps(row, sort_keys=True)+'\n')
                journal.flush()
                entries.append(row)
                counts[split] += 1
                if counts[split] % 64 == 0:
                    os.fsync(journal.fileno())
                if emit and counts[split] % 256 == 0:
                    emit({'stage': 'preparing', 'fold': fold, 'split': split, 'samples': counts[split]})
            if not counts[split]:
                raise ValueError(f'empty {split}')
    manifest = {'format': FORMAT, 'source': source, 'samples': entries,
                'train_epochs': len({e['epoch'] for e in entries if e['split']=='train'})}
    manifest['checksum'] = json_sha(manifest)
    atomic_json(root/'manifest.json', manifest)
    return root/'manifest.json'


def subset_freeze(source_path, destination, groups, draws=2):
    """Verified copy of development subsets; never alter the existing freeze."""
    root, source_path = Path(destination), Path(source_path)
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if source['checksum'] != json_sha({k:v for k,v in source.items() if k!='checksum'}):
        raise ValueError('invalid source manifest')
    if (root/'manifest.json').exists():
        old = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if (old['source']['parent_checksum'] != source['checksum'] or old['source']['groups'] != groups
                or old['source']['draws'] != draws):
            raise ValueError('pilot source changed')
        return root/'manifest.json'
    root.mkdir(parents=True, exist_ok=True)
    (root/'samples').mkdir(exist_ok=True)
    entries = []
    for entry in source['samples']:
        split = entry['split']
        if entry['metadata']['room_group'] not in groups[split] or split=='train' and entry['epoch']>=draws:
            continue
        path = (source_path.parent/entry['path']).resolve()
        if not path.is_relative_to(source_path.parent.resolve()):
            raise ValueError('source path escapes freeze')
        with np.load(path, allow_pickle=False) as z:
            arrays = {k:z[k] for k in ARRAY_KEYS}
        if sample_sha(entry['metadata'], arrays) != entry['sha256']:
            raise ValueError('pilot sample integrity mismatch')
        np.savez_compressed(root/entry['path'], **arrays)
        entries.append(copy.deepcopy(entry))
    record = {'format': FORMAT, 'source': {'kind':'dual_development_subset', 'parent_checksum':source['checksum'],
               'groups':groups, 'draws':draws, 'selection':'seeded group IDs; no metric-based selection'},
              'samples': entries, 'train_epochs': draws}
    if any(not any(e['split']==s for e in entries) for s in groups):
        raise ValueError('empty pilot split')
    record['checksum'] = json_sha(record)
    atomic_json(root/'manifest.json', record)
    return root/'manifest.json'
