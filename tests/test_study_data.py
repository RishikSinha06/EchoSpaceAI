"""Interrupt/resume preparation without changing sample identity or split ownership."""
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from echospace.training import study_data
from echospace.training.audio_frozen import FrozenAudioDataset
from echospace.training.synthetic_audio import synthetic_splits


def test_freeze_resume_recovers_partial_journal_and_matches_uninterrupted(tmp_path,monkeypatch):
    splits = synthetic_splits(3,1,1,1)
    def online(*args):
        yield from itertools.islice(splits['train'],args[-1],None)
    monkeypatch.setattr(study_data,'online_resume',online)
    monkeypatch.setattr(study_data,'fixed_items',lambda *a:iter(splits[a[-1]]))
    kwargs = dict(cache='unused',eval_dir='unused',folds={'version':'test'},spec={'checksum':'test'},fold=0,
                  settings={'seed':0,'epochs':1,'samples_per_room':1,'workers':1},provenance='test')
    whole = study_data.prepare_fold(tmp_path/'whole',**kwargs)
    real_save = np.savez_compressed
    calls = [0]
    def interrupted(*args,**kwargs):
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError('test preparation interruption')
        return real_save(*args,**kwargs)
    monkeypatch.setattr(np,'savez_compressed',interrupted)
    with pytest.raises(RuntimeError,match='interruption'):
        study_data.prepare_fold(tmp_path/'resumed',**kwargs)
    with (tmp_path/'resumed/entries.jsonl').open('ab') as f:
        f.write(b'{"incomplete":')
    monkeypatch.setattr(np,'savez_compressed',real_save)
    resumed = study_data.prepare_fold(tmp_path/'resumed',**kwargs)
    assert json.loads(whole.read_text()) == json.loads(resumed.read_text())
    ds = FrozenAudioDataset(resumed,'train',k=8)
    assert len(ds)==3
    ds[0]


def test_freeze_resume_rejects_changed_sample(tmp_path,monkeypatch):
    splits = synthetic_splits(1,1,1,1)
    root = tmp_path/'partial'
    root.mkdir()
    arrays = study_data._arrays(splits['train'][0])
    meta = {k:splits['train'][0][k] for k in study_data.META_KEYS}
    (root/'samples').mkdir()
    np.savez_compressed(root/'samples/train_000000.npz',**arrays)
    row = {'split':'train','epoch':0,'path':'samples/train_000000.npz','metadata':meta,
           'sha256':study_data.sample_sha(meta,arrays)}
    (root/'entries.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    arrays['rir'][0,0] += 1
    np.savez_compressed(root/'samples/train_000000.npz',**arrays)
    with pytest.raises(ValueError,match='integrity'):
        study_data.verified_journal(root)


def test_online_resume_does_not_recompute_skipped_jobs(monkeypatch):
    import echospace.data
    class FakeDataset:
        rooms = ['room']
        def __init__(self,*a,**k): pass
        def __len__(self): return 5
    monkeypatch.setattr(echospace.data,'EchoSpaceDataset',FakeDataset)
    calls = []
    def worker(args):
        calls.append((args[-2],args[-1]))
        return [(args[-2],i) for i in args[-1]]
    monkeypatch.setattr(study_data,'_online_worker',worker)
    result = list(study_data.online_resume('',{'folds':[{'train_rooms':['room']}]},{},0,0,3,1,1,7))
    assert result == [(1,i) for i in range(2,5)] + [(2,i) for i in range(5)]
    assert calls[0] == (1,[2,3,4])
