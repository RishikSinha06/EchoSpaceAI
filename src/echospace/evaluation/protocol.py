"""Stage II identity gates and validation-only temperature selection."""
import hashlib
import json
from pathlib import Path
import numpy as np

TEMPERATURES = (.5,.75,1.,1.5,2.,3.)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def verify_lock(selection_path, checkpoints, frozen_checksum):
    lock=json.loads(Path(selection_path).read_text(encoding='utf-8'))
    if lock.get('status')!='locked': raise ValueError('Stage I checkpoint selection is not locked')
    if lock.get('frozen_checksum',frozen_checksum)!=frozen_checksum:
        raise ValueError('selection/frozen mismatch')
    hashes={m['checkpoint_sha256']:m for m in lock['models']}
    for path in checkpoints:
        digest=sha(path)
        if digest not in hashes: raise ValueError('checkpoint is not in selection lock')
        if hashes[digest].get('frozen_checksum',frozen_checksum)!=frozen_checksum:
            raise ValueError('checkpoint/frozen mismatch')
    return {'selection_sha256':sha(selection_path),'checkpoint_sha256':[sha(p) for p in checkpoints]}

def probability(logits,temperature=1.):
    return 1/(1+np.exp(-np.clip(np.asarray(logits,dtype=float)/temperature,-80,80)))

def choose_temperature(records,split):
    if split!='val': raise ValueError('calibration may be fitted only on validation')
    if not records: raise ValueError('empty calibration set')
    rooms=sorted({r['room_id'] for r in records})
    scores=[]
    for temperature in TEMPERATURES:
        scores.append(float(np.mean([np.mean([np.mean((probability(r['logits'],temperature)-r['target'])**2)
                                              for r in records if r['room_id']==room]) for room in rooms])))
    best=min(range(len(scores)),key=lambda i:(scores[i],abs(TEMPERATURES[i]-1)))
    return {'temperature':TEMPERATURES[best],'objective':'equal-room validation missing-cell Brier',
            'grid':list(TEMPERATURES),'scores':scores,'fit_split':'val','rooms':len(rooms)}

def require_finished_study(root):
    root=Path(root)
    required=('selection.json','results.json')
    missing=[name for name in required if not (root/name).is_file()]
    if missing: raise ValueError('Stage I study is not frozen/scored: missing '+', '.join(missing))
    selection=json.loads((root/'selection.json').read_text(encoding='utf-8'))
    results=json.loads((root/'results.json').read_text(encoding='utf-8'))
    if selection.get('status')!='locked' or results.get('status')!='internal_study_complete':
        raise ValueError('Stage I study is not complete')
    if results.get('selection_sha256')!=sha(root/'selection.json'):
        raise ValueError('Stage I results/selection identity mismatch')
    if not selection['models']: raise ValueError('Stage I selection is empty')
    keys=[(e['fold'],e['name'],e['seed']) for e in selection['models']]
    if len(keys)!=len(set(keys)): raise ValueError('duplicate selected fit')
    for entry in selection['models']:
        path=root/'full'/f"fold{entry['fold']}"/entry['name']/f"seed{entry['seed']}"/'best.pt'
        if sha(path)!=entry['checkpoint_sha256']:raise ValueError('Stage I checkpoint changed')
    return selection


def verify_study_coverage(selection,settings):
    expected={(fold,m['name'],seed) for fold in settings['folds'] for m in settings['models'] for seed in settings['seeds']}
    expected|={(0,m['name'],seed) for m in settings.get('ablations',[]) for seed in settings['seeds']}
    actual={(e['fold'],e['name'],e['seed']) for e in selection['models']}
    if actual!=expected:raise ValueError('Stage I selected fit coverage differs from protocol')
