"""Local two-architecture study: verified preparation, pilot, 5 folds x 3 seeds.

Run --mode all --resume after the model/correctness gate succeeds. Every epoch
checkpoints; preparation has its own sample journal. No paid services or downloads.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
import numpy as np
import torch
from echospace.data import load_folds, load_eval_spec
from echospace.models.factory import build
from echospace.training.audio_frozen import FrozenAudioDataset
from echospace.training.demo import select_groups, room_metrics, paired_group_delta
from echospace.training.frozen import json_sha
from echospace.training.fusion_runner import FusionTrainConfig, run_fusion, load_fusion_checkpoint, evaluate, _source_hashes
from echospace.training.packed import pack_freeze, file_sha
from echospace.training.runner import atomic_json, seed_everything
from echospace.training.study_data import prepare_fold, subset_freeze
from echospace.training.quality import evaluate_quality

ROOT = REPO/'artifacts/dual_study_20261017'


def emit(row):
    record = {'time_utc': datetime.now(timezone.utc).isoformat(), **row}
    print(json.dumps(record), flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record)+'\n')
    atomic_json(ROOT/'status.json', record)


def fingerprint():
    files = sorted((REPO/'src/echospace').rglob('*.py')) + [Path(__file__), REPO/'configs/dual_architecture_study.json']
    return {p.relative_to(REPO).as_posix(): file_sha(p) for p in files}


def verify_sources():
    previous = json.loads((REPO/'artifacts/p9_full_fold0/preflight.json').read_text(encoding='utf-8'))
    if previous['gate'] != 'PASS' or len(previous['rooms']) != 249:
        raise ValueError('verified original data preflight is missing')
    audits = {r['room_id']:r for r in map(json.loads, (REPO/'data/manifests/acousticrooms_index.jsonl').read_text(encoding='utf-8').splitlines())}
    for i, (room, row) in enumerate(previous['rooms'].items()):
        audit = audits[room]
        if audit.get('accepted') is not True or json_sha(audit) != row['audit_sha256']:
            raise ValueError('accepted D0 identity changed')
        if file_sha(REPO/'data/processed/rooms'/f'{room}.npz') != row['cache_sha256']:
            raise ValueError('room cache changed since verified waveform/pose audit')
        if (i+1) % 25 == 0:
            emit({'stage':'preflight', 'verified_rooms':i+1, 'total':249})
    original = json.loads((REPO/'data/cache/p9_full_fold0/manifest.json').read_text(encoding='utf-8'))
    if original['checksum'] != json_sha({k:v for k,v in original.items() if k!='checksum'}):
        raise ValueError('original freeze changed')
    if file_sha(REPO/'data/splits/folds.json') != original['source']['folds_sha256']:
        raise ValueError('fold assignments changed')
    if load_eval_spec(REPO/'data/splits/eval_masks.json')['checksum'] != original['source']['eval_spec_checksum']:
        raise ValueError('fixed evaluation masks changed')
    if original['source']['preflight']['sha256'] != file_sha(REPO/'artifacts/p9_full_fold0/preflight.json'):
        raise ValueError('original preflight binding changed')
    return {'gate':'PASS', 'previous_preflight_sha256':file_sha(REPO/'artifacts/p9_full_fold0/preflight.json'),
            'rooms':249, 'room_cache_hashes':previous['rooms'], 'fold0_checksum':original['checksum']}


def model_config(settings, row):
    config = dict(settings['backbone'], family=row['family'])
    if row['family']=='WallFusion':
        config.update(settings['wall'])
    config.update(row.get('overrides', {}))
    return config


def train_config(settings, seed, packed, pilot=False):
    config = dict(settings['training'], seed=seed, packed_dir=str(packed))
    if pilot:
        config.update(epochs=settings['pilot']['epochs'], patience=settings['pilot']['patience'])
    config['k_range'], config['eval_k'] = tuple(config['k_range']), tuple(config['eval_k'])
    return FusionTrainConfig(**config)


def ensure_packed(frozen, destination):
    destination = Path(destination)
    if not (destination/'manifest.json').exists():
        if destination.exists() and any(destination.iterdir()):
            # Interrupted pack is a regenerable sidecar; preserve it for review.
            saved = destination.with_name(destination.name+'_partial_'+str(time.time_ns()))
            if not destination.resolve().is_relative_to((REPO/'data/cache').resolve()):
                raise ValueError('packed recovery outside cache')
            destination.rename(saved)
            emit({'stage':'preserved_partial_pack', 'path':str(saved)})
        pack_freeze(frozen, destination, progress=emit)
    # Verify all arrays even when reusing a previous completed sidecar.
    FrozenAudioDataset(frozen, 'val', k=8, packed_dir=destination)
    return destination


def run_job(settings, row, seed, fold, frozen, packed, phase):
    root = ROOT/phase/f'fold{fold}'/row['name']/f'seed{seed}'
    cfg = train_config(settings, seed, packed, phase=='pilot')
    config = model_config(settings, row)
    if (root/'run.json').exists():
        recorded = json.loads((root/'run.json').read_text(encoding='utf-8'))
        if recorded['status']=='complete':
            model, payload = load_fusion_checkpoint(root/'best.pt')
            if (recorded['source_files_sha256'] != _source_hashes()
                    or recorded['checkpoint_sha256'] != file_sha(root/'best.pt')
                    or payload['frozen_checksum'] != FrozenAudioDataset(frozen,'val',k=8).checksum
                    or recorded['model'] != build(row['variant'],config).specification()):
                raise ValueError('completed study run identity changed')
            return recorded
        if not (root/'last.pt').exists() and not recorded['history']:
            # The initial epoch never committed. Preserve diagnostics and retry
            # from the same seed; no optimizer state exists to resume.
            saved = root.with_name(root.name+'_uncommitted_'+str(time.time_ns()))
            root.rename(saved)
            emit({'stage':'preserved_uncommitted_run','path':str(saved)})
    started = time.perf_counter()
    emit({'stage':'training_start', 'phase':phase, 'fold':fold, 'model':row['name'], 'seed':seed})
    def progress(event):
        emit({'stage':'training', 'phase':phase, 'fold':fold, 'model':row['name'], 'seed':seed,
              'elapsed_seconds':time.perf_counter()-started, **event})
        if (ROOT/'STOP_AFTER_EPOCH').exists():
            raise RuntimeError('study stop requested; epoch checkpoint saved')
    record = run_fusion(row['variant'], frozen, root, cfg, config, progress=progress,
                        resume=(root/'run.json').exists())
    emit({'stage':'fit_complete', 'phase':phase, 'fold':fold, 'model':row['name'], 'seed':seed,
          'epochs':len(record['history']), 'best_epoch':record['best_epoch'],
          'validation_score':record['best_validation_score'], 'seconds':time.perf_counter()-started})
    return record


def profile(settings, frozen, packed):
    seed_everything(0)
    torch.set_num_threads(settings['training']['threads'])
    ds = FrozenAudioDataset(frozen, 'train', k=8, packed_dir=packed)
    count = settings['training']['batch_size']
    batch = {key:torch.stack([ds[i][key] for i in range(count)]).cuda() for key in ds[0]}
    from echospace.training.losses import masked_bce_dice
    rows = []
    for row in (settings['models'][2], settings['models'][4]):
        net = build(row['variant'], model_config(settings,row)).cuda().train()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(4):
            torch.cuda.synchronize()
            start = time.perf_counter()
            net.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=True):
                loss = masked_bce_dice(net(batch),batch)
            loss.backward()
            torch.cuda.synchronize()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite profile loss')
            times.append(time.perf_counter()-start)
        result = {'model':row['name'], 'parameters':sum(p.numel() for p in net.parameters()),
                  'batch_size':count, 'seconds_per_train_batch':float(np.median(times[1:])),
                  'peak_cuda_allocated_MB':torch.cuda.max_memory_allocated()/1e6,
                  'scope':'forward/backward only; excludes optimizer, validation and data preparation'}
        rows.append(result)
        emit({'stage':'profile', **result})
        del net
        torch.cuda.empty_cache()
    atomic_json(ROOT/'profile.json', rows)


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO/'configs/dual_architecture_study.json')
    parser.add_argument('--out-dir', type=Path, default=ROOT)
    parser.add_argument('--mode', choices=('profile','pilot','all'), default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    ROOT = args.out_dir.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    if settings['format']!='dual.study.1' or settings['training']['device']!='cuda':
        raise ValueError('unknown protocol or nonlocal compute configuration')
    identity = {'config_sha256':file_sha(args.config), 'source_files':fingerprint(),
                'authorization':settings['authorization']}
    if (ROOT/'plan.json').exists():
        if not args.resume or json.loads((ROOT/'plan.json').read_text(encoding='utf-8')) != identity:
            raise ValueError('resume needs identical study source/configuration')
    else:
        atomic_json(ROOT/'plan.json', identity)
    awake = False
    try:
        if os.name == 'nt':
            import ctypes
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        emit({'stage':'starting', 'pid':os.getpid(), 'idle_sleep_inhibited':awake,
              'independent_measured_validation':'pending user data'})
        if not torch.cuda.is_available():
            raise ValueError('local CUDA unavailable')
        if shutil.disk_usage(REPO).free < 25*1024**3:
            raise ValueError('at least 25 GiB free disk space required for this study')
        if not (ROOT/'data_preflight.json').exists():
            verified = verify_sources()
            verified['versions'] = {d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
            atomic_json(ROOT/'data_preflight.json', verified)
        else:
            verified = json.loads((ROOT/'data_preflight.json').read_text(encoding='utf-8'))
            # Revalidate source archives on every restart, rather than trust stale state.
            checked = verify_sources()
            if any(verified[k] != checked[k] for k in checked):
                raise ValueError('data preflight changed')
        folds = load_folds(REPO/'data/splits/folds.json')
        spec = load_eval_spec(REPO/'data/splits/eval_masks.json')
        parent = REPO/'data/cache/p9_full_fold0/manifest.json'
        initial_source = json.loads(parent.read_text(encoding='utf-8'))['source']
        expected = settings['preparation']
        if (initial_source['seed'] != expected['seed'] or initial_source['train_epochs'] != expected['epochs']
                or initial_source['samples_per_room'] != expected['samples_per_room']):
            raise ValueError('reused fold-0 preparation differs from this study protocol')
        _, selected = select_groups(folds, 0, settings['pilot']['groups'], seed=741)
        groups = {split:part['groups'] for split,part in selected.items()}
        pilot = subset_freeze(parent, REPO/'data/cache/dual_study_pilot', groups, settings['pilot']['draws'])
        pilot_packed = ensure_packed(pilot, REPO/'data/cache/dual_study_pilot_packed')
        profile(settings, pilot, pilot_packed)
        if args.mode=='profile':
            emit({'stage':'profile_complete'})
            return
        pilot_rows = []
        for row in settings['models']+settings['ablations']:
            for seed in settings['seeds']:
                result = run_job(settings,row,seed,0,pilot,pilot_packed,'pilot')
                pilot_rows.append({'model':row['name'],'seed':seed,'parameters':result['parameter_count'],
                                   'best_epoch':result['best_epoch'],'completed_epochs':len(result['history']),
                                   'validation_score':result['best_validation_score'],
                                   'checkpoint_sha256':result['checkpoint_sha256']})
        atomic_json(ROOT/'pilot_summary.json',{'scope':'development validation only; settings remain fixed',
                    'fits':pilot_rows, 'test_scored':False})
        emit({'stage':'pilot_complete', 'fits':len(settings['seeds'])*(len(settings['models'])+len(settings['ablations']))})
        if args.mode=='pilot':
            return
        manifests = {}
        models = []
        for fold in settings['folds']:
            if fold==0:
                frozen, packed = parent, REPO/'data/cache/p9_full_fold0_packed'
            else:
                frozen = prepare_fold(REPO/f'data/cache/dual_study_fold{fold}', cache=REPO/'data/processed/rooms',
                           eval_dir=REPO/'data/processed/eval_samples', folds=folds, spec=spec, fold=fold,
                           settings=settings['preparation'], provenance=json_sha(verified), emit=emit)
                packed = REPO/f'data/cache/dual_study_fold{fold}_packed'
            ensure_packed(frozen,packed)
            manifests[fold] = (frozen,packed)
            # Fold-specific preflight: no room/group membership inferred from paths.
            body = json.loads(frozen.read_text(encoding='utf-8'))
            for split in ('train','val','test'):
                rooms = {e['metadata']['room_id'] for e in body['samples'] if e['split']==split}
                if rooms != set(folds['folds'][fold][split+'_rooms']):
                    raise ValueError('study freeze room coverage differs from fold')
            for row in settings['models'] + (settings['ablations'] if fold==0 else []):
                for seed in settings['seeds']:
                    record = run_job(settings,row,seed,fold,frozen,packed,'full')
                    models.append({'fold':fold, 'seed':seed, 'name':row['name'],
                                   'checkpoint_sha256':record['checkpoint_sha256'], 'best_epoch':record['best_epoch'],
                                   'validation_score':record['best_validation_score'], 'frozen_checksum':record['frozen_checksum']})
        # No final held-out metrics are read until every scheduled fit is locked.
        lock = {'status':'locked', 'models':models, 'config_sha256':file_sha(args.config),
                'scope':'internal five-fold comparison; prior development exposure; independent measured test pending'}
        atomic_json(ROOT/'selection.json',lock)
        lock_sha = file_sha(ROOT/'selection.json')
        emit({'stage':'selection_locked', 'fits':len(models)})
        final = {}
        for entry in models:
            fold, seed, name = entry['fold'], entry['seed'], entry['name']
            root = ROOT/'full'/f'fold{fold}'/name/f'seed{seed}'
            if file_sha(root/'best.pt') != entry['checkpoint_sha256']:
                raise ValueError('locked checkpoint changed')
            net, _ = load_fusion_checkpoint(root/'best.pt',entry['frozen_checksum'])
            net.cuda()
            frozen, packed = manifests[fold]
            final.setdefault(name,{}).setdefault(str(seed),{})
            for k in settings['training']['eval_k']:
                destination = root/f'test_K{k}.json'
                if destination.exists():
                    record = json.loads(destination.read_text(encoding='utf-8'))
                    if record['selection_sha256'] != lock_sha:
                        raise ValueError('existing test result uses different selection')
                else:
                    dataset = FrozenAudioDataset(frozen,'test',k=k,packed_dir=packed)
                    metrics, rows = evaluate(net,dataset,train_config(settings,seed,packed),torch.device('cuda'),keep_rows=True)
                    record = {'selection_sha256':lock_sha,'checkpoint_sha256':entry['checkpoint_sha256'],
                              'frozen_checksum':entry['frozen_checksum'],'metrics':metrics,'rows':rows}
                    atomic_json(destination,record)
                final[name][str(seed)].setdefault(str(k),[]).extend(record['rows'])
                if k == 8:
                    diagnostic = root/'quality_K8.json'
                    if diagnostic.exists():
                        if json.loads(diagnostic.read_text(encoding='utf-8'))['selection_sha256'] != lock_sha:
                            raise ValueError('quality result uses different selection')
                    else:
                        dataset = FrozenAudioDataset(frozen,'test',k=k,packed_dir=packed)
                        quality = evaluate_quality(net,dataset,settings['training']['batch_size'],torch.device('cuda'))
                        atomic_json(diagnostic,{'selection_sha256':lock_sha,'checkpoint_sha256':entry['checkpoint_sha256'],
                                                'domain':'valid unobserved cells; threshold .5; raw uncalibrated Brier',
                                                'rows':quality})
            emit({'stage':'heldout_scored', 'fold':fold, 'model':name, 'seed':seed})
            del net
        summary = {name:{seed:{k:room_metrics(rows) for k,rows in ks.items()} for seed,ks in seeds.items()} for name,seeds in final.items()}
        paired = {str(seed):paired_group_delta(final['echo_D'][str(seed)]['8'],final['wall_D'][str(seed)]['8']) for seed in settings['seeds']}
        if file_sha(ROOT/'selection.json') != lock_sha:
            raise ValueError('selection changed while scoring')
        atomic_json(ROOT/'results.json', {'status':'internal_study_complete', 'selection_sha256':lock_sha,
                    'scores':summary, 'paired_wall_minus_echo_K8':paired,
                    'production_ready':False, 'pending':['independent measured rooms','calibration/robustness','inference API/viewer/runtime release gates']})
        emit({'stage':'internal_study_complete','production_ready':False})
    except BaseException as error:
        emit({'stage':'failed', 'error':type(error).__name__+': '+str(error)})
        raise
    finally:
        if awake:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    main()
