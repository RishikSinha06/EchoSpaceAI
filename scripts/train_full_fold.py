"""Prepare verified full-fold augmented data, train A/P/D, then lock and evaluate."""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import torch
from echospace.acoustics import RirConfig
from echospace.data import load_folds, load_eval_spec, load_room_cache, EchoSpaceDataset
from echospace.training.audio_frozen import freeze_p8, FrozenAudioDataset, validate_audio_item, _arrays
from echospace.training.demo import room_metrics
from echospace.training.fusion_runner import FusionTrainConfig, run_fusion, load_fusion_checkpoint, evaluate, _source_hashes
from echospace.training.packed import pack_freeze, file_sha
from echospace.training.runner import atomic_json, seed_everything

ROOT=REPO/'artifacts/p9_full_fold0'
FROZEN=REPO/'data/cache/p9_full_fold0/manifest.json'
PACKED=REPO/'data/cache/p9_full_fold0_packed'


def emit(event):
    row={'time_utc':datetime.now(timezone.utc).isoformat(),**event}
    print(json.dumps(row),flush=True)
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'events.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(row)+'\n')
    atomic_json(ROOT/'status.json',row)


def preflight(settings):
    folds=load_folds(REPO/'data/splits/folds.json')
    spec=load_eval_spec(REPO/'data/splits/eval_masks.json')
    part=folds['folds'][settings['freeze']['fold']]
    audits={r['room_id']:r for r in map(json.loads,(REPO/'data/manifests/acousticrooms_index.jsonl').read_text(encoding='utf-8').splitlines())}
    rooms=sorted(set(part['train_rooms']+part['val_rooms']+part['test_rooms']))
    checked={}
    for i,room in enumerate(rooms):
        path=REPO/'data/processed/rooms'/f'{room}.npz'
        cache=load_room_cache(path,RirConfig())
        audit=audits[room]
        audit_sha=hashlib.sha256(json.dumps(audit,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (audit.get('accepted') is not True or cache.room_id!=room
                or cache.room_group!=folds['rooms'][room]['room_group']
                or cache.meta['audit_record_sha256']!=audit_sha
                or cache.meta['geometry_signature']!=audit['geometry_signature']):
            raise ValueError(f'{room}: room cache audit identity mismatch')
        n=len(cache.rir_ids)
        if (not n or len(set(cache.rir_ids))!=n or any(not r.startswith(room+'/') for r in cache.rir_ids)
                or len(cache.meta['wav_sha256'])!=n or len(cache.meta['metadata_sha256'])!=n
                or not np.isfinite(cache.waveforms).all() or not np.isfinite(cache.positions_scene_m).all()):
            raise ValueError(f'{room}: invalid cached pairs')
        for hashes in (cache.meta['wav_sha256'],cache.meta['metadata_sha256']):
            if any(len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
                raise ValueError(f'{room}: missing source pair hashes')
        checked[room]={'cache_sha256':file_sha(path),'audit_sha256':audit_sha,'pairs':n,'room_group':cache.room_group}
        if (i+1)%25==0: emit({'stage':'preflight','rooms_checked':i+1,'total':len(rooms)})
    probe=EchoSpaceDataset(REPO/'data/processed/rooms',folds,settings['freeze']['fold'],'train',eval_spec=spec,
                           seed=settings['freeze']['seed'],samples_per_room=settings['freeze']['samples_per_room'],k_range=(8,8),max_cached_rooms=2)
    if set(probe.rooms)!=set(part['train_rooms']):
        raise ValueError('some full-fold training rooms are unavailable/infeasible')
    start=time.perf_counter()
    for index in (0,len(probe)//2,len(probe)-1):
        item=probe[index]
        meta={key:item[key] for key in ('sample_id','room_id','room_group','grid','spatial_transform','occlusion_type','coverage_bin','coverage')}
        validate_audio_item(_arrays(item),meta)
    record={'gate':'PASS','rooms':checked,'split_counts':{s:{'rooms':len(part[s+'_rooms']),'groups':len(part[s+'_groups'])} for s in ('train','val','test')},
            'training_samples_per_draw':len(probe),'probe_seconds':time.perf_counter()-start,
            'rir_config':RirConfig().to_dict(),'cuda_available':torch.cuda.is_available(),
            'source_files_sha256':_source_hashes()}
    if not record['cuda_available']: raise ValueError('full run requires the requested CUDA device')
    atomic_json(ROOT/'preflight.json',record)
    emit({'stage':'preflight_complete','rooms':len(rooms),'training_samples_per_draw':len(probe),'probe_seconds':record['probe_seconds']})
    return record


def main():
    global ROOT,FROZEN,PACKED
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','all'))
    parser.add_argument('--config',type=Path,default=REPO/'configs/p9_full_fold0.json')
    parser.add_argument('--out-dir',type=Path,default=ROOT)
    parser.add_argument('--frozen-dir',type=Path,default=FROZEN.parent)
    parser.add_argument('--packed-dir',type=Path,default=PACKED)
    parser.add_argument('--resume',action='store_true',help='reuse verified preparation and resume recorded checkpoints')
    args=parser.parse_args()
    ROOT,FROZEN,PACKED=args.out_dir.resolve(),args.frozen_dir.resolve()/'manifest.json',args.packed_dir.resolve()
    settings=json.loads(args.config.read_text(encoding='utf-8'))
    if settings['format']!='p9.full.1': raise ValueError('unknown full training config')
    ROOT.mkdir(parents=True,exist_ok=True)
    paths={'outputs':str(ROOT),'frozen':str(FROZEN),'packed':str(PACKED)}
    if (ROOT/'plan.json').exists():
        plan=json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
        if not args.resume or plan['config_sha256']!=file_sha(args.config) or plan['paths']!=paths:
            raise ValueError('existing round needs --resume with identical configuration and paths')
    else:
        atomic_json(ROOT/'plan.json',{'settings':settings,'config_sha256':file_sha(args.config),'status':'planned','paths':paths})
    try:
        if args.resume and (ROOT/'preflight.json').exists():
            verified=json.loads((ROOT/'preflight.json').read_text(encoding='utf-8'))
            if verified['gate']!='PASS' or verified['source_files_sha256']!=_source_hashes():
                raise ValueError('preflight source changed; create a new round')
        else: verified=preflight(settings)
        if args.command=='preflight': return
        if FROZEN.exists():
            if not args.resume: raise ValueError('frozen inputs already exist; use recorded resume')
            frozen=FROZEN
            body=json.loads(frozen.read_text(encoding='utf-8'))
            if body['source']['preflight']['sha256']!=file_sha(ROOT/'preflight.json'):
                raise ValueError('frozen inputs differ from recorded preflight')
        else:
            emit({'stage':'freezing_start','draw_sets':settings['freeze']['epochs'],'workers':settings['freeze']['workers']})
            frozen=freeze_p8(FROZEN.parent,eval_dir=REPO/'data/processed/eval_samples',folds_path=REPO/'data/splits/folds.json',
                         eval_spec_path=REPO/'data/splits/eval_masks.json',room_cache_dir=REPO/'data/processed/rooms',
                         **settings['freeze'],progress=emit,provenance={'sha256':file_sha(ROOT/'preflight.json'),'rooms':len(verified['rooms'])})
            for room,entry in verified['rooms'].items():
                if file_sha(REPO/'data/processed/rooms'/f'{room}.npz')!=entry['cache_sha256']:
                    raise ValueError('room cache changed during preparation')
        if not (PACKED/'manifest.json').exists():
            emit({'stage':'packing_start'})
            pack_freeze(frozen,PACKED,progress=emit)
        config=dict(settings['training'])
        config['k_range'],config['eval_k']=tuple(config['k_range']),tuple(config['eval_k'])
        config['packed_dir']=str(PACKED)
        trained=[]
        for variant in settings['variants']:
            emit({'stage':'training_start','variant':variant})
            existing=ROOT/variant/'run.json'
            if args.resume and existing.exists() and json.loads(existing.read_text(encoding='utf-8'))['status']=='complete':
                run=json.loads(existing.read_text(encoding='utf-8'))
                expected=FrozenAudioDataset(frozen,'val',k=8).checksum
                if run['frozen_checksum']!=expected or run['source_files_sha256']!=_source_hashes() or file_sha(ROOT/variant/'best.pt')!=run['checkpoint_sha256']:
                    raise ValueError('completed run identity changed')
            else:
                run=run_fusion(variant,frozen,ROOT/variant,FusionTrainConfig(**config),settings['model'],
                               progress=lambda row,v=variant:emit({'stage':'training','variant':v,**row}),resume=args.resume and existing.exists())
            trained.append({'variant':variant,'best_epoch':run['best_epoch'],'checkpoint_sha256':run['checkpoint_sha256'],
                            'validation_score':run['best_validation_score'],'parameters':run['parameter_count']})
        selection={'status':'locked','models':trained,'selected_variant':sorted(trained,key=lambda r:(-r['validation_score'],r['variant']))[0]['variant'],
                   'rule':'best room-averaged mean occupancy/boundary validation IoU; no test-based choice',
                   'frozen_checksum':json.loads(frozen.read_text(encoding='utf-8'))['checksum']}
        atomic_json(ROOT/'selection.json',selection)
        lock=file_sha(ROOT/'selection.json')
        emit({'stage':'selection_locked','selected_variant':selection['selected_variant']})
        scores={}
        for variant in settings['variants']:
            seed_everything(config['seed'])
            model,payload=load_fusion_checkpoint(ROOT/variant/'best.pt',selection['frozen_checksum'])
            model.to('cuda')
            scores[variant]={}
            for k in config['eval_k']:
                ds=FrozenAudioDataset(frozen,'test',k=k,packed_dir=PACKED)
                metrics,rows=evaluate(model,ds,FusionTrainConfig(**config),torch.device('cuda'),keep_rows=True)
                balanced=room_metrics(rows)
                scores[variant][str(k)]={'sample_weighted':metrics,'room_weighted':balanced}
                atomic_json(ROOT/variant/f'final_test_K{k}.json',{'frozen_checksum':ds.checksum,'checkpoint_sha256':next(t['checkpoint_sha256'] for t in trained if t['variant']==variant),'rows':rows})
            emit({'stage':'final_test_complete','variant':variant,'room_occupancy_K8':scores[variant]['8']['room_weighted']['occupancy_iou'],
                  'room_boundary_K8':scores[variant]['8']['room_weighted']['boundary_iou']})
        if file_sha(ROOT/'selection.json')!=lock: raise ValueError('selection changed during test scoring')
        results={'status':'scored','selection':selection,'selection_sha256':lock,'scores':scores,
                 'scope':'one full fold, one seed; prior use of fold-0 test pool; no general acoustic-gain claim'}
        atomic_json(ROOT/'results.json',results)
        from report_full_fold import build_report
        report=build_report(ROOT,frozen,PACKED,settings,scores,selection,emit=emit)
        results.update(status='complete',prediction_gate=report['gate'],prediction_count=report['prediction_count'])
        atomic_json(ROOT/'results.json',results)
        emit({'stage':'complete','selected_variant':selection['selected_variant']})
    except BaseException as error:
        emit({'stage':'failed','error':type(error).__name__+': '+str(error)})
        raise


if __name__=='__main__': main()
