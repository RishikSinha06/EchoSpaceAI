"""P11 C/S controls: refuse to train until the preceding study is fully frozen/scored."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from echospace.evaluation.protocol import require_finished_study, verify_study_coverage, sha
from echospace.training.audio_frozen import FrozenAudioDataset
from echospace.training.fusion_runner import FusionTrainConfig, run_fusion, evaluate, load_fusion_checkpoint
from echospace.training.runner import atomic_json
from echospace.training.demo import room_metrics
import torch

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,default=REPO)
    ap.add_argument('--config',type=Path,default=REPO/'configs/p11_stage2.json')
    ap.add_argument('--output',type=Path,default=REPO/'artifacts/p11_cs')
    ap.add_argument('--mode',choices=['preflight','train'],default='preflight')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();source=args.source_root.resolve();out=args.output.resolve()
    settings=json.loads(args.config.read_text());stage=source/settings['source_study']
    # This gate runs before making directories or starting GPU work.
    locked=require_finished_study(stage)
    parent=json.loads((source/settings['source_config']).read_text())
    verify_study_coverage(locked,parent)
    if sha(source/settings['source_config'])!=locked['config_sha256']:raise ValueError('source protocol changed')
    plan={'format':'p11.cs.1','parent_selection_sha256':sha(stage/'selection.json'),
          'parent_results_sha256':sha(stage/'results.json'),'config_sha256':sha(args.config),
          'variants':settings['variants'],'folds':settings['folds'],'seeds':settings['seeds'],
          'source_hashes':{str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'src/echospace').rglob('*.py'))+[Path(__file__)]}}
    jobs=[]
    for fold in settings['folds']:
        frozen=source/('data/cache/p9_full_fold0/manifest.json' if fold==0 else f'data/cache/dual_study_fold{fold}/manifest.json')
        packed=source/('data/cache/p9_full_fold0_packed' if fold==0 else f'data/cache/dual_study_fold{fold}_packed')
        ds=FrozenAudioDataset(frozen,'val',k=8,packed_dir=packed)
        parents=[e for e in locked['models'] if e['fold']==fold and e['name']=='echo_D']
        if {e['seed'] for e in parents}!=set(settings['seeds']) or any(e['frozen_checksum']!=ds.checksum for e in parents):raise ValueError('fold identity mismatch')
        jobs.extend((fold,v,s,frozen,packed) for v in settings['variants'] for s in settings['seeds'])
    print(json.dumps({'status':'preflight_passed','fits':len(jobs),'parent_selection_sha256':plan['parent_selection_sha256']}))
    if args.mode=='preflight':return
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():
        if not args.resume or json.loads((out/'plan.json').read_text())!=plan:raise ValueError('existing study needs matching --resume')
    else:
        if any(out.iterdir()):raise ValueError('destination must be empty')
        atomic_json(out/'plan.json',plan)
    entries=[]
    for fold,variant,seed,frozen,packed in jobs:
        cfg=FusionTrainConfig(**dict(parent['training'],seed=seed,packed_dir=str(packed),evaluate_test=False))
        destination=out/'full'/f'fold{fold}'/variant/f'seed{seed}'
        def progress(row):
            status={'stage':'training','fold':fold,'variant':variant,'seed':seed,**row}
            atomic_json(out/'status.json',status)
            print(json.dumps(status),flush=True)
            if (out/'STOP_AFTER_EPOCH').exists():raise KeyboardInterrupt('P11 stopped after checkpointed epoch')
        if (out/'STOP_AFTER_EPOCH').exists():raise ValueError('remove STOP_AFTER_EPOCH before explicit resume')
        record=run_fusion(variant,frozen,destination,cfg,parent['backbone'],resume=args.resume,progress=progress)
        entries.append({'fold':fold,'variant':variant,'seed':seed,'checkpoint_sha256':record['checkpoint_sha256'],
                        'frozen_checksum':record['frozen_checksum'],'path':str(destination.relative_to(out)/'best.pt')})
    lock={'status':'locked','models':entries,'plan_sha256':sha(out/'plan.json')}
    if (out/'selection.json').exists() and json.loads((out/'selection.json').read_text())!=lock:raise ValueError('selection changed')
    atomic_json(out/'selection.json',lock);lock_sha=sha(out/'selection.json')
    # All C/S fits lock before any C/S test scoring.
    result={}
    for entry in entries:
        fold=entry['fold'];variant=entry['variant'];seed=entry['seed']
        frozen=source/('data/cache/p9_full_fold0/manifest.json' if fold==0 else f'data/cache/dual_study_fold{fold}/manifest.json')
        packed=source/('data/cache/p9_full_fold0_packed' if fold==0 else f'data/cache/dual_study_fold{fold}_packed')
        checkpoint=out/entry['path']
        if sha(checkpoint)!=entry['checkpoint_sha256']:raise ValueError('selected checkpoint changed')
        model,_=load_fusion_checkpoint(checkpoint,entry['frozen_checksum']);model.cuda().eval()
        cfg=FusionTrainConfig(**dict(parent['training'],seed=seed,packed_dir=str(packed),evaluate_test=False))
        for k in settings['evaluation_k']:
            ds=FrozenAudioDataset(frozen,'test',k=k,packed_dir=packed)
            metrics,rows=evaluate(model,ds,cfg,torch.device('cuda'),keep_rows=True)
            record={'selection_sha256':lock_sha,'checkpoint_sha256':entry['checkpoint_sha256'],'metrics':metrics,'rows':rows}
            atomic_json(checkpoint.parent/f'test_K{k}.json',record)
            result.setdefault(variant,{}).setdefault(str(seed),{}).setdefault(str(k),[]).extend(rows)
        del model
    if sha(out/'selection.json')!=lock_sha:raise ValueError('selection changed while scoring')
    atomic_json(out/'results.json',{'status':'internal_controls_complete','selection_sha256':lock_sha,'production_ready':False,
                'scores':{v:{s:{k:room_metrics(rows) for k,rows in ks.items()} for s,ks in seeds.items()} for v,seeds in result.items()}})

if __name__=='__main__':main()
