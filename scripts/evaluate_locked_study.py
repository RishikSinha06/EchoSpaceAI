"""P11 matched evaluation of the completed dual study, isolated from its artifacts."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import torch
from echospace.evaluation.protocol import require_finished_study,verify_study_coverage,sha
from echospace.evaluation.controls import ControlledDataset,CONDITIONS
from echospace.evaluation.engine import calibrate,score
from echospace.evaluation.stats import room_summary,paired_metrics
from echospace.training.audio_frozen import FrozenAudioDataset
from echospace.training.fusion_runner import load_fusion_checkpoint
from echospace.training.runner import atomic_json,seed_everything

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,default=REPO)
    ap.add_argument('--study',default='artifacts/dual_study_20261017')
    ap.add_argument('--output',type=Path,default=REPO/'artifacts/p11_dual_evaluation')
    ap.add_argument('--mode',choices=['preflight','evaluate'],default='preflight')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();source=args.source_root.resolve();stage=source/args.study;out=args.output.resolve()
    selection=require_finished_study(stage)
    settings_path=source/'configs/dual_architecture_study.json'
    if sha(settings_path)!=selection['config_sha256']:raise ValueError('parent protocol changed')
    verify_study_coverage(selection,json.loads(settings_path.read_text()))
    # Explicit core names; ablations remain identifiable in the parent study.
    entries=[e for e in selection['models'] if e['name'] in ('echo_A','echo_P','echo_D','wall_P','wall_D')]
    if not entries:raise ValueError('no completed core fits')
    plan={'format':'p11.dual_evaluation.1','selection_sha256':sha(stage/'selection.json'),
          'parent_results_sha256':sha(stage/'results.json'),'ks':[1,2,4,8],'conditions':list(CONDITIONS),
          'threshold':.5,'calibration_k':2,'models':entries,
          'source_hashes':{p.relative_to(REPO).as_posix():sha(p) for p in sorted((REPO/'src/echospace').rglob('*.py'))+[Path(__file__)]}}
    print(json.dumps({'stage':'preflight_passed','fits':len(entries),'selection_sha256':plan['selection_sha256']}),flush=True)
    if args.mode=='preflight':return
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():
        if not args.resume or json.loads((out/'plan.json').read_text())!=plan:raise ValueError('existing output requires matching resume')
    else:
        if any(out.iterdir()):raise ValueError('destination must be empty')
        atomic_json(out/'plan.json',plan)
    plan_sha=sha(out/'plan.json');seed_everything(0);torch.set_num_threads(2)
    accumulated={}
    for entry in entries:
        fold,seed,name=entry['fold'],entry['seed'],entry['name']
        frozen=source/('data/cache/p9_full_fold0/manifest.json' if fold==0 else f'data/cache/dual_study_fold{fold}/manifest.json')
        packed=source/('data/cache/p9_full_fold0_packed' if fold==0 else f'data/cache/dual_study_fold{fold}_packed')
        checkpoint=stage/'full'/f'fold{fold}'/name/f'seed{seed}'/'best.pt'
        if sha(checkpoint)!=entry['checkpoint_sha256']:raise ValueError('locked checkpoint changed')
        ds=FrozenAudioDataset(frozen,'val',k=2,packed_dir=packed)
        if ds.checksum!=entry['frozen_checksum']:raise ValueError('frozen identity mismatch')
        model,_=load_fusion_checkpoint(checkpoint,ds.checksum);model.cpu().eval()
        destination=out/f'fold{fold}'/name/f'seed{seed}';destination.mkdir(parents=True,exist_ok=True)
        calibration_path=destination/'calibration.json'
        if calibration_path.exists():
            c=json.loads(calibration_path.read_text())
            if c['plan_sha256']!=plan_sha:raise ValueError('calibration identity mismatch')
            calibration=c['calibration']
        else:
            calibration=calibrate(model,ds,16)
            atomic_json(calibration_path,{'plan_sha256':plan_sha,'calibration':calibration})
        for k in plan['ks']:
            base=FrozenAudioDataset(frozen,'test',k=k,packed_dir=packed)
            for condition in (CONDITIONS if name in ('echo_D','wall_D') and k==2 else ['correct']):
                path=destination/f'{condition}_K{k}.json'
                if path.exists():
                    saved=json.loads(path.read_text())
                    if saved['plan_sha256']!=plan_sha:raise ValueError('evaluation identity mismatch')
                    rows=saved['rows']
                else:
                    ds=ControlledDataset(base,condition)
                    rows=score(model,ds,16,calibration)
                    atomic_json(path,{'plan_sha256':plan_sha,'rows':rows,'donors':ds.donor_manifest()})
                accumulated.setdefault(str(seed),{}).setdefault(f'{name}/{condition}/K{k}',[]).extend(rows)
                print(json.dumps({'stage':'scored','fold':fold,'seed':seed,'model':name,'condition':condition,'k':k}),flush=True)
        del model
    effects={}
    for seed,records in accumulated.items():
        effects[seed]={}
        for k in plan['ks']:
            for left,right in [('echo_A','echo_D'),('echo_P','echo_D'),('echo_A','wall_D'),('wall_P','wall_D'),('echo_D','wall_D')]:
                effects[seed][f'{right}_minus_{left}/K{k}']=paired_metrics(records[f'{left}/correct/K{k}'],records[f'{right}/correct/K{k}'])
        for name in ('echo_D','wall_D'):
            for condition in CONDITIONS:
                if condition!='correct':effects[seed][f'{name}_{condition}_minus_correct/K2']=paired_metrics(records[f'{name}/correct/K2'],records[f'{name}/{condition}/K2'])
    require_finished_study(stage)
    if sha(stage/'selection.json')!=plan['selection_sha256'] or sha(stage/'results.json')!=plan['parent_results_sha256']:raise ValueError('parent identities changed')
    if any(sha(REPO/p)!=digest for p,digest in plan['source_hashes'].items()):raise ValueError('evaluation source changed')
    atomic_json(out/'summary.json',{'status':'internal_diagnostics_complete','production_ready':False,
                'plan_sha256':plan_sha,'summaries_by_seed':{s:{key:room_summary(rows) for key,rows in records.items()} for s,records in accumulated.items()},
                'paired_effects_by_seed':effects,'scope':'seeds reported separately; overlapping folds do not constitute independent external rooms',
                'pending':['C/S full controls','retrained window sweep','external measured-room validation','validated rejection/release gates']})

if __name__=='__main__':main()
