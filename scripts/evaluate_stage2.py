"""P11 offline audit of an already locked original P9 study (CPU by default)."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from echospace.training.audio_frozen import FrozenAudioDataset
from echospace.training.fusion_runner import load_fusion_checkpoint
from echospace.training.quality import boundary_quality
from echospace.training.runner import atomic_json, seed_everything
from echospace.evaluation.controls import ControlledDataset, CONDITIONS
from echospace.evaluation.protocol import verify_lock, choose_temperature, probability, sha
from echospace.evaluation.stats import room_summary, paired_metrics

from echospace.evaluation.engine import calibrate, score

def slices(rows):
    return {kind:{value:room_summary([r for r in rows if r[kind]==value])
                  for value in sorted({r[kind] for r in rows})}
            for kind in ('coverage_bin','occlusion_type')}

def write_review(out,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(report['summaries']);values=[report['summaries'][k]['metrics']['occupancy_iou'] for k in labels]
    fig,ax=plt.subplots(figsize=(11,5));ax.bar(labels,values);ax.set_ylim(0,1);ax.set_ylabel('Equal-room missing-region occupancy IoU');ax.tick_params(axis='x',rotation=70);fig.tight_layout();fig.savefig(out/'conditions.png',dpi=130);plt.close(fig)
    text=['# P11 Stage II diagnostic report','',
          '**Exploratory original P9 fold 0 / seed 0. Not the unfinished dual study.**','',
          'Threshold 0.5 retained; no test-based checkpoint or threshold selection. Calibration temperatures fit only on validation.','',
          '| Model / condition / K | Rooms | Occupancy IoU | Hidden-wall F1 @ 0.20 m | Boundary distance (m) |',
          '| --- | ---: | ---: | ---: | ---: |']
    for key,s in report['summaries'].items():
        m=s['metrics'];text.append(f"| {key} | {s['rooms']} | {m['occupancy_iou']:.4f} | {m['boundary_f1_20cm']:.4f} | {m['boundary_chamfer_m']:.4f} |")
    text+=['','## Interpretation boundaries','',
           'Swapped audio uses different physical-room-group waveforms with matching actual K while retaining original poses/masks. Donor reuse is allowed and recorded; this is a controlled mismatch diagnostic, not a balanced permutation or a deployment condition.',
           'Zero audio retains valid masks and poses; it is not K=0. Timing shifts are exactly 3 samples (0.1875 ms) with zero padding, not circular shifts. Noise is 20 dB per waveform; gain is +/-6 dB after fixed peak preprocessing. Pose stress is bounded uniform +/-5 cm per coordinate, rejected outside observed free cells.',
           'Window40/60 preserve emission time and tensor length with trailing zero padding. These are inference truncation checks, not retrained window sweeps. Confidence calibration is not an uncertainty/rejection guarantee.',
           'Primary-plan slice is K=2, mid coverage, viewpoint/missing_wall. Its actual masks may differ from exact 50% wall coverage; this retrospective slice does not restore preregistration. Current training used random K=1..8 and K=8 validation selection. Previously explored rooms do not constitute independent external validation.',
           'Boundary distance uses a grid-diagonal penalty if exactly one set is empty; two empty sets have F1=1/distance=0. All metrics exclude observed and invalid cells. Paired intervals resample room groups, not masks; per-seed results must not be treated as independent room populations.',
           '','## Pending release gates','']
    text+=['- '+x for x in report['pending']]
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,default=REPO)
    ap.add_argument('--output',type=Path,default=REPO/'artifacts/p11_original_fold0')
    ap.add_argument('--ks',type=int,nargs='+',default=[1,2,4,8],choices=[1,2,4,8])
    ap.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    ap.add_argument('--batch-size',type=int,default=16)
    args=ap.parse_args()
    if len(set(args.ks))!=len(args.ks) or len(set(args.conditions))!=len(args.conditions):raise ValueError('duplicate conditions/K')
    if 2 not in args.ks or 'correct' not in args.conditions:raise ValueError('K=2 and correct condition are required')
    if args.batch_size<1:raise ValueError('batch size must be positive')
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('output must be empty; preserve earlier evaluations')
    source=args.source_root.resolve();study=source/'artifacts/p9_full_fold0'
    frozen=source/'data/cache/p9_full_fold0/manifest.json';packed=source/'data/cache/p9_full_fold0_packed'
    paths=[study/v/'best.pt' for v in ('A','P','D')]
    seed_everything(0);torch.set_num_threads(2)
    ds=FrozenAudioDataset(frozen,'val',k=2,packed_dir=packed)
    identity=verify_lock(study/'selection.json',paths,ds.checksum)
    # All identities verified before any held-out scoring; no checkpoint writes.
    out.mkdir(parents=True,exist_ok=True)
    plan={'format':'p11.evaluation.1','frozen_checksum':ds.checksum,**identity,
          'ks':args.ks,'conditions':args.conditions,'threshold':.5,'seed':0,'calibration_k':2,
          'source_hashes':{str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'src/echospace').rglob('*.py'))+[Path(__file__)]}}
    atomic_json(out/'plan.json',plan)
    all_rows={};calibration={}
    for variant,path in zip(('A','P','D'),paths):
        model,payload=load_fusion_checkpoint(path,ds.checksum);model.cpu().eval()
        if payload['model']['variant']!=variant:raise ValueError('checkpoint variant mismatch')
        print(f'Calibrating {variant} on validation only',flush=True)
        calibration[variant]=calibrate(model,ds,args.batch_size)
        atomic_json(out/f'calibration_{variant}.json',calibration[variant])
        for k in args.ks:
            base=FrozenAudioDataset(frozen,'test',k=k,packed_dir=packed)
            conditions=args.conditions if variant=='D' and k==2 else ['correct']
            for condition in conditions:
                controlled=ControlledDataset(base,condition)
                key=f'{variant}/{condition}/K{k}'
                rows=score(model,controlled,args.batch_size,calibration[variant]);all_rows[key]=rows
                atomic_json(out/f'{variant}_{condition}_K{k}.json',{'rows':rows,'donors':controlled.donor_manifest(),'identity':identity})
                print(json.dumps({'stage':'scored','condition':key,**room_summary(rows)['metrics']}),flush=True)
        del model
    comparisons={}
    for k in args.ks:
        for reference in ('A','P'):
            comparisons[f'D_minus_{reference}/K{k}']=paired_metrics(all_rows[f'{reference}/correct/K{k}'],all_rows[f'D/correct/K{k}'])
    for condition in args.conditions:
        if condition!='correct': comparisons[f'D_{condition}_minus_correct/K2']=paired_metrics(all_rows['D/correct/K2'],all_rows[f'D/{condition}/K2'])
    primary={v:[r for r in all_rows[f'{v}/correct/K2'] if r['coverage_bin']=='mid' and r['occlusion_type'] in ('viewpoint','missing_wall')] for v in ('A','P','D')}
    primary_effects={f'D_minus_{v}':paired_metrics(primary[v],primary['D']) for v in ('A','P')} if primary['D'] else {}
    failure=sorted(all_rows['D/correct/K2'],key=lambda r:(r['occupancy_iou'],r['sample_id']))[:12]
    report={'status':'diagnostics_complete','production_ready':False,'plan_sha256':sha(out/'plan.json'),
            'scope':'exploratory original P9 fold0 seed0; previously exposed test population',
            'summaries':{key:room_summary(rows) for key,rows in all_rows.items()},
            'strata':{key:slices(rows) for key,rows in all_rows.items() if '/correct/' in key},
            'paired_effects':comparisons,'primary_plan_slice':{v:room_summary(rs) for v,rs in primary.items()},
            'primary_plan_effects':primary_effects,'negative_cases':failure,'calibration':calibration,
            'pending':['finish and freeze current dual study','full C/S controls after Stage I lock',
                       'apply matched controls/reporting to final dual-study checkpoints',
                       'retrained window sweeps (40/60/80 ms)','independent measured-room audit and evaluation',
                       'validated uncertainty/rejection behavior and release/rollback package']}
    verify_lock(study/'selection.json',paths,ds.checksum)
    if sha(study/'selection.json')!=identity['selection_sha256']:raise ValueError('selection changed during scoring')
    if any(sha(REPO/p)!=digest for p,digest in plan['source_hashes'].items()):raise ValueError('evaluation source changed while scoring')
    atomic_json(out/'summary.json',report);write_review(out,report)
    print('P11_DIAGNOSTICS_COMPLETE',out,flush=True)

if __name__=='__main__':main()
