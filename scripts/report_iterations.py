"""Evaluate locked small-search settings once and export a verified comparison."""
import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from echospace.contract_v0 import validate_prediction
from echospace.data.evalmasks import read_manifest
from echospace.training.audio_frozen import FrozenAudioDataset
from echospace.training.demo import room_metrics, paired_group_delta
from echospace.training.fusion_runner import FusionTrainConfig, evaluate, load_fusion_checkpoint, _source_hashes
from echospace.training.runner import atomic_json, seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, default=REPO/'artifacts/p9_iterations/study.json')
    p.add_argument('--frozen', type=Path, default=REPO/'data/cache/p9_iterations/manifest.json')
    p.add_argument('--out-dir', type=Path, default=REPO/'artifacts/p9_iterations/final')
    args = p.parse_args()
    study = json.loads(args.study.read_text(encoding='utf-8'))
    if study['status'] != 'selection_locked':
        raise ValueError('settings must be locked before holdout evaluation')
    means = study['confirmation_mean_scores']
    if sorted(means, key=lambda t: (-means[t], t))[0] != study['winner']:
        raise ValueError('winner differs from validation-only selection')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError('final report must use an empty directory')
    # Validate every trial and all validation scores BEFORE opening any test file.
    runs = {}
    for trial in study['trials']:
        root = REPO/trial['path']
        run = json.loads((root/'run.json').read_text(encoding='utf-8'))
        if run['status'] != 'complete' or not run['test_evaluation_deferred'] or run['test_metrics']:
            raise ValueError('trial read test data or did not complete')
        if run['source_files_sha256'] != _source_hashes() or run['frozen_checksum'] != study['frozen_checksum']:
            raise ValueError('trial source or frozen identity mismatch')
        if hashlib.sha256((root/'best.pt').read_bytes()).hexdigest() != trial['checkpoint_sha256']:
            raise ValueError('selected checkpoint checksum mismatch')
        best = next(row for row in run['history'] if row['epoch']==run['best_epoch'])
        if best['selection_score'] != trial['score']:
            raise ValueError('validation score differs from study record')
        runs[trial['trial'], trial['seed']] = run
    for name in means:
        scores = [t['score'] for t in study['trials'] if t['trial']==name]
        if len(scores)!=2 or sum(scores)/2!=means[name]:
            raise ValueError('confirmation mean mismatch')
    lock_sha = hashlib.sha256(args.study.read_bytes()).hexdigest()
    torch.set_num_threads(2)
    ds = FrozenAudioDataset(args.frozen, 'test', k=8)
    if ds.checksum != study['frozen_checksum']:
        raise ValueError('holdout frozen mismatch')
    # Array integrity checks are separate from evaluating quality.
    for split in ('train','val','test'):
        part = FrozenAudioDataset(args.frozen, split, k=8)
        for i in range(len(part)):
            part[i]
    ids = [e['metadata']['sample_id'] for e in ds.entries]
    records = read_manifest(REPO/'data/processed/eval_samples')
    chosen = sorted(np.random.default_rng(0).choice(len(ds), min(6,len(ds)), replace=False).tolist())
    selected = [('A_reference',0), (study['winner'],0), (study['winner'],1)]
    selected = list(dict.fromkeys(selected))
    summary = {'format':'p9.iteration_report.1','gate':'running','winner':study['winner'],'deploy_seed':0,
               'study_sha256_before_holdout':lock_sha,'frozen_checksum':ds.checksum,
               'scope':ds.manifest['source']['holdout_scope'],'source':ds.manifest['source'],
               'validation_trials':study['trials'],'confirmation_mean_scores':means,'test':{},
               'prediction_count':0,'visual_sample_ids':[ids[i] for i in chosen]}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    images, test_rows = {}, {}
    for name, seed in selected:
        trial = next(t for t in study['trials'] if t['trial']==name and t['seed']==seed)
        root = REPO/trial['path']
        run = runs[name,seed]
        seed_everything(seed)
        model,payload = load_fusion_checkpoint(root/'best.pt',ds.checksum)
        if payload['epoch'] != trial['best_epoch']:
            raise ValueError('checkpoint epoch mismatch')
        model.to('cuda').eval()
        config = dict(run['config'])
        config['k_range'],config['eval_k']=tuple(config['k_range']),tuple(config['eval_k'])
        metrics, rows = evaluate(model, ds, FusionTrainConfig(**config), torch.device('cuda'), keep_rows=True)
        label = f'{name}_seed{seed}'
        test_rows[label] = rows
        balanced = room_metrics(rows)
        summary['test'][label] = {'sample_weighted':metrics,'room_weighted':balanced,
                                 'mean_head_iou':(balanced['occupancy_iou']+balanced['boundary_iou'])/2,
                                 'checkpoint_sha256':trial['checkpoint_sha256'], 'cpu_replay_max_probability_error':0.0}
        atomic_json(args.out_dir/f'{label}_test_rows.json',{'checkpoint_sha256':trial['checkpoint_sha256'],
                    'frozen_checksum':ds.checksum,'rows':rows})
        export = args.out_dir/'predictions'/label
        export.mkdir(parents=True)
        entries=[]
        offset=0
        with torch.inference_mode():
            for batch in DataLoader(ds,batch_size=16,shuffle=False):
                output=model({key:value.to('cuda') for key,value in batch.items()})
                for j in range(len(batch['k'])):
                    index=offset+j
                    arrays={f'{head}_probability':output[f'{head}_logits'][j].float().sigmoid().cpu().numpy()
                            for head in ('occupancy','boundary')}
                    validate_prediction(records[ids[index]],arrays)
                    path=export/f'{index:06d}.npz'
                    np.savez_compressed(path,**arrays)
                    entry=ds.entries[index]
                    entries.append({'sample_id':ids[index],'source_sample_sha256':entry['sha256'],
                                    'grid':entry['metadata']['grid'],'spatial_transform':entry['metadata']['spatial_transform'],
                                    'actual_k':int(batch['k'][j]),'path':path.name,
                                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                    if index in chosen:
                        images[label,index]=arrays
                offset+=len(batch['k'])
        atomic_json(export/'manifest.json',{'frozen_checksum':ds.checksum,'checkpoint_sha256':trial['checkpoint_sha256'],
                                           'requested_k':8,'predictions':entries})
        # Replay three deterministic examples on CPU from the same checkpoint.
        model.to('cpu')
        with torch.inference_mode():
            for index in (0,len(ds)//2,len(ds)-1):
                output=model({key:value[None] for key,value in ds[index].items()})
                with np.load(export/f'{index:06d}.npz',allow_pickle=False) as saved:
                    error=max(float(np.max(np.abs(output[f'{head}_logits'][0].float().sigmoid().numpy()-saved[f'{head}_probability'])))
                              for head in ('occupancy','boundary'))
                if error>1e-4:
                    raise ValueError('CPU checkpoint replay exceeds numerical tolerance')
                summary['test'][label]['cpu_replay_max_probability_error']=max(error,summary['test'][label]['cpu_replay_max_probability_error'])
        summary['prediction_count']+=len(entries)
        print(json.dumps({'model':label,'room_occupancy_iou':balanced['occupancy_iou'],
                          'room_boundary_iou':balanced['boundary_iou']}),flush=True)
    winner_label=f"{study['winner']}_seed0"
    visual_label = winner_label if winner_label!='A_reference_seed0' else 'A_reference_seed1'
    summary['visual_comparison_labels'] = ['A_reference_seed0', visual_label]
    summary['winner_minus_reference_occupancy']=paired_group_delta(test_rows['A_reference_seed0'],test_rows[winner_label])
    balanced_rows = lambda rows: [dict(r, occupancy_iou=(r['occupancy_iou']+r['boundary_iou'])/2) for r in rows]
    summary['winner_minus_reference_mean_head_iou'] = paired_group_delta(
        balanced_rows(test_rows['A_reference_seed0']), balanced_rows(test_rows[winner_label]))
    summary['winner_minus_reference_mean_head_iou']['metric'] = 'equal-room mean of occupancy and boundary IoU'
    for page in range(0,len(chosen),2):
        for head in ('occupancy','boundary'):
            fig,axes=plt.subplots(2,4,figsize=(11,7),squeeze=False)
            for row,index in enumerate(chosen[page:page+2]):
                item=ds[index]
                axes[row,0].imshow(item['observed_free']+2*item['observed_wall'],origin='lower',vmin=0,vmax=2)
                axes[row,0].set_title('Observed geometry')
                axes[row,1].imshow(item[f'target_{head}'],origin='lower',vmin=0,vmax=1)
                axes[row,1].set_title(f'Target {head}')
                for column,label in enumerate(('A_reference_seed0',visual_label),2):
                    axes[row,column].imshow(images[label,index][f'{head}_probability'],origin='lower',vmin=0,vmax=1)
                    axes[row,column].set_title(label,fontsize=9)
                for ax in axes[row]:
                    ax.contour(item['observed_cells'],levels=[.5],colors='red',linewidths=.4)
                    ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle('Small training iterations: holdout, K=8, red=observation outline')
            fig.tight_layout(rect=(0,0,1,.94),h_pad=2.5)
            fig.savefig(args.out_dir/f'{head}_{page//2+1}.png',dpi=120);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    for (name,seed),run in runs.items():
        ax.plot([r['epoch'] for r in run['history']],[r['selection_score'] for r in run['history']],label=f'{name}, seed {seed}')
    ax.set(xlabel='Epoch',ylabel='Room-averaged mean occupancy/boundary IoU',title='Validation-only model selection')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(args.out_dir/'validation_curves.png',dpi=130);plt.close(fig)
    if hashlib.sha256(args.study.read_bytes()).hexdigest()!=lock_sha:
        raise ValueError('selection record changed during holdout evaluation')
    summary['gate']='PASS'
    deployment = next(t for t in study['trials'] if t['trial']==study['winner'] and t['seed']==0)
    summary['deployment'] = {'checkpoint_path': deployment['path']+'/best.pt',
                             'checkpoint_sha256': deployment['checkpoint_sha256'],
                             'model': runs[study['winner'],0]['model'], 'seed': 0,
                             'selection': 'locked two-seed mean validation objective; no holdout-based selection'}
    atomic_json(args.out_dir/'deployment.json', summary['deployment'])
    atomic_json(args.out_dir/'summary.json',summary)
    table=''.join(f"<tr><td>{html.escape(label)}</td><td>{r['room_weighted']['occupancy_iou']:.3f}</td>"
                  f"<td>{r['room_weighted']['boundary_iou']:.3f}</td><td>{r['mean_head_iou']:.3f}</td></tr>" for label,r in summary['test'].items())
    gallery=''.join(f'<img src="occupancy_{i}.png" alt="Occupancy {i}"><img src="boundary_{i}.png" alt="Boundary {i}">' for i in range(1,4))
    page=('<!doctype html><html lang="en"><meta charset="utf-8"><title>Small model training iterations</title>'
          '<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:20px;color:#20252b}'
          'img{width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}</style>'
          '<h1>Small model training iterations</h1><p>Six preset trials, two finalists repeated with seed 1. '
          'Settings selected only on validation; seed 0 fixed as deployment checkpoint.</p><p>'
          +html.escape(summary['scope'])+'</p><h2>Final K=8 comparison</h2><table><tr><th>Model</th>'
          '<th>Occupancy IoU</th><th>Boundary IoU</th><th>Mean</th></tr>'+table+'</table>'
          '<h2>Validation curves</h2><img src="validation_curves.png" alt="Validation curves">'
          '<h2>Six score-independent holdout examples</h2><p>Raw probabilities are shown. Measured geometry remains separate.</p>'
          +gallery+'<p><a href="summary.json">Results and provenance</a></p></html>')
    (args.out_dir/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({'gate':'PASS','winner':study['winner'],'predictions':summary['prediction_count']}),flush=True)


if __name__=='__main__':
    main()
