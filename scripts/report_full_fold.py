"""Contract-checked full-round probability exports and local review page."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from echospace.contract_v0 import validate_prediction
from echospace.data.evalmasks import read_manifest
from echospace.training.audio_frozen import FrozenAudioDataset
from echospace.training.fusion_runner import load_fusion_checkpoint, _source_hashes
from echospace.training.packed import file_sha
from echospace.training.runner import atomic_json,seed_everything
from echospace.training.demo import paired_group_delta


def build_report(root,frozen,packed,settings,scores,selection,emit=print):
    root,frozen=Path(root),Path(frozen)
    out=root/'review'
    if out.exists() and any(out.iterdir()): raise ValueError('review destination must be empty')
    out.mkdir(parents=True,exist_ok=True)
    ds=FrozenAudioDataset(frozen,'test',k=8,packed_dir=packed)
    records=read_manifest(REPO/'data/processed/eval_samples')
    chosen=sorted(np.random.default_rng(0).choice(len(ds),min(6,len(ds)),replace=False).tolist())
    images={}
    report={'gate':'running','frozen_checksum':ds.checksum,'source':ds.manifest['source'],
            'selection':selection,'scores':scores,'prediction_count':0,'cpu_replay_errors':{},
            'visual_sample_ids':[ds.entries[i]['metadata']['sample_id'] for i in chosen],
            'scope':'one fold, one seed; previously used fold-0 test pool; not the 75-run study'}
    for variant in settings['variants']:
        run=json.loads((root/variant/'run.json').read_text(encoding='utf-8'))
        if run['source_files_sha256']!=_source_hashes() or run['frozen_checksum']!=ds.checksum:
            raise ValueError('report source/frozen identity changed')
        sha=file_sha(root/variant/'best.pt')
        if sha!=run['checkpoint_sha256']: raise ValueError('checkpoint checksum mismatch')
        seed_everything(settings['training']['seed'])
        model,payload=load_fusion_checkpoint(root/variant/'best.pt',ds.checksum)
        model.to('cuda').eval()
        export=out/'predictions'/variant
        export.mkdir(parents=True)
        entries=[]
        offset=0
        with torch.inference_mode():
            for batch in DataLoader(ds,batch_size=settings['training']['batch_size'],shuffle=False):
                output=model({k:v.to('cuda') for k,v in batch.items()})
                for j in range(len(batch['k'])):
                    index=offset+j
                    entry=ds.entries[index]
                    sid=entry['metadata']['sample_id']
                    arrays={f'{h}_probability':output[f'{h}_logits'][j].float().sigmoid().cpu().numpy() for h in ('occupancy','boundary')}
                    validate_prediction(records[sid],arrays)
                    path=export/f'{index:06d}.npz'
                    np.savez_compressed(path,**arrays)
                    entries.append({'sample_id':sid,'grid':entry['metadata']['grid'],'spatial_transform':entry['metadata']['spatial_transform'],
                                    'source_sample_sha256':entry['sha256'],'actual_k':int(batch['k'][j]),'path':path.name,'sha256':file_sha(path)})
                    if index in chosen: images[variant,index]=arrays
                offset+=len(batch['k'])
        atomic_json(export/'manifest.json',{'frozen_checksum':ds.checksum,'checkpoint_sha256':sha,'variant':variant,'requested_k':8,'predictions':entries})
        model.to('cpu')
        maximum=0.0
        with torch.inference_mode():
            for index in sorted({0,len(ds)//2,len(ds)-1}):
                output=model({k:v[None] for k,v in ds[index].items()})
                with np.load(export/f'{index:06d}.npz',allow_pickle=False) as saved:
                    maximum=max(maximum,max(float(np.max(np.abs(output[f'{h}_logits'][0].float().sigmoid().numpy()-saved[f'{h}_probability']))) for h in ('occupancy','boundary')))
        if maximum>1e-4: raise ValueError('CPU checkpoint replay exceeds tolerance')
        report['cpu_replay_errors'][variant]=maximum
        report['prediction_count']+=len(entries)
        emit({'stage':'prediction_export_verified','variant':variant,'predictions':len(entries),'cpu_replay_error':maximum})
    rows={v:json.loads((root/v/'final_test_K8.json').read_text(encoding='utf-8'))['rows'] for v in settings['variants']}
    report['paired_occupancy_deltas']={f'D_minus_{v}':paired_group_delta(rows[v],rows['D']) for v in ('A','P')}
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for variant in settings['variants']:
        run=json.loads((root/variant/'run.json').read_text(encoding='utf-8'))
        axes[0].plot([r['epoch'] for r in run['history']],[r['train_loss'] for r in run['history']],label=variant)
        axes[1].plot([r['epoch'] for r in run['history']],[r['selection_score'] for r in run['history']],label=variant)
    for ax,title in zip(axes,('Training masked loss','Balanced room-averaged validation IoU')):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=130);plt.close(fig)
    for page in range(0,len(chosen),2):
        for head in ('occupancy','boundary'):
            fig,axes=plt.subplots(2,5,figsize=(14,7),squeeze=False)
            for row,index in enumerate(chosen[page:page+2]):
                item=ds[index]
                axes[row,0].imshow(item['observed_free']+2*item['observed_wall'],origin='lower',vmin=0,vmax=2)
                axes[row,0].set_title('Observed')
                axes[row,1].imshow(item['target_'+head],origin='lower',vmin=0,vmax=1);axes[row,1].set_title('Target '+head)
                for column,variant in enumerate(settings['variants'],2):
                    axes[row,column].imshow(images[variant,index][head+'_probability'],origin='lower',vmin=0,vmax=1);axes[row,column].set_title(variant)
                for ax in axes[row]: ax.contour(item['observed_cells'],levels=[.5],colors='red',linewidths=.4);ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle('A/P/D holdout probabilities, K=8; red=observed outline')
            fig.tight_layout(rect=(0,0,1,.94),h_pad=2.5);fig.savefig(out/f'{head}_{page//2+1}.png',dpi=120);plt.close(fig)
    report['gate']='PASS'
    atomic_json(out/'summary.json',report)
    table=''.join(f"<tr><td>{v}</td><td>{scores[v]['8']['room_weighted']['occupancy_iou']:.3f}</td><td>{scores[v]['8']['room_weighted']['boundary_iou']:.3f}</td></tr>" for v in settings['variants'])
    gallery=''.join(f'<img src="occupancy_{i}.png" alt="Occupancy {i}"><img src="boundary_{i}.png" alt="Boundary {i}">' for i in range(1,(len(chosen)+1)//2+1))
    document=('<!doctype html><html lang="en"><meta charset="utf-8"><title>Full-fold training review</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:20px}img{width:100%}td,th{padding:12px;text-align:left}</style>'
              '<h1>A/P/D training review</h1><p>Geometry, positions control and EchoFusion. Checkpoints and model selection use validation only. '
              'One fold, one seed; the fold-0 test pool has been evaluated previously. These results do not establish general acoustic gain.</p>'
              '<table><tr><th>Model</th><th>Room occupancy IoU, K=8</th><th>Room boundary IoU, K=8</th></tr>'+table+'</table>'
              '<img src="learning_curves.png" alt="Learning curves"><h2>Score-independent held-out examples</h2>'
              '<p>Raw probabilities. Measured geometry remains separate.</p>'+gallery+'<p><a href="summary.json">Full results and provenance</a></p></html>')
    (out/'index.html').write_text(document,encoding='utf-8')
    return report
