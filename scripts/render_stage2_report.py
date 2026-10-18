"""Render P11 strata and deterministic negative cases from completed diagnostics."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from echospace.evaluation.protocol import sha,verify_lock
from echospace.training.audio_frozen import FrozenAudioDataset
from echospace.training.fusion_runner import load_fusion_checkpoint
from echospace.training.runner import atomic_json

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=Path,default=REPO);ap.add_argument('--results',type=Path,required=True)
    args=ap.parse_args();root=args.results;source=args.source_root
    report=json.loads((root/'summary.json').read_text());plan=json.loads((root/'plan.json').read_text())
    if sha(root/'plan.json')!=report['plan_sha256']:raise ValueError('plan identity changed')
    if report['status']!='diagnostics_complete':raise ValueError('diagnostics incomplete')
    keys=['A/correct/K2','P/correct/K2','D/correct/K2']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,kind in zip(axes,('coverage_bin','occlusion_type')):
        categories=sorted(report['strata'][keys[0]][kind]);x=np.arange(len(categories))
        for offset,key in enumerate(keys):
            values=[report['strata'][key][kind][c]['metrics']['occupancy_iou'] for c in categories]
            ax.bar(x+(offset-1)*.24,values,width=.24,label=key[0])
        ax.set_xticks(x,categories,rotation=20);ax.set_ylim(0,1);ax.set_title(kind+' (K=2)');ax.set_ylabel('Equal-room missing IoU');ax.legend()
    fig.tight_layout();fig.savefig(root/'coverage_occlusion.png',dpi=130);plt.close(fig)
    # Rank by score, then require at least 20 hidden interior cells for informative figures.
    # The numerical report retains all cases, including empty/tiny hidden targets.
    rows=json.loads((root/'D_correct_K2.json').read_text())['rows'];chosen=[];rooms=set()
    frozen=source/'data/cache/p9_full_fold0/manifest.json';packed=source/'data/cache/p9_full_fold0_packed';study=source/'artifacts/p9_full_fold0'
    ds=FrozenAudioDataset(frozen,'test',k=2,packed_dir=packed)
    indices={e['metadata']['sample_id']:i for i,e in enumerate(ds.entries)}
    for row in sorted(rows,key=lambda r:(r['occupancy_iou'],r['sample_id'])):
        if row['room_id'] in rooms:continue
        item=ds[indices[row['sample_id']]]
        missing=(item['observed_cells']==0)&(item['valid_cells']==1)
        support=int((missing&item['target_occupancy'].bool()).sum())
        if support<20:continue
        chosen.append(dict(row,hidden_interior_cells=support));rooms.add(row['room_id'])
        if len(chosen)==6:break
    paths=[study/v/'best.pt' for v in ('A','P','D')];identity=verify_lock(study/'selection.json',paths,ds.checksum)
    if identity['selection_sha256']!=plan['selection_sha256']:raise ValueError('checkpoint selection changed')
    torch.set_num_threads(2);models={v:load_fusion_checkpoint(study/v/'best.pt',ds.checksum)[0].cpu().eval() for v in ('A','D')}
    indices={e['metadata']['sample_id']:i for i,e in enumerate(ds.entries)}
    cmap=ListedColormap(['#142632','#315b62','#20c8c1','#ffba66','#f4f1de'])
    for page in range(0,len(chosen),3):
        fig,axes=plt.subplots(3,4,figsize=(12,9))
        for row_number,row in enumerate(chosen[page:page+3]):
            item=ds[indices[row['sample_id']]];batch={k:v[None] for k,v in item.items()}
            free=item['observed_free'].numpy().astype(bool);wall=item['observed_wall'].numpy().astype(bool)
            observation=np.zeros((64,64));observation[free]=1;observation[wall]=2
            images=[observation]
            with torch.inference_mode():
                for v in ('A','D'):
                    pred=models[v](batch)['occupancy_logits'][0].sigmoid().numpy()>=.5
                    image=np.where(pred,3,0);image[free]=1;image[wall]=2;images.append(image)
            images.append(np.where(item['target_occupancy'].numpy(),4,0))
            for col,(image,title) in enumerate(zip(images,['Observation','A','D','Truth'])):
                ax=axes[row_number,col];ax.imshow(image,cmap=cmap,vmin=0,vmax=4,origin='upper',interpolation='nearest');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title if row_number==0 else '')
                if col==0:ax.set_ylabel(row['room_id']+'\nD IoU '+f"{row['occupancy_iou']:.3f}"+'\nHidden interior cells: '+str(row['hidden_interior_cells']),fontsize=8)
        fig.suptitle('Worst occupancy cases with >=20 hidden interior cells: one per room, K=2')
        fig.text(.5,.015,'Source: AcousticRooms / Meta, CC BY 4.0. Teal: observation; amber: model; cream: truth. Scanner-aligned 0.20 m grid.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,.96));fig.savefig(root/f'negative_cases_{page//3+1}.png',dpi=130);plt.close(fig)
    atomic_json(root/'figure_provenance.json',{'summary_sha256':sha(root/'summary.json'),'plan_sha256':report['plan_sha256'],
                'script_sha256':sha(Path(__file__)),'selection_rule':'ascending D K2 occupancy IoU then sample ID; >=20 hidden interior cells; first six distinct room IDs',
                'sample_ids':[r['sample_id'] for r in chosen],'data_license':'AcousticRooms CC BY 4.0','git_assets':'figures remain local'})
    print('P11 figures rendered',root)

if __name__=='__main__':main()
