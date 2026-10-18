"""CPU missing-region scoring and validation calibration for frozen datasets."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from echospace.training.quality import boundary_quality
from .protocol import choose_temperature, probability

def batches(model,ds,batch_size):
    offset=0
    with torch.inference_mode():
        for batch in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
            output=model(batch)
            for i in range(len(batch['k'])):
                mask=((batch['valid_cells'][i]==1)&(batch['observed_cells'][i]==0)).numpy()
                if not mask.any(): raise ValueError('empty missing-region supervision')
                yield ds.entries[offset+i]['metadata'],batch,i,mask,{h:output[h+'_logits'][i].float().numpy() for h in ('occupancy','boundary')}
            offset+=len(batch['k'])

def calibrate(model,ds,batch_size):
    records={h:[] for h in ('occupancy','boundary')}
    for meta,batch,i,mask,logits in batches(model,ds,batch_size):
        for h in records:
            records[h].append({'room_id':meta['room_id'],'logits':logits[h][mask],
                               'target':batch['target_'+h][i].numpy()[mask]})
    return {h:choose_temperature(rs,ds.split) for h,rs in records.items()}

def score(model,ds,batch_size,calibration):
    rows=[]
    for meta,batch,i,mask,logits in batches(model,ds,batch_size):
        row={k:meta[k] for k in ('sample_id','room_id','room_group','occlusion_type','coverage_bin')}
        row['k']=int(batch['k'][i]);row['coverage']=meta['coverage']
        for h in logits:
            target=batch['target_'+h][i].numpy().astype(bool)
            p=probability(logits[h]);cal=probability(logits[h],calibration[h]['temperature'])
            pred=(p>=.5)&mask;truth=target&mask;union=(pred|truth).sum()
            row[h+'_iou']=float((pred&truth).sum()/union) if union else 1.
            row[h+'_brier']=float(((p[mask]-target[mask])**2).mean())
            row[h+'_calibrated_brier']=float(((cal[mask]-target[mask])**2).mean())
            if h=='boundary': row.update(boundary_quality(pred,truth,meta['grid']['cell_size_m']))
        rows.append(row)
    return rows

