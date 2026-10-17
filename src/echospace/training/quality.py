"""Missing-region shape and calibration diagnostics at fixed metric tolerances."""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader


def boundary_quality(predicted, target, cell_m=.2):
    predicted, target = predicted.astype(bool), target.astype(bool)
    if not predicted.any() and not target.any():
        return {'boundary_f1_20cm':1., 'boundary_f1_40cm':1., 'boundary_chamfer_m':0.}
    if not predicted.any() or not target.any():
        # Explicit finite penalty, the grid diagonal, for a missing entire set.
        return {'boundary_f1_20cm':0., 'boundary_f1_40cm':0.,
                'boundary_chamfer_m':float(np.hypot(*predicted.shape)*cell_m)}
    to_target = distance_transform_edt(~target)*cell_m
    to_prediction = distance_transform_edt(~predicted)*cell_m
    result = {'boundary_chamfer_m':float((to_target[predicted].mean()+to_prediction[target].mean())/2)}
    for tolerance, name in ((.2,'boundary_f1_20cm'),(.4,'boundary_f1_40cm')):
        precision = (to_target[predicted]<=tolerance+1e-8).mean()
        recall = (to_prediction[target]<=tolerance+1e-8).mean()
        result[name] = float(2*precision*recall/max(precision+recall,1e-12))
    return result


def evaluate_quality(model, dataset, batch_size, device):
    """No calibration fitted here; raw probabilities scored only on valid M=0."""
    model.eval()
    rows, offset = [], 0
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0):
            output = model({k:v.to(device) for k,v in batch.items()})
            probability = {h:output[h+'_logits'].float().sigmoid().cpu().numpy() for h in ('occupancy','boundary')}
            for i in range(len(batch['k'])):
                meta = dataset.entries[offset+i]['metadata']
                mask = ((batch['observed_cells'][i]==0)&(batch['valid_cells'][i]==1)).numpy()
                if not mask.any():
                    raise ValueError('quality sample has no missing-region supervision')
                row = {'sample_id':meta['sample_id'],'room_id':meta['room_id'],'room_group':meta['room_group']}
                for head in probability:
                    target = batch['target_'+head][i].numpy()
                    row[head+'_brier'] = float(((probability[head][i][mask]-target[mask])**2).mean())
                row.update(boundary_quality((probability['boundary'][i]>=.5)&mask,
                                            batch['target_boundary'][i].numpy().astype(bool)&mask,
                                            meta['grid']['cell_size_m']))
                rows.append(row)
            offset += len(batch['k'])
    return rows
