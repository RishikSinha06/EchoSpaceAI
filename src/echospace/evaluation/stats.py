"""Equal-room summaries and paired physical-group resampling."""
from collections import defaultdict
import numpy as np
from echospace.training.demo import paired_group_delta

METRICS = ('occupancy_iou','boundary_iou','boundary_f1_20cm','boundary_chamfer_m',
           'occupancy_brier','boundary_brier','occupancy_calibrated_brier','boundary_calibrated_brier')

def room_summary(rows):
    if not rows: return {'rooms':0,'samples':0,'metrics':{}}
    seen = set(); rooms = defaultdict(list)
    for row in rows:
        if row['sample_id'] in seen: raise ValueError('duplicate sample ID')
        seen.add(row['sample_id']); rooms[row['room_id']].append(row)
    values = {room:{key:float(np.mean([r[key] for r in rs])) for key in METRICS}
              for room,rs in sorted(rooms.items())}
    return {'rooms':len(rooms),'samples':len(rows),
            'metrics':{key:float(np.mean([v[key] for v in values.values()])) for key in METRICS},
            'per_room':values}

def paired_metrics(left,right,repeats=2000):
    result = {}
    for metric in ('occupancy_iou','boundary_f1_20cm','boundary_chamfer_m'):
        a = [dict(r,occupancy_iou=r[metric]) for r in left]
        b = [dict(r,occupancy_iou=r[metric]) for r in right]
        value = paired_group_delta(a,b,repeats=repeats)
        value['scope'] = 'exploratory, equal-room effect; physical-room-group percentile bootstrap; not independent external validation'
        value['positive_meaning'] = 'worse' if metric.endswith('_m') else 'better'
        result[metric] = value
    return result
