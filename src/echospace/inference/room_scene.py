"""Metric RoomScene v1 with observation-only measured segments and model layers.

Measured segments are free/hit interfaces in the scan raster, independent of
the model prediction. Their floor projection is quantized at the cell size;
they are not an exact mesh fit or recovered three-dimensional walls.
"""
from urllib.parse import quote

import numpy as np

from .export import _EDGES, wall_edges


def room_scene_document(result):
    arrays, record = result.sample.arrays, result.sample.record
    grid = record['grid']
    origin = np.asarray(grid['origin_scene_m'],float)
    u, v = np.asarray(grid['axis_u_scene'],float), np.asarray(grid['axis_v_scene'],float)
    size = float(grid['cell_size_m'])
    observation_id = result.sample.sample_id + ':partial-scan'
    source = '/v1/samples/' + quote(result.sample.sample_id,safe='') + '/observation.npz'

    def segment(row,col,index):
        _, _, a, b = _EDGES[index]
        def point(c,r):
            p = origin + size*(c*u+r*v)
            p[1] = 0
            return [round(float(x),8) for x in p]
        return point(col+a[0],row+a[1]), point(col+b[0],row+b[1])

    def key(a,b):
        return tuple(sorted((tuple(a),tuple(b))))

    measured, inferred, measured_keys = [], [], set()
    for row,col,index,hit in wall_edges(arrays['observed_free'],arrays['observed_wall']):
        if not hit:
            continue
        a,b = segment(row,col,index)
        measured.append({'kind':'wall_segment','start_scene_m':a,'end_scene_m':b,
                         'provenance':'measured','observation_id':observation_id})
        measured_keys.add(key(a,b))
    for row,col,index,_ in wall_edges(result.interior,arrays['observed_wall']):
        a,b = segment(row,col,index)
        if key(a,b) not in measured_keys:
            inferred.append({'kind':'wall_segment','start_scene_m':a,'end_scene_m':b,'provenance':'model',
                             'model_version':result.model.model_id+'@'+result.model.checkpoint_sha256})
    positions = arrays['rir_positions_scene_m']
    acoustic = []
    for i in range(result.k_used):
        acoustic.append({'rir_id':record['rir_ids'][i], 'rir_ref':source+f'#rir_waveforms[{i}]',
                         'sample_rate_hz':record['rir_sample_rate_hz'],
                         'source_scene_m':positions[i,0].astype(float).tolist(),
                         'receiver_scene_m':positions[i,1].astype(float).tolist(),
                         'timing_metadata':{'alignment':'emission','window_ms':80,'normalization':'peak',
                                            'processing':'verified P5/P6 fixed sample'}})
    return {'format':'echospace.room-scene.v1','scene_id':result.reconstruction_id,'scene_version':1,
            'units':'m','coordinate_system':'scene: X right, Y up, Z forward','world_origin_m':[0.,0.,0.],
            'source_sample_id':result.sample.sample_id,'observed_geometry':measured,'inferred_geometry':inferred,
            'observations':[{'observation_id':observation_id,'sensor_type':'simulated_partial_scan',
                             'payload_ref':source}], 'acoustic_measurements':acoustic,
            'grid_frame':grid,'geometry_interpretation':'2D raster interfaces projected to Y=0; measured layer uses only scan free/hit masks',
            'quantization_m':size,'wall_height_recovered':False,
            'provenance_ref':f'/v1/reconstructions/{result.reconstruction_id}/provenance.json',
            'prediction_ref':f'/v1/reconstructions/{result.reconstruction_id}/prediction.npz',
            'data_attribution':result.provenance['data_attribution']}
