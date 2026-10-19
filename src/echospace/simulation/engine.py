"""Deterministic virtual capture and exact replay with private reference labels."""
from dataclasses import dataclass
from copy import deepcopy
import hashlib,json,platform
from importlib.metadata import version
import numpy as np
from shapely import contains_xy,points,distance
from . import ENGINE_VERSION
from .config import SimulationConfig
from .physics import Room
from .sensors import scan,audio,RATE,SAMPLES
from echospace.geometry.frames import GridFrame
from echospace.contract_v0 import validate_sample
from echospace.inference.samples import ServedSample

def runtime_versions():
    return {'python':platform.python_version(),**{name:version(name) for name in ('numpy','scipy','shapely')}}

OBSERVATION_KEYS=('observed_points_scene_m','observed_cells','observed_free','observed_wall',
                  'rir_waveforms','rir_positions_scene_m','rir_valid')

def digest(document,arrays=None):
    h=hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    for key,value in sorted((arrays or {}).items()):
        a=np.ascontiguousarray(value)
        h.update(f'{key}|{a.dtype}|{a.shape}|'.encode());h.update(a.tobytes())
    return h.hexdigest()

@dataclass
class Capture:
    sample: ServedSample
    metadata: dict
    beams: list
    reference: dict
    signals: dict
    def observation(self):
        return {key:self.sample.arrays[key].copy() for key in OBSERVATION_KEYS}
    def public(self):
        a=self.sample.arrays;used=min(self.metadata['requested_k'],int(a['rir_valid'].sum()))
        return {'format':'echospace.capture.v1',**self.metadata,
                'grid':self.sample.record['grid'],
                'observed_free':a['observed_free'].tolist(),'observed_wall':a['observed_wall'].tolist(),
                'observed':a['observed_cells'].tolist(),'beams':self.beams,
                'waveforms':a['rir_waveforms'][:used].astype(float).tolist(),
                'positions_scene_m':a['rir_positions_scene_m'][:used].astype(float).tolist(),
                'sample_rate_hz':RATE,'window_ms':80,
                'observed_cell_count':int(a['observed_cells'].sum()),
                'valid_pairs':int(a['rir_valid'].sum()),'k_used':used,
                'training_eligible':False}

def virtual_capture(config,anchor=None,anchor_heading=None,ordinal=0):
    if config.mode!='virtual':raise ValueError('virtual configuration required')
    room=Room(config.room)
    anchor=tuple(anchor or config.scanner.position_m)
    heading=config.scanner.heading_deg if anchor_heading is None else anchor_heading
    if not room.inside(anchor):raise ValueError('initial scanner anchor must be inside room')
    frame=GridFrame(anchor,float(np.deg2rad(heading)))
    corners=np.column_stack((room.vertices[:,0],np.full(len(room.vertices),anchor[1]),room.vertices[:,1]))
    cells=frame.scene_to_cell(corners)
    if (cells<0).any() or (cells>64).any():raise ValueError('room clips the 12.8 m scanner-anchored canvas; move the initial scanner or reduce the room')
    seed=np.random.SeedSequence([config.seed,ordinal]);rng_scan,rng_audio,rng_pose=[np.random.default_rng(s) for s in seed.spawn(3)]
    arrays,beams=scan(room,config.scanner,frame,rng_scan)
    if not arrays['observed_free'].any():raise ValueError('scan has no observed free cells; increase field of view/rays or reduce dropout')
    x,z=frame.cell_centres_xz();inside=contains_xy(room.polygon,x,z)
    boundary=inside&(distance(room.polygon.boundary,points(x,z))<=np.sqrt(2)*.1+1e-9)
    arrays.update(valid_cells=np.ones((64,64),np.uint8),target_occupancy=inside.astype(np.uint8),target_boundary=boundary.astype(np.uint8),
                  rir_waveforms=np.zeros((8,SAMPLES),np.float32),rir_positions_scene_m=np.zeros((8,2,3),np.float32),rir_valid=np.zeros(8,np.uint8))
    accepted=[];rejected=[];references=[];signals={}
    for index,pair in enumerate(config.pairs):
        if not room.inside(pair.source_m) or not room.inside(pair.receiver_m):raise ValueError(f'pair {index+1}: source/microphone must be inside the room')
        reported=np.asarray([pair.source_m,pair.receiver_m],float)
        def observed(positions):
            uv=np.floor(frame.scene_to_cell(positions)).astype(int)
            return bool(((uv>=0)&(uv<64)).all() and arrays['observed_free'][uv[:,1],uv[:,0]].all())
        if not observed(reported):
            rejected.append({'pair':index,'reason':'both source and microphone must occupy observed free cells'});continue
        if config.audio.pose_noise_m:
            for _ in range(32):
                candidate=reported+rng_pose.uniform(-config.audio.pose_noise_m,config.audio.pose_noise_m,(2,3))
                if ((candidate[:,1]>0)&(candidate[:,1]<config.room.height_m)).all() and observed(candidate):reported=candidate;break
        waveform,device,reference=audio(room,pair,config.audio,rng_audio)
        row=len(accepted);arrays['rir_waveforms'][row]=waveform
        arrays['rir_positions_scene_m'][row]=reported;arrays['rir_valid'][row]=1
        accepted.append(index);references.append(dict(reference,pair_index=index,true_positions_scene_m=[list(pair.source_m),list(pair.receiver_m)]))
        for name,value in device.items():signals[f'pair_{row}_{name}']=value
    identity={'engine':ENGINE_VERSION,'configuration':config.document(),'anchor':anchor,'anchor_heading_deg':heading,'ordinal':ordinal}
    capture_sha=digest(identity,arrays);sample_id='virtual_'+capture_sha[:20]
    # Coverage uses reference only for diagnostics; never enters model-facing keys.
    coverage=float((boundary&arrays['observed_wall'].astype(bool)).sum()/max(int(boundary.sum()),1))
    record={'schema_version':'0.1.0','sample_id':sample_id,'room_id':'virtual_'+digest(config.room.__dict__)[:12],
            'dataset':'EchoSpaceVirtual','geometry_id':digest(config.room.__dict__),'room_group':-1,
            'rir_ids':[f'{sample_id}/pair_{i}' for i in range(8)],'rir_sample_rate_hz':RATE,
            'sample_path':'observation.npz','split':'unassigned',
            'frame':{'description':'scene: X right, Y up, Z forward; virtual metric room','source_units_to_meters':1.,'source_to_scene':np.eye(4).tolist()},
            'grid':frame.manifest_grid(),'coverage':coverage,'coverage_bin':'synthetic_diagnostic','occlusion_type':'virtual_scanner','scan_seed':config.seed,
            'data_attribution':'EchoSpace procedurally generated virtual room; software-in-the-loop diagnostic, not AcousticRooms or physical hardware',
            'acquisition':{'engine_version':ENGINE_VERSION,'mode':'virtual','training_eligible':False,'ordinal':ordinal,'reference_is_private':True}}
    validate_sample(record,arrays)
    metadata={'capture_id':'capture_'+capture_sha[:20],'capture_sha256':capture_sha,'sample_id':sample_id,'mode':'virtual',
              'engine_version':ENGINE_VERSION,'runtime_versions':runtime_versions(),'requested_k':config.k,'ordinal':ordinal,'accepted_pair_indices':accepted,'rejected_pairs':rejected,
              'anchor_scene_m':list(anchor),'anchor_heading_deg':heading,'configuration':config.document(),
              'device_diagnostics':[{k:v for k,v in r.items() if k not in ('paths','true_positions_scene_m')} for r in references],
              'warning':'Synthetic diagnostic; simple first-order acoustics; not a held-out benchmark or validated hardware model'}
    reference={'format':'echospace.simulation_reference.v1','room_vertices_xz':room.vertices.tolist(),'room_height_m':config.room.height_m,
               'target_occupancy':arrays['target_occupancy'].tolist(),'target_boundary':arrays['target_boundary'].tolist(),
               'acoustic_paths':references,'privileged':True,'not_model_input':True}
    return Capture(ServedSample(sample_id,capture_sha,record,arrays),metadata,beams,reference,signals)

def replay_capture(config,store,ordinal=0):
    if config.mode!='replay':raise ValueError('replay configuration required')
    original=store.load(config.sample_id)
    # Deep copies prevent session state from mutating the verified source store.
    sample=ServedSample(original.sample_id,original.sha256,deepcopy(original.record),{k:v.copy() for k,v in original.arrays.items()})
    validate_sample(sample.record,sample.arrays)
    key=digest({'engine':ENGINE_VERSION,'mode':'replay','sample_sha256':sample.sha256,'k':config.k})
    metadata={'capture_id':'capture_'+key[:20],'capture_sha256':key,'source_sample_sha256':sample.sha256,
              'sample_id':sample.sample_id,'mode':'replay','engine_version':ENGINE_VERSION,'runtime_versions':runtime_versions(),'requested_k':config.k,
              'ordinal':ordinal,'configuration':config.document(),
              'warning':'Exact audited dataset replay; source RIRs are simulated AcousticRooms responses, not new hardware recordings'}
    return Capture(sample,metadata,[],{'format':'echospace.simulation_reference.v1','privileged':True,'not_model_input':True,
                                    'target_occupancy':sample.arrays['target_occupancy'].tolist(),'target_boundary':sample.arrays['target_boundary'].tolist()}, {})
