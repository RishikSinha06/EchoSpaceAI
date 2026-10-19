"""Analytical software sensor tests and target-isolated inference integration."""
from dataclasses import replace
import copy,io,json,zipfile
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('torch');pytest.importorskip('fastapi');pytest.importorskip('shapely');pytest.importorskip('scipy')
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from echospace.simulation.config import SimulationConfig,RoomConfig,ScannerConfig,AudioConfig,PairConfig,from_dict,SCHEMA
from echospace.simulation.physics import Room,fractional_impulses
from echospace.simulation.engine import virtual_capture,replay_capture
from echospace.simulation.sessions import Sessions,Session
from echospace.inference.samples import model_inputs
from echospace.inference.api import create_app,Settings
from test_inference import repo,_store

BASE=Path(__file__).resolve().parents[1]

def client(repo):
    return TestClient(create_app(Settings(repo_root=repo,models='models.json',eval_dir='eval',eval_spec='spec.json',folds='folds.json')))

def create(c,config=None):
    response=c.post('/v1/simulation/sessions',json={'schema_version':SCHEMA,'config':config or {}})
    assert response.status_code==201,response.text
    return response.json()['session_id']

@pytest.mark.parametrize('bad',[{'seed':True},{'seed':-1},{'k':9},{'unknown':1},{'scanner':{'rays':9000}},
                              {'scanner':{'position_m':[1,2]}},{'room':{'width_m':float('nan')}},
                              {'audio':{'lowpass_hz':8000}},{'pairs':[]},{'mode':'replay'}])
def test_configuration_rejects_invalid_fields(bad):
    with pytest.raises(ValueError):from_dict(bad)

def test_json_schema_and_config_roundtrip():
    config=SimulationConfig();document={'schema_version':SCHEMA,'config':config.document()}
    Draft202012Validator(json.loads((BASE/'contracts/simulation-config.v1.schema.json').read_text())).validate(document)
    assert from_dict(config.document())==config
    with pytest.raises(ValueError,match='duplicate'):from_dict({'pairs':[config.document()['pairs'][0]]*2})
    with pytest.raises(ValueError,match='exact'):from_dict({'mode':'replay','sample_id':'sample','audio':{'gain_db':2}})

def test_analytic_direct_and_finite_reflections():
    room=Room(RoomConfig());paths=room.path((1.2,1.2,1.4),(2.,1.2,1.4));by={p['kind']:p for p in paths}
    assert by['direct']['length_m']==pytest.approx(.8)
    assert by['wall_3']['length_m']==pytest.approx(3.2)
    assert by['floor']['length_m']==pytest.approx(np.hypot(.8,2.4))
    assert by['ceiling']['length_m']==pytest.approx(np.hypot(.8,3.2))
    signal=fractional_impulses([by['wall_3']])
    index=3.2/343*16000
    assert abs(int(np.argmax(signal))-index)<=.5
    assert np.sum(signal)==pytest.approx(by['wall_3']['amplitude'])
    vertical=room.path((2,1.,2),(2,1.5,2));assert any(p['kind']=='direct' for p in vertical)
    with pytest.raises(ValueError,match='inside'):room.path((-1,1,2),(2,1,2))

def test_nonconvex_paths_require_visibility():
    room=Room(RoomConfig(shape='l_room'));s=np.array([5.8,1.2,2.5]);m=np.array([3.,1.2,4.8])
    assert not room.visible(s,m)
    paths=room.path(s,m)
    assert not any(p['kind']=='direct' for p in paths)
    for path in paths:
        pts=[np.array(p) for p in path['points_scene_m']]
        assert all(room.visible(a,b) for a,b in zip(pts,pts[1:]))

def test_capture_is_deterministic_and_labels_never_reach_model_inputs():
    config=SimulationConfig(audio=AudioConfig(noise_snr_db=20,offset_ms=.2,pose_noise_m=.02),scanner=ScannerConfig(dropout=.1))
    a=virtual_capture(config);b=virtual_capture(config)
    assert a.metadata['capture_sha256']==b.metadata['capture_sha256']
    assert int(a.sample.arrays['rir_valid'].sum())==1
    for key in a.sample.arrays:np.testing.assert_array_equal(a.sample.arrays[key],b.sample.arrays[key])
    before=model_inputs(a.sample,1)
    assert not {'target_occupancy','target_boundary','valid_cells','room_id'}&before.keys()
    a.sample.arrays['target_occupancy'][:]=1-a.sample.arrays['target_occupancy']
    a.sample.arrays['target_boundary'][:]=1-a.sample.arrays['target_boundary']
    a.sample.arrays['valid_cells'][:]=0
    after=model_inputs(a.sample,1)
    for key in before:np.testing.assert_array_equal(before[key],after[key])
    public=b.public()
    assert not {'target_occupancy','target_boundary','acoustic_paths','true_positions_scene_m'}&public.keys()
    assert b.observation().keys().isdisjoint({'target_occupancy','target_boundary','valid_cells'})
    c=virtual_capture(config,ordinal=1)
    assert c.metadata['capture_sha256']!=b.metadata['capture_sha256']

def test_scan_range_and_ineligible_pairs_remain_unknown():
    capture=virtual_capture(SimulationConfig(scanner=ScannerConfig(range_m=.5)))
    assert capture.sample.arrays['observed_wall'].sum()==0
    assert capture.sample.arrays['observed_free'].sum()<64
    assert capture.public()['valid_pairs']==0
    assert capture.metadata['rejected_pairs']
    with pytest.raises(ValueError,match='clips'):virtual_capture(SimulationConfig(room=RoomConfig(width_m=10)))

def test_sweep_device_chain_and_faults():
    cfg=SimulationConfig(audio=AudioConfig(excitation='sweep',gain_db=12,clip_level=.05,noise_snr_db=25,lowpass_hz=4000,offset_ms=.2,clock_drift_ppm=100,adc_bits=12))
    capture=virtual_capture(cfg)
    assert len(capture.signals['pair_0_excitation'])==2048
    assert len(capture.signals['pair_0_recording'])==3327
    assert capture.metadata['device_diagnostics'][0]['clipped_samples']>0
    assert np.isfinite(capture.sample.arrays['rir_waveforms']).all()
    assert np.max(np.abs(capture.sample.arrays['rir_waveforms']))==pytest.approx(1)
    assert capture.metadata['device_diagnostics'][0]['clock_faults_compensated'] is False

def test_replay_tensors_and_hashes_are_exact(repo):
    store=_store(repo);config=SimulationConfig(mode='replay',sample_id='RoomTest_idx_0__viewpoint__mid__1',k=2)
    original=store.load(config.sample_id);capture=replay_capture(config,store)
    assert capture.sample.sha256==original.sha256
    for key,value in model_inputs(original,2).items():np.testing.assert_array_equal(value,model_inputs(capture.sample,2)[key])
    capture.sample.arrays['rir_waveforms'][:]=0
    assert store.load(config.sample_id).arrays['rir_waveforms'].any()

def test_session_anchor_clock_bounds_and_observation_bundle():
    sessions=Sessions(None,max_sessions=2,max_captures=2);s=sessions.create({});a=sessions.capture(s.identifier)
    old_grid=a.sample.record['grid'];config=s.config.document();config['scanner']['position_m'][0]=2.2
    sessions.update(s.identifier,config);assert s.last_capture is None
    b=sessions.capture(s.identifier);assert b.sample.record['grid']==old_grid
    times=[e['time_s'] for e in s.events];assert times==sorted(times)
    zipped=sessions.bundle(s.identifier)
    with zipfile.ZipFile(io.BytesIO(zipped)) as z:
        assert 'REFERENCE_ONLY.json' not in z.namelist()
        with np.load(io.BytesIO(z.read('observation.npz')),allow_pickle=False) as arrays:assert 'target_occupancy' not in arrays
        with np.load(io.BytesIO(z.read('device_signals.npz')),allow_pickle=False) as arrays:assert not any(k.endswith('clean_room_rir') for k in arrays.files)
        assert json.loads(z.read('session.json'))['training_eligible'] is False
    with zipfile.ZipFile(io.BytesIO(sessions.bundle(s.identifier,True))) as z:assert 'REFERENCE_ONLY.json' in z.namelist()
    sessions.create({});sessions.create({})
    with pytest.raises(LookupError):sessions.get(s.identifier)
    for i in range(300):s.event('test')
    sequence=[e['sequence'] for e in s.events];assert len(sequence)==256 and len(set(sequence))==256

def test_api_virtual_capture_inference_exports_and_stale_state(repo):
    c=client(repo);sid=create(c)
    assert c.post(f'/v1/simulation/sessions/{sid}/infer',json={'model_id':'tiny-d'}).status_code==409
    cap=c.post(f'/v1/simulation/sessions/{sid}/capture').json();cid=cap['capture_id']
    assert cap['mode']=='virtual' and cap['valid_pairs']==1
    assert 'target_occupancy' not in cap
    result=c.post(f'/v1/simulation/sessions/{sid}/infer',json={'model_id':'tiny-d'})
    assert result.status_code==200,result.text
    result=result.json();assert result['split_for_model_fold']=='synthetic_diagnostic'
    document=c.get(result['links']['room_scene']).json()
    Draft202012Validator(json.loads((BASE/'contracts/v0.1.0/room-scene.schema.json').read_text())).validate(document)
    observation=c.get(document['observations'][0]['payload_ref']);assert observation.status_code==200
    with np.load(io.BytesIO(observation.content),allow_pickle=False) as data:assert 'target_occupancy' not in data
    provenance=c.get(result['links']['provenance']).json()
    assert 'procedurally generated' in provenance['data_attribution']
    assert provenance['simulation']['physical_hardware'] is False
    assert c.get(f'/v1/samples/{cap["sample_id"]}/observation.npz').status_code==404
    assert c.get(f'/v1/simulation/captures/{cid}/audio.wav').content[:4]==b'RIFF'
    signals=c.get(f'/v1/simulation/captures/{cid}/signals').json()
    assert signals['clean_reference_included'] is False
    assert c.get(f'/v1/simulation/captures/{cid}/reference.json').json()['privileged'] is True
    config=SimulationConfig().document();config['scanner']['heading_deg']=15
    assert c.patch(f'/v1/simulation/sessions/{sid}',json={'schema_version':SCHEMA,'config':config}).status_code==200
    assert c.post(f'/v1/simulation/sessions/{sid}/infer',json={'model_id':'tiny-d'}).status_code==409
    assert c.get(document['observations'][0]['payload_ref']).status_code==200 # old capture reference never silently changes
    assert c.get(f'/v1/simulation/sessions/{sid}/bundle.zip').status_code==422

def test_api_replay_seen_rooms_invalid_configuration_and_zero_pairs(repo):
    c=client(repo)
    assert c.post('/v1/simulation/sessions',json={'schema_version':'wrong','config':{}}).status_code==422
    assert c.post('/v1/simulation/sessions',json={'schema_version':SCHEMA,'config':{'surprise':True}}).status_code==422
    assert c.post('/v1/simulation/sessions',json={'schema_version':SCHEMA,'config':{'mode':'replay','sample_id':'unknown'}}).status_code==404
    sid=create(c,{'mode':'replay','sample_id':'RoomTrain_idx_0__viewpoint__mid__2'})
    response=c.post(f'/v1/simulation/sessions/{sid}/capture');assert response.status_code==200,response.text
    assert c.post(f'/v1/simulation/sessions/{sid}/infer',json={'model_id':'tiny-d'}).status_code==409
    sid=create(c,{'scanner':{'range_m':.5}})
    assert c.post(f'/v1/simulation/sessions/{sid}/capture').json()['valid_pairs']==0
    assert c.post(f'/v1/simulation/sessions/{sid}/infer',json={'model_id':'tiny-d'}).status_code==409
