"""Verify software sensor capture, exact real replay, registered A/P/D and exports."""
import io,json,zipfile,urllib.request,argparse,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
from jsonschema import Draft202012Validator
from echospace.inference.samples import SampleStore,model_inputs
from echospace.simulation.config import SimulationConfig
from echospace.simulation.engine import replay_capture
parser=argparse.ArgumentParser(description='Local real-checkpoint simulation/replay integration gate; requires API and verified eval samples.')
parser.add_argument('--api-url',default='http://127.0.0.1:8000')
parser.add_argument('--source-root',type=Path,default=REPO)
parser.add_argument('--output',type=Path,default=REPO/'artifacts/simulation_gate')
args=parser.parse_args();base=args.api_url.rstrip('/');out=args.output;out.mkdir(exist_ok=True,parents=True)
def req(path,data=None):
    request=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=120) as r:return r.read()
def get(path):return json.loads(req(path))
def post(path,data):return json.loads(req(path,data))
models=get('/v1/models')['models'];models=[m for m in models if m['available'] and m['variant'] in ['A','P','D']]
assert {m['variant'] for m in models}=={'A','P','D'},'register available A/P/D checkpoints before the integration gate'
samples=get('/v1/samples?split=test&fold=0&limit=1')['samples'];sample_id=samples[0]['sample_id']
store=SampleStore(args.source_root/'data/processed/eval_samples',args.source_root/'data/splits/eval_masks.json',args.source_root/'data/splits/folds.json')
original=store.load(sample_id);replay=replay_capture(SimulationConfig(mode='replay',sample_id=sample_id,k=2),store)
assert original.sha256==replay.sample.sha256
for k,v in model_inputs(original,2).items():np.testing.assert_array_equal(v,model_inputs(replay.sample,2)[k])
report={'exact_replay_sample':sample_id,'exact_replay_sha256':original.sha256,'exact_replay_model_tensors':True,'cases':[]}
configs=[('rectangle',{}),('l_room',{'room':{'shape':'l_room'}}),('sweep_faults',{'audio':{'excitation':'sweep','noise_snr_db':25,'offset_ms':.2,'clock_drift_ppm':100,'lowpass_hz':6000}}),('replay',{'mode':'replay','sample_id':sample_id,'k':2})]
for name,config in configs:
    session=post('/v1/simulation/sessions',{'schema_version':'echospace.simulation_config.v1','config':config});sid=session['session_id']
    cap=post(f'/v1/simulation/sessions/{sid}/capture',{});cid=cap['capture_id'];case={'name':name,'capture_id':cid,'valid_pairs':cap['valid_pairs'],'observed_cells':cap['observed_cell_count'],'models':[]}
    obs=req(f'/v1/simulation/captures/{cid}/observation.npz')
    with np.load(io.BytesIO(obs),allow_pickle=False) as a:assert not any(k.startswith('target_') or k=='valid_cells' for k in a.files)
    bundle=req(f'/v1/simulation/sessions/{sid}/bundle.zip');(out/f'{name}.zip').write_bytes(bundle)
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:assert 'REFERENCE_ONLY.json' not in z.namelist()
    for model in models:
        r=post(f'/v1/simulation/sessions/{sid}/infer',{'model_id':model['model_id']})
        glb=req(r['links']['glb']);assert glb[:4]==b'glTF';scene=get(r['links']['room_scene'])
        Draft202012Validator(json.loads((REPO/'contracts/v0.1.0/room-scene.schema.json').read_text())).validate(scene)
        assert req(scene['observations'][0]['payload_ref'])[:2]==b'PK'
        if name=='replay':
            direct=post('/v1/reconstructions',{'schema_version':'echospace.reconstruction_request.v1','model_id':model['model_id'],'sample_id':sample_id,'k':2})
            assert r['reconstruction_id']==direct['reconstruction_id'];assert r['metrics']==direct['metrics']
        case['models'].append({'model_id':model['model_id'],'reconstruction_id':r['reconstruction_id'],'split':r['split_for_model_fold'],'metrics':r['metrics'],'glb_bytes':len(glb),'room_scene_schema':'PASS'})
    report['cases'].append(case)
report['status']='PASS';(out/'real_gate.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','cases':len(report['cases']),'model_fits_used_for_inference':len(models),'inferences':sum(len(x['models']) for x in report['cases']),'exact_replay':sample_id}))
