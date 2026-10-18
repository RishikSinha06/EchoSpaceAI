import json
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from echospace.evaluation.controls import ControlledDataset, CONDITIONS
from echospace.evaluation.protocol import choose_temperature, verify_lock, sha
from echospace.evaluation.stats import room_summary, paired_metrics, METRICS

class Tiny:
    split='test'
    def __init__(self):
        self.entries=[{'metadata':{'sample_id':f's{i}','room_id':f'r{i}','room_group':i}} for i in range(3)]
    def __len__(self): return len(self.entries)
    def __getitem__(self,i):
        wave=torch.zeros(8,1280);wave[:2,20]=i+1
        valid=torch.tensor([1,1,0,0,0,0,0,0],dtype=torch.uint8)
        poses=torch.zeros(8,3);poses[:2]=torch.tensor([1.1,1.1,1.])
        return {'rir':wave,'rir_valid':valid,'k':torch.tensor(2),'src_pos':poses.clone(),'mic_pos':poses.clone(),
                'observed_free':torch.ones(64,64,dtype=torch.uint8),'target_occupancy':torch.zeros(64,64)}

@pytest.mark.parametrize('condition',CONDITIONS)
def test_controls_preserve_identity_masks_and_targets(condition):
    base=Tiny();ds=ControlledDataset(base,condition)
    a,b=ds[0],ds[0]
    for key in a: assert torch.equal(a[key],b[key])
    for key in ('rir_valid','k','observed_free','target_occupancy'):assert torch.equal(a[key],base[0][key])
    if condition!='pose5cm':
        assert torch.equal(a['src_pos'],base[0]['src_pos'])
        assert torch.equal(a['mic_pos'],base[0]['mic_pos'])
    else:
        assert (a['src_pos']-base[0]['src_pos']).abs().max()<=.05+1e-7
    assert not a['rir'][2:].any()
    assert base[0]['rir'][0,20]==1

def test_control_semantics_and_donor_rejection():
    base=Tiny()
    assert not ControlledDataset(base,'zero')[0]['rir'].any()
    delayed=ControlledDataset(base,'delay_plus3')[0]['rir']
    assert delayed[0,23]==1 and not delayed[0,:23].any()
    early=ControlledDataset(base,'delay_minus3')[0]['rir'];assert early[0,17]==1
    noisy=ControlledDataset(base,'noise20')[0]['rir'];original=base[0]['rir']
    ratio=original[0].square().mean()/(noisy[0]-original[0]).square().mean()
    assert float(10*torch.log10(ratio))==pytest.approx(20,abs=1e-4)
    swapped=ControlledDataset(base,'swapped');assert all(i!=j for i,j in swapped.donors.items())
    assert swapped.donor_manifest()[0]['donor_room_group']!=0
    base.entries=base.entries[:1]
    with pytest.raises(ValueError,match='donor'):ControlledDataset(base,'swapped')

def test_temperature_uses_validation_only():
    records=[{'room_id':'a','logits':np.array([8.,-8.]),'target':np.array([0.,1.])}]
    with pytest.raises(ValueError,match='validation'):choose_temperature(records,'test')
    result=choose_temperature(records,'val')
    assert result['temperature']==3 and result['fit_split']=='val'

def rows():
    return [dict(sample_id=f's{i}',room_id=room,room_group=group,k=2,
                 **{m:score for m in METRICS}) for i,(room,group,score) in enumerate([('a',0,0),('a',0,0),('b',1,1)])]

def test_room_weighting_and_paired_identity():
    a=rows();assert room_summary(a)['metrics']['occupancy_iou']==.5
    b=[dict(r,occupancy_iou=r['occupancy_iou']+.1) for r in a]
    effect=paired_metrics(a,b,repeats=50)['occupancy_iou']
    assert effect['mean_room_delta']==pytest.approx(.1)
    b[0]['k']=4
    with pytest.raises(ValueError,match='identities'):paired_metrics(a,b,repeats=20)
    with pytest.raises(ValueError,match='duplicate'):room_summary(a+a)

def test_selection_lock_checks_bytes_and_frozen_identity(tmp_path):
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'checkpoint')
    lock=tmp_path/'selection.json'
    record={'status':'locked','frozen_checksum':'frozen','models':[{'checkpoint_sha256':sha(checkpoint)}]}
    lock.write_text(json.dumps(record))
    assert verify_lock(lock,[checkpoint],'frozen')['selection_sha256']==sha(lock)
    with pytest.raises(ValueError,match='frozen'):verify_lock(lock,[checkpoint],'other')
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint'):verify_lock(lock,[checkpoint],'frozen')


def test_parent_study_gate_refuses_missing_or_incompatible_results(tmp_path):
    from echospace.evaluation.protocol import require_finished_study
    with pytest.raises(ValueError,match='missing'):require_finished_study(tmp_path)
    checkpoint=tmp_path/'full/fold0/echo_D/seed0/best.pt';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'locked')
    selection=tmp_path/'selection.json';selection.write_text(json.dumps({'status':'locked','models':[{'fold':0,'name':'echo_D','seed':0,'checkpoint_sha256':sha(checkpoint)}]}))
    result=tmp_path/'results.json';result.write_text(json.dumps({'status':'internal_study_complete','selection_sha256':'wrong'}))
    with pytest.raises(ValueError,match='identity'):require_finished_study(tmp_path)
    result.write_text(json.dumps({'status':'internal_study_complete','selection_sha256':sha(selection)}))
    assert require_finished_study(tmp_path)['status']=='locked'


def test_full_study_coverage_gate():
    from echospace.evaluation.protocol import verify_study_coverage
    protocol={'folds':[0,1],'seeds':[0],'models':[{'name':'echo_D'}],'ablations':[]}
    with pytest.raises(ValueError,match='coverage'):verify_study_coverage({'models':[{'fold':0,'name':'echo_D','seed':0}]},protocol)
    verify_study_coverage({'models':[{'fold':i,'name':'echo_D','seed':0} for i in (0,1)]},protocol)

def test_engine_scores_only_valid_missing_cells_and_fits_val_only():
    from echospace.evaluation.engine import calibrate,score
    class Data(Tiny):
        split='val'
        def __init__(self):
            super().__init__()
            for e in self.entries:e['metadata'].update(occlusion_type='viewpoint',coverage_bin='mid',coverage=.5,grid={'cell_size_m':.2})
        def __getitem__(self,i):
            item=super().__getitem__(i)
            item['observed_cells']=torch.ones(64,64,dtype=torch.uint8);item['observed_cells'][0,0]=0
            item['valid_cells']=torch.ones(64,64,dtype=torch.uint8)
            item['target_occupancy']=torch.ones(64,64,dtype=torch.uint8)
            item['target_boundary']=torch.ones(64,64,dtype=torch.uint8)
            return item
    class Net(torch.nn.Module):
        def forward(self,batch):
            logits=torch.full((len(batch['k']),64,64),-8.);logits[:,0,0]=8.
            return {'occupancy_logits':logits,'boundary_logits':logits.clone()}
    ds=Data();model=Net();cal=calibrate(model,ds,2);rows=score(model,ControlledDataset(ds),2,cal)
    assert len(rows)==3
    assert all(r['occupancy_iou']==1 and r['boundary_f1_20cm']==1 and r['boundary_chamfer_m']==0 for r in rows)
    ds.split='test'
    with pytest.raises(ValueError,match='validation'):calibrate(model,ds,2)


def test_cs_driver_locks_every_fit_before_any_test_scoring(tmp_path,monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    script=Path(__file__).resolve().parents[1]/'scripts/train_stage2_controls.py'
    spec=importlib.util.spec_from_file_location('p11_cs_test',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source';(source/'configs').mkdir(parents=True)
    parent={'folds':[0],'seeds':[0,1],'models':[{'name':'echo_D'}],'ablations':[],
            'training':{'epochs':1,'device':'cpu'},'backbone':{}}
    parent_path=source/'configs/parent.json';parent_path.write_text(json.dumps(parent))
    stage=source/'study';stage.mkdir();(stage/'selection.json').write_text('{}');(stage/'results.json').write_text('{}')
    locked={'config_sha256':sha(parent_path),'models':[{'fold':0,'name':'echo_D','seed':s,'frozen_checksum':'f'} for s in (0,1)]}
    monkeypatch.setattr(module,'require_finished_study',lambda root:locked)
    config=tmp_path/'config.json';config.write_text(json.dumps({'source_study':'study','source_config':'configs/parent.json','folds':[0],'seeds':[0,1],'variants':['C','S'],'evaluation_k':[2]}))
    out=tmp_path/'out';events=[]
    monkeypatch.setattr(module,'FrozenAudioDataset',lambda *a,**kw:SimpleNamespace(checksum='f'))
    def train(variant,frozen,destination,*args,**kwargs):
        destination.mkdir(parents=True);path=destination/'best.pt';path.write_bytes(f'{variant}{len(events)}'.encode());events.append('train')
        return {'checkpoint_sha256':sha(path),'frozen_checksum':'f'}
    monkeypatch.setattr(module,'run_fusion',train)
    class Model:
        def cuda(self):return self
        def eval(self):return self
    monkeypatch.setattr(module,'load_fusion_checkpoint',lambda *a:(Model(),{}))
    def evaluate(*a,**kw):
        assert events.count('train')==4
        assert json.loads((out/'selection.json').read_text())['status']=='locked'
        events.append('test')
        return {},[{'sample_id':'s','room_id':'r','loss':0.,'occupancy_iou':1.,'boundary_iou':1.}]
    monkeypatch.setattr(module,'evaluate',evaluate)
    monkeypatch.setattr('sys.argv',['train_stage2_controls.py','--source-root',str(source),'--config',str(config),'--output',str(out),'--mode','train'])
    module.main()
    assert events==['train']*4+['test']*4
    assert json.loads((out/'results.json').read_text())['status']=='internal_controls_complete'
