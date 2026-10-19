"""Capture a deterministic virtual room or an exact audited replay into a local bundle."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from echospace.simulation.config import SCHEMA,SimulationConfig
from echospace.simulation.sessions import Sessions
from echospace.inference.samples import SampleStore

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path)
    ap.add_argument('--sample-id',help='exact replay of this audited sample; no fault overrides')
    ap.add_argument('--source-root',type=Path,default=REPO)
    ap.add_argument('--output',type=Path,default=REPO/'artifacts/simulation/demo')
    ap.add_argument('--include-reference',action='store_true')
    args=ap.parse_args()
    envelope=json.loads(args.config.read_text()) if args.config else {'schema_version':SCHEMA,'config':SimulationConfig().document()}
    if envelope.get('schema_version')!=SCHEMA:raise ValueError('unsupported simulation schema')
    config=envelope['config']
    if args.sample_id:
        if args.config:raise ValueError('use a replay configuration or --sample-id, not both')
        config={'mode':'replay','sample_id':args.sample_id}
    root=args.source_root
    store=SampleStore(root/'data/processed/eval_samples',root/'data/splits/eval_masks.json',root/'data/splits/folds.json') if config.get('mode')=='replay' else None
    sessions=Sessions(store);session=sessions.create(config);capture=sessions.capture(session.identifier)
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('output must be empty; preserve prior captures')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'session.zip').write_bytes(sessions.bundle(session.identifier,args.include_reference))
    (args.output/'capture.json').write_text(json.dumps(capture.public(),indent=2,allow_nan=False)+'\n')
    (args.output/'configuration.json').write_text(json.dumps({'schema_version':SCHEMA,'config':session.config.document()},indent=2)+'\n')
    print(json.dumps({'status':'captured','capture_id':capture.metadata['capture_id'],'valid_pairs':int(capture.sample.arrays['rir_valid'].sum()),'output':str(args.output),'training_eligible':False}))

if __name__=='__main__':main()
