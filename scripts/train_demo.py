"""Freeze a small P9 room-group subset; train via the existing P8 runner."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from echospace.training.demo import freeze_demo
from echospace.training.fusion_runner import FusionTrainConfig, run_fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/p9_demo.json')
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--out-dir', type=Path, default=REPO / 'data/cache/p9_demo')
    freeze.add_argument('--eval-dir', type=Path, default=REPO / 'data/processed/eval_samples')
    freeze.add_argument('--folds', type=Path, default=REPO / 'data/splits/folds.json')
    freeze.add_argument('--eval-spec', type=Path, default=REPO / 'data/splits/eval_masks.json')
    run = sub.add_parser('run')
    run.add_argument('--frozen', type=Path, default=REPO / 'data/cache/p9_demo/manifest.json')
    run.add_argument('--out-dir', type=Path, default=REPO / 'artifacts/p9_demo')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    if args.command == 'freeze':
        demo = settings['demo']
        path = freeze_demo(args.out_dir, args.eval_dir, args.folds, args.eval_spec,
                           demo['groups'], demo['selection_seed'], demo['fold'])
        manifest = json.loads(path.read_text(encoding='utf-8'))
        print(json.dumps({'path': str(path), 'checksum': manifest['checksum'],
                          'counts': {s: sum(e['split'] == s for e in manifest['samples'])
                                     for s in ('train', 'val', 'test')}}), flush=True)
    else:
        config = dict(settings['training'])
        config['k_range'], config['eval_k'] = tuple(config['k_range']), tuple(config['eval_k'])
        for variant in settings['variants']:
            result = run_fusion(variant, args.frozen, args.out_dir / variant, FusionTrainConfig(**config),
                                settings['model'], progress=lambda row, v=variant:
                                print(json.dumps({'variant': v, **row}), flush=True))
            print(json.dumps({'variant': variant, 'status': result['status'], 'test': result['test_metrics']}), flush=True)


if __name__ == '__main__':
    main()
