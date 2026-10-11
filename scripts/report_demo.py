"""Verify P9 checkpoints, replay held-out scores, export predictions and demo plots."""
import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from echospace.contract_v0 import validate_prediction
from echospace.data.evalmasks import read_manifest
from echospace.training.audio_frozen import FrozenAudioDataset
from echospace.training.demo import paired_group_delta, room_metrics
from echospace.training.fusion_runner import (FusionTrainConfig, evaluate, load_fusion_checkpoint,
                                            mean_shape_baseline, _source_hashes)
from echospace.training.runner import atomic_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, default=REPO / 'data/cache/p9_demo/manifest.json')
    parser.add_argument('--runs', type=Path, default=REPO / 'artifacts/p9_demo')
    parser.add_argument('--eval-dir', type=Path, default=REPO / 'data/processed/eval_samples')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--out-dir', type=Path, help='fresh report destination; defaults to RUNS/demonstration')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    test = FrozenAudioDataset(args.frozen, 'test', k=8)
    records = read_manifest(args.eval_dir)
    ids = [e['metadata']['sample_id'] for e in test.entries]
    available = [int(test[i]['rir_valid'].sum()) for i in range(len(test))]
    chosen = sorted(np.random.default_rng(0).choice(len(test), min(10, len(test)), replace=False).tolist())
    out = args.out_dir or args.runs / 'demonstration'
    if out.exists() and any(out.iterdir()):
        raise ValueError('report destination must be empty; preserve earlier evidence')
    out.mkdir(parents=True, exist_ok=True)
    summary = {'format': 'p9.demo.1', 'scope': 'one fold, one seed, small fixed-mask subset; not the 75-run study',
               'frozen_checksum': test.checksum, 'source': test.manifest['source'],
               'splits': {}, 'variants': {}, 'paired_D_minus_A': {}, 'prediction_count': 0,
               'visual_selection': {'seed': 0, 'method': 'uniform without replacement, independent of scores',
                                    'sample_ids': [ids[i] for i in chosen]}}
    summary['test_available_pairs'] = {str(k): available.count(k) for k in sorted(set(available))}
    for split in ('train', 'val', 'test'):
        ds = FrozenAudioDataset(args.frozen, split, k=8)
        for i in range(len(ds)):
            ds[i]  # verify every frozen sample checksum
        summary['splits'][split] = {'samples': len(ds),
                                   'rooms': len({e['metadata']['room_id'] for e in ds.entries}),
                                   'groups': len({e['metadata']['room_group'] for e in ds.entries})}
    train = FrozenAudioDataset(args.frozen, 'train', k=8)
    summary['mean_shape_sample_weighted'] = mean_shape_baseline(train, test)
    replay_rows = {}
    probabilities = {}
    runs = {}
    for variant in ('A', 'P', 'S', 'C', 'D'):
        root = args.runs / variant
        run = json.loads((root / 'run.json').read_text(encoding='utf-8'))
        if run['status'] != 'complete' or run['frozen_checksum'] != test.checksum:
            raise ValueError(f'{variant} incomplete or frozen identity mismatch')
        if run['source_files_sha256'] != _source_hashes():
            raise ValueError(f'{variant} source files changed after training')
        if hashlib.sha256((root / 'best.pt').read_bytes()).hexdigest() != run['checkpoint_sha256']:
            raise ValueError(f'{variant} checkpoint checksum mismatch')
        if runs and any(run[key] != runs['A'][key] for key in ('config', 'source', 'versions')):
            raise ValueError('comparison configurations differ')
        runs[variant] = run
        seed_everything(run['config']['seed'])  # match training CUDA precision/determinism before loading
        model, payload = load_fusion_checkpoint(root / 'best.pt', test.checksum)
        if payload['epoch'] != run['best_epoch']:
            raise ValueError('selected checkpoint epoch mismatch')
        model.to(device).eval()
        cfg = dict(run['config'])
        cfg['k_range'], cfg['eval_k'] = tuple(cfg['k_range']), tuple(cfg['eval_k'])
        config = FusionTrainConfig(**cfg)
        result = {'best_epoch': run['best_epoch'], 'epochs_completed': len(run['history']),
                  'stop_reason': run['stop_reason'], 'parameters': run['parameter_count'],
                  'checkpoint_sha256': run['checkpoint_sha256'], 'test_room_weighted': {},
                  'maximum_score_replay_error': 0.0,
                  'test_sample_weighted': run['test_metrics'], 'by_coverage': {}, 'by_occlusion': {}}
        for k in config.eval_k:
            ds = FrozenAudioDataset(args.frozen, 'test', k=k)
            metrics, rows = evaluate(model, ds, config, device, keep_rows=True)
            recorded = json.loads((root / f'test_rows_K{k}.json').read_text(encoding='utf-8'))
            if recorded['frozen_checksum'] != test.checksum or recorded['checkpoint_sha256'] != run['checkpoint_sha256']:
                raise ValueError('metric rows not bound to selected checkpoint')
            if [r['sample_id'] for r in recorded['rows']] != ids:
                raise ValueError('test sample IDs differ')
            for actual, saved in zip(rows, recorded['rows']):
                if any(actual[key] != saved[key] for key in ('sample_id', 'room_id', 'room_group', 'k')):
                    raise ValueError('test row identity mismatch')
                error = max(abs(actual[key] - saved[key]) for key in ('loss', 'occupancy_iou', 'boundary_iou'))
                result['maximum_score_replay_error'] = max(result['maximum_score_replay_error'], error)
                if error > 1e-4:
                    raise ValueError('checkpoint score replay differs')
            if any(abs(metrics[key] - run['test_metrics'][f'K={k}'][key]) > 1e-4
                   for key in ('loss', 'occupancy_iou', 'boundary_iou')):
                raise ValueError('aggregate score replay differs')
            replay_rows[variant, k] = rows
            result['test_room_weighted'][f'K={k}'] = room_metrics(rows)
            for key, dest in (('coverage_bin', 'by_coverage'), ('occlusion_type', 'by_occlusion')):
                result[dest][f'K={k}'] = {label: room_metrics([r for r in rows if r[key] == label])
                                        for label in sorted({r[key] for r in rows})}
            export_dir = out / 'predictions' / variant / f'K{k}'
            export_dir.mkdir(parents=True)
            exports = []
            offset = 0
            with torch.inference_mode():
                for batch in DataLoader(ds, batch_size=config.batch_size, shuffle=False):
                    output = model({key: value.to(device) for key, value in batch.items()})
                    probs = {f'{head}_probability': output[f'{head}_logits'].float().sigmoid().cpu().numpy()
                             for head in ('occupancy', 'boundary')}
                    for j in range(len(batch['k'])):
                        index = offset + j
                        sid, meta = ids[index], ds.entries[index]['metadata']
                        arrays = {key: value[j] for key, value in probs.items()}
                        validate_prediction(records[sid], arrays)
                        path = export_dir / f'{index:06d}.npz'
                        np.savez_compressed(path, **arrays)
                        exports.append({'sample_id': sid, 'source_sample_sha256': ds.entries[index]['sha256'],
                                        'grid': meta['grid'], 'spatial_transform': meta['spatial_transform'],
                                        'actual_k': int(batch['k'][j]), 'path': path.name,
                                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                        if k == 8 and index in chosen:
                            probabilities[variant, index] = arrays
                    offset += len(batch['k'])
            atomic_json(export_dir / 'manifest.json', {'frozen_checksum': test.checksum, 'variant': variant,
                                                       'requested_k': k, 'checkpoint_sha256': run['checkpoint_sha256'],
                                                       'predictions': exports})
            summary['prediction_count'] += len(exports)
        summary['variants'][variant] = result
        print(f'{variant}: replayed all scores and exported {len(ids) * len(config.eval_k)} predictions', flush=True)
    for k in (1, 2, 4, 8):
        summary['paired_D_minus_A'][f'K={k}'] = paired_group_delta(replay_rows['A', k], replay_rows['D', k])
    summary['S_learning_check'] = {'S_sample_iou_K8': runs['S']['test_metrics']['K=8']['occupancy_iou'],
                                   'mean_shape_sample_iou': summary['mean_shape_sample_weighted']['occupancy_iou'],
                                   'scope': 'diagnostic only; do not retune on this test comparison'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for variant, run in runs.items():
        axes[0].plot([r['epoch'] for r in run['history']], [r['train_loss'] for r in run['history']], label=variant)
        axes[1].plot([r['epoch'] for r in run['history']],
                     [r['validation']['occupancy_iou'] for r in run['history']], label=variant)
    for ax, title in zip(axes, ('Training masked BCE + Dice', 'Validation missing-region occupancy IoU, K=8')):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'learning_curves.png', dpi=140); plt.close(fig)
    for page in range(0, len(chosen), 2):
        for head in ('occupancy', 'boundary'):
            fig, axes = plt.subplots(2, 7, figsize=(17, 5), squeeze=False)
            for row, index in enumerate(chosen[page:page+2]):
                sample = test[index]
                axes[row, 0].imshow(sample['observed_free'] + sample['observed_wall'] * 2, origin='lower', vmin=0, vmax=2)
                axes[row, 0].set_title(f"Observed\n{test.entries[index]['metadata']['room_id']}", fontsize=7)
                axes[row, 1].imshow(sample[f'target_{head}'], origin='lower', vmin=0, vmax=1)
                axes[row, 1].set_title(f'Target {head}')
                for column, variant in enumerate(('A', 'P', 'S', 'C', 'D'), 2):
                    axes[row, column].imshow(probabilities[variant, index][f'{head}_probability'], origin='lower', vmin=0, vmax=1)
                    axes[row, column].set_title(variant)
                for ax in axes[row]:
                    ax.contour(sample['observed_cells'], levels=[.5], colors='red', linewidths=.4)
                    ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle('P9 small-data demo: held-out K=8; red = observation outline; probabilities 0–1')
            fig.tight_layout(); fig.savefig(out / f'{head}_{page//2+1}.png', dpi=120); plt.close(fig)
    summary['gate'] = 'PASS'
    summary['gate_meaning'] = 'data integrity, completed training, checkpoint replay and prediction contract; not acoustic gain'
    atomic_json(out / 'summary.json', summary)
    table = ''.join('<tr><td>' + v + '</td><td>' + str(r['best_epoch']) + '</td><td>'
                    + f"{r['test_room_weighted']['K=8']['occupancy_iou']:.3f}" + '</td><td>'
                    + f"{r['test_room_weighted']['K=8']['boundary_iou']:.3f}" + '</td></tr>'
                    for v, r in summary['variants'].items())
    images = ''.join(f'<h2>Held-out examples {p}</h2><img src="occupancy_{p}.png" alt="Occupancy comparison {p}">'
                     f'<img src="boundary_{p}.png" alt="Boundary comparison {p}">' for p in range(1, 6))
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>P9 training demonstration</title>'
                '<style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 20px;background:#fafafa;'
                'color:#20252b}img{width:100%;height:auto}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}'
                'code{overflow-wrap:anywhere}table{border-collapse:collapse}</style><h1>P9 training demonstration</h1>'
                '<p>Five models trained on 20 training room groups, validated on 5 and tested on 5. '
                'One seed, fixed masks, no augmentation. This small run does not establish acoustic gain.</p>'
                '<p>A: geometry; P: geometry + positions; S: audio + positions; C: pooled fusion; D: EchoFusion.</p>'
                '<h2>Held-out results</h2><p>Equal room weight, valid unobserved cells, threshold 0.5, requested K=8. '
                'Actual K is capped by available verified pairs. Checkpoints were selected on validation only.</p>'
                '<table><tr><th>Model</th><th>Selected epoch</th><th>Occupancy IoU</th><th>Boundary IoU</th></tr>'
                + table + '</table><p>Frozen checksum: <code>' + html.escape(test.checksum) + '</code></p>'
                '<h2>Learning curves</h2><img src="learning_curves.png" alt="Training loss and validation IoU">'
                '<p>The following ten examples were selected uniformly with seed 0, independently of scores. '
                'Red marks the observation outline. Observed geometry remains separate from raw model probabilities.</p>'
                + images + '<p><a href="summary.json">Machine-readable results and provenance</a></p></html>')
    (out / 'index.html').write_text(document, encoding='utf-8')
    print(json.dumps({'gate': summary['gate'], 'predictions': summary['prediction_count'], 'report': str(out)}), flush=True)


if __name__ == '__main__':
    main()
