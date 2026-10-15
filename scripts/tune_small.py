"""P9 bounded validation-only search, seed confirmation and locked final holdout."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from echospace.data import load_folds, load_eval_spec
from echospace.training.audio_frozen import fixed_items, write_freeze
from echospace.training.demo import room_metrics, select_groups
from echospace.training.frozen import json_sha
from echospace.training.fusion_runner import FusionTrainConfig, run_fusion
from echospace.training.runner import atomic_json


def freeze(root, settings):
    folds_path, spec_path = REPO/'data/splits/folds.json', REPO/'data/splits/eval_masks.json'
    folds, spec = load_folds(folds_path), load_eval_spec(spec_path)
    view, selected = select_groups(folds, 0, {'train': 20, 'val': 5, 'test': 5}, 0)
    prior = selected['test']['groups']
    candidates = [g for g in folds['folds'][0]['test_groups'] if g not in prior]
    ranked = sorted(candidates, key=lambda g: hashlib.sha256(f"p9-holdout|{settings['holdout']['seed']}|{g}".encode()).digest())
    chosen = sorted(ranked[:settings['holdout']['groups']])
    rooms = sorted(r for r in folds['folds'][0]['test_rooms'] if folds['rooms'][r]['room_group'] in chosen)
    view['folds'][0]['test_groups'], view['folds'][0]['test_rooms'] = chosen, rooms
    selected['test'] = {'groups': chosen, 'rooms': rooms}
    source = {'kind': 'P9_validation_search_fixed_masks', 'selection': selected,
              'selection_checksum': json_sha(selected), 'fold': 0, 'training_augmentation': 'none',
              'folds_sha256': hashlib.sha256(folds_path.read_bytes()).hexdigest(), 'eval_spec_checksum': spec['checksum'],
              'excluded_previous_P9_test_groups': prior,
              'holdout_scope': 'new to this tuning round; broader fold-0 test pool previously evaluated in P7'}
    return write_freeze(root, {s: fixed_items(REPO/'data/processed/eval_samples', view, spec, 0, s)
                               for s in ('train', 'val', 'test')}, source)


def run_trial(trial, seed, frozen, root, base, settings):
    model = copy.deepcopy(base['model'])
    if trial.get('compact'):
        model.update(settings['compact_model'])
    model.update(trial['model'])
    training = dict(base['training'], evaluate_test=False, room_weighted_selection=True,
                    selection_metric='mean_head_iou', seed=seed)
    training.update(trial['training'])
    training['k_range'], training['eval_k'] = tuple(training['k_range']), tuple(training['eval_k'])
    result = run_fusion(trial['variant'], frozen, root, FusionTrainConfig(**training), model,
                        progress=lambda row: print(json.dumps({'trial': trial['id'], 'seed': seed,
                        'epoch': row['epoch'], 'train_loss': row['train_loss'], 'validation_score': row['selection_score']}), flush=True))
    selected = next(row for row in result['history'] if row['epoch'] == result['best_epoch'])
    return {'trial': trial['id'], 'seed': seed, 'path': str(root.relative_to(REPO)),
            'best_epoch': result['best_epoch'], 'epochs_completed': len(result['history']),
            'score': selected['selection_score'], 'validation': selected['validation'],
            'checkpoint_sha256': result['checkpoint_sha256'], 'parameters': result['parameter_count']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'search'))
    p.add_argument('--config', type=Path, default=REPO/'configs/p9_iterations.json')
    args = p.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    frozen = REPO/'data/cache/p9_iterations/manifest.json'
    root = REPO/'artifacts/p9_iterations'
    if args.command == 'freeze':
        print(freeze(frozen.parent, settings), flush=True)
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError('search destination must be empty; recorded trials cannot be overwritten')
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads((REPO/settings['base_config']).read_text(encoding='utf-8'))
    study = {'format': 'p9.search.1', 'status': 'running', 'settings': settings,
             'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
             'frozen_checksum': json.loads(frozen.read_text(encoding='utf-8'))['checksum'],
             'objective': settings['selection']['metric'], 'trials': [], 'holdout_read_for_selection': False}
    atomic_json(root/'study.json', study)
    for trial in settings['trials']:
        study['trials'].append(run_trial(trial, 0, frozen, root/trial['id']/'seed0', base, settings))
        atomic_json(root/'study.json', study)
    ranked = sorted(study['trials'], key=lambda r: (-r['score'], r['trial']))[:2]
    study['confirmation_candidates'] = [r['trial'] for r in ranked]
    atomic_json(root/'study.json', study)
    for candidate in ranked:
        trial = next(t for t in settings['trials'] if t['id'] == candidate['trial'])
        study['trials'].append(run_trial(trial, 1, frozen, root/trial['id']/'seed1', base, settings))
        atomic_json(root/'study.json', study)
    means = {r['trial']: sum(t['score'] for t in study['trials'] if t['trial']==r['trial'])/2 for r in ranked}
    winner = sorted(means, key=lambda name: (-means[name], name))[0]
    study.update(status='selection_locked', confirmation_mean_scores=means, winner=winner,
                 deploy_seed=0, winner_path=next(t['path'] for t in study['trials'] if t['trial']==winner and t['seed']==0))
    atomic_json(root/'study.json', study)
    print(json.dumps({'status': study['status'], 'winner': winner, 'validation_means': means}), flush=True)


if __name__ == '__main__':
    main()
