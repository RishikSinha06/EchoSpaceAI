"""P9 selection and aggregation checks independent of real data."""
import pytest
pytest.importorskip('torch')
from echospace.training.demo import paired_group_delta, room_metrics, select_groups


def test_selection_is_deterministic_and_keeps_original_partitions():
    fold = {f'{s}_groups': list(range(i * 4, i * 4 + 4)) for i, s in enumerate(('train', 'val', 'test'))}
    rooms = {f'r{g}_{j}': {'room_group': g} for g in range(12) for j in range(2)}
    for s in ('train', 'val', 'test'):
        fold[f'{s}_rooms'] = [r for r, info in rooms.items() if info['room_group'] in fold[f'{s}_groups']]
    folds = {'folds': [fold], 'rooms': rooms}
    counts = {'train': 2, 'val': 1, 'test': 1}
    _, selected = select_groups(folds, 0, counts)
    assert selected == select_groups(folds, 0, counts)[1]
    assert len(folds['folds'][0]['train_groups']) == 4
    for s in counts:
        assert len(selected[s]['groups']) == counts[s]
        assert len(selected[s]['rooms']) == counts[s] * 2
        assert set(selected[s]['rooms']) <= set(fold[f'{s}_rooms'])
    with pytest.raises(ValueError):
        select_groups(folds, 0, {'train': 5, 'val': 1, 'test': 1})


def row(sid, room, group, iou):
    return dict(sample_id=sid, room_id=room, room_group=group, k=8,
                occupancy_iou=iou, boundary_iou=iou, loss=1-iou)


def test_equal_room_weight_and_paired_group_identity():
    a = [row('a', 'one', 0, 0), row('b', 'one', 0, 0), row('c', 'two', 1, 1)]
    assert room_metrics(a)['occupancy_iou'] == 0.5
    d = [dict(r, occupancy_iou=r['occupancy_iou'] + 0.1) for r in a]
    summary = paired_group_delta(a, d, repeats=20)
    assert summary['mean_room_delta'] == pytest.approx(0.1)
    assert summary['groups'] == 2
    assert summary == paired_group_delta(a, d, repeats=20)
    with pytest.raises(ValueError):
        paired_group_delta(a, d[:-1])
