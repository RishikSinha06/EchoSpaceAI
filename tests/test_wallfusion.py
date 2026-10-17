"""Physical correctness, forbidden-input isolation and checkpoint reproducibility."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from echospace.models.wallfusion import build, reflection_delays, sample_delay_features
from echospace.models.factory import from_specification
from echospace.training.audio_frozen import write_freeze
from echospace.training.synthetic_audio import synthetic_room, synthetic_splits
from echospace.training.fusion_runner import FusionTrainConfig, run_fusion, load_fusion_checkpoint

SMALL = dict(family='WallFusion', width=32, channels=[8, 16, 32, 32], audio_channels=[8, 16, 32, 32],
             feedforward=64, fusion_layers=1, heads=4, dropout=.1, temporal_channels=4,
             temporal_tokens=8, wall_resolution=8, wall_orientations=4, wall_chunk=16)


def batch():
    items = [synthetic_room(1, 0, k=3), synthetic_room(2, 0, k=1)]
    return {key: torch.tensor(np.stack([i[key] for i in items])).float()
            for key in ('observed_free', 'observed_wall', 'observed_cells', 'src_pos', 'mic_pos', 'rir_valid', 'rir')}


def model(variant='D', **changes):
    config = dict(SMALL, **changes)
    config.pop('family')
    return build(variant, config)


def test_reflection_time_height_sidedness_rotation_and_reciprocity():
    s = torch.tensor([[[1., 2., 1.]]])
    m = torch.tensor([[[2., 2., 2.]]])
    p = torch.tensor([[5., 3.]])
    n = torch.tensor([[1., 0.]])
    delay, legal = reflection_delays(s, m, p, n)
    assert delay.item() == pytest.approx(math.sqrt(50)/343, rel=1e-6)
    assert legal.item()
    assert torch.equal(delay, reflection_delays(m, s, p, n)[0])
    assert not reflection_delays(s, torch.tensor([[[6., 2., 2.]]]), p, n)[1].item()
    # Rotate by 90 degrees and translate both poses and plane together.
    def transform(x):
        y = x.clone()
        y[..., 0], y[..., 1] = -x[..., 1]+7, x[..., 0]+3
        return y
    assert torch.allclose(delay, reflection_delays(transform(s), transform(m), transform(p), torch.tensor([[0., 1.]]))[0])


def test_fractional_delay_interpolation_and_gradients():
    f = torch.arange(10.).reshape(1, 1, 1, 10).requires_grad_()
    delay = torch.tensor([[[[2.5 * 4/16000]]]], requires_grad=True)
    result = sample_delay_features(f, delay)
    assert result.item() == pytest.approx(2.5)
    result.sum().backward()
    assert torch.allclose(f.grad.flatten()[2:4], torch.tensor([.5, .5]))
    assert delay.grad.item() == pytest.approx(4000.)
    assert sample_delay_features(f, torch.tensor([[[[1.]]]])).item() == 0


@pytest.mark.parametrize('variant', ['P', 'D'])
def test_pair_order_invalid_nan_rows_and_targets_do_not_change_output(variant):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    net, x = model(variant).eval(), batch()
    with torch.no_grad():
        baseline = net(x)['occupancy_logits']
        changed = {k: v.clone() for k, v in x.items()}
        for key in ('rir', 'src_pos', 'mic_pos'):
            changed[key][~x['rir_valid'].bool()] = float('nan')
        changed['target_occupancy'] = torch.randn(2, 64, 64)
        changed['valid_cells'] = torch.zeros(2, 64, 64)
        assert torch.equal(baseline, net(changed)['occupancy_logits'])
        order = torch.tensor([7, 0, 2, 4, 1, 3, 5, 6])
        for key in ('rir', 'src_pos', 'mic_pos', 'rir_valid'):
            changed[key] = x[key][:, order]
        assert torch.allclose(baseline, net(changed)['occupancy_logits'], atol=3e-6)
        x['rir'] = torch.randn_like(x['rir'])
        same = torch.allclose(baseline, net(x)['occupancy_logits'], atol=1e-6)
        assert same == (variant == 'P')


def test_zero_pairs_is_finite_and_serialization_exact():
    net, x = model().eval(), batch()
    x['rir_valid'].zero_()
    x['rir'].fill_(float('nan'))
    clone = from_specification(net.specification()).eval()
    clone.load_state_dict(net.state_dict(), strict=True)
    with torch.no_grad():
        result = net(x)
        assert torch.isfinite(result['occupancy_logits']).all()
        assert torch.equal(result['occupancy_logits'], clone(x)['occupancy_logits'])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_training_resume_is_exact_and_both_families_load(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('no CUDA')
    frozen = write_freeze(tmp_path/'data', synthetic_splits(2, 1, 1, 1), {'kind': 'synthetic'})
    cfg = FusionTrainConfig(epochs=2, batch_size=2, device=device, warmup_epochs=1,
                           evaluate_test=False, room_weighted_selection=True)
    whole = run_fusion('D', frozen, tmp_path/'whole', cfg, SMALL)
    def stop(row):
        if row['epoch'] == 1:
            raise RuntimeError('interrupted test')
    with pytest.raises(RuntimeError, match='interrupted test'):
        run_fusion('D', frozen, tmp_path/'resume', cfg, SMALL, progress=stop)
    resumed = run_fusion('D', frozen, tmp_path/'resume', cfg, SMALL, resume=True)
    assert resumed['history'] == whole['history']
    first, _ = load_fusion_checkpoint(tmp_path/'whole/best.pt')
    second, _ = load_fusion_checkpoint(tmp_path/'resume/best.pt')
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    assert resumed['test_evaluation_deferred']
