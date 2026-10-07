"""P7 mask leakage, position controls, frozen identity and training lifecycle."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from echospace.models import SpatialConfig, SpatialUNet, WallConfig, WallExtrapolation
from echospace.models.baselines import extrapolate_walls
from echospace.training.frozen import FrozenDataset, freeze_samples
from echospace.training.losses import masked_bce_dice
from echospace.training.runner import TrainConfig, load_checkpoint, run_baseline, seed_everything


def item(room, sample=0):
    seen = np.zeros((64, 64), np.uint8)
    free = seen.copy()
    free[25:32, 20:36] = 1
    wall = seen.copy()
    wall[24, 20:36] = 1
    seen = free | wall
    target = seen.copy()
    target[24:44, 19:40] = 1
    boundary = np.zeros_like(seen)
    boundary[24, 19:40] = boundary[43, 19:40] = 1
    boundary[24:44, 19] = boundary[24:44, 39] = 1
    src, mic = np.zeros((8, 3), np.float32), np.zeros((8, 3), np.float32)
    src[:2] = [[4.1, 5.1, 1.2], [4.5, 5.1, 1.2]]
    mic[:2] = [[5.1, 5.1, 1.3], [5.5, 5.3, 1.3]]
    from echospace.geometry.frames import GridFrame
    return dict(observed_cells=seen, observed_free=free, observed_wall=wall,
                valid_cells=np.ones_like(seen), target_occupancy=target, target_boundary=boundary,
                src_pos=src, mic_pos=mic, rir_valid=np.array([1, 1, 0, 0, 0, 0, 0, 0], np.uint8),
                sample_id=f"room{room}_sample{sample}", room_id=f"room{room}", room_group=room,
                grid=GridFrame((0, 1.1, 0), 0).manifest_grid(),
                spatial_transform={"quarter_turns": 0, "flip_columns": False})


def batch():
    return {k: torch.from_numpy(v)[None] for k, v in item(0).items() if isinstance(v, np.ndarray)}


def fixture_freeze(root):
    return freeze_samples(root, {"train": [item(0), item(0, 1)], "val": [item(1)], "test": [item(2)]},
                          {"kind": "synthetic_test"})


@pytest.fixture(autouse=True)
def small_cpu():
    torch.set_num_threads(2)


def test_masked_loss_ignores_observed_and_invalid_targets_and_has_zero_gradient_there():
    b = batch()
    b["valid_cells"][:, 0] = 0
    logits = torch.randn(1, 64, 64, requires_grad=True)
    output = {"occupancy_logits": logits, "boundary_logits": logits}
    loss = masked_bce_dice(output, b)
    outside = (b["observed_cells"] == 1) | (b["valid_cells"] == 0)
    changed = dict(b)
    for head in ("occupancy", "boundary"):
        changed[f"target_{head}"] = b[f"target_{head}"].float().clone()
        changed[f"target_{head}"][outside] = float("nan")
    assert torch.equal(loss, masked_bce_dice(output, changed))
    loss.backward()
    assert torch.all(logits.grad[outside] == 0)
    assert torch.any(logits.grad[~outside] != 0)


def test_loss_all_observed_is_differentiable_zero_and_empty_boundary_is_finite():
    b = batch()
    b["observed_cells"][:] = 1
    logits = torch.zeros(1, 64, 64, requires_grad=True)
    loss = masked_bce_dice({"occupancy_logits": logits, "boundary_logits": logits}, b)
    assert loss == 0
    loss.backward()
    assert torch.all(logits.grad == 0)
    b["observed_cells"][:] = 0
    b["target_boundary"][:] = 0
    assert torch.isfinite(masked_bce_dice({"occupancy_logits": logits, "boundary_logits": logits}, b))


@pytest.mark.parametrize("variant", ["B0", "A", "P"])
def test_models_do_not_read_targets_waveforms_or_identity(variant):
    seed_everything(2)
    model = WallExtrapolation() if variant == "B0" else SpatialUNet(variant, SpatialConfig(base_channels=4))
    model.eval()
    b = batch()
    changed = dict(b)
    changed.update(target_occupancy=torch.randn(1, 64, 64), target_boundary=torch.randn(1, 64, 64),
                   valid_cells=torch.zeros(1, 64, 64), rir=torch.full((1, 8, 1280), float("nan")), room_id="hidden")
    with torch.no_grad():
        first, second = model(b), model(changed)
    for key in first:
        assert first[key].shape == (1, 64, 64)
        assert torch.equal(first[key], second[key])


def test_a_ignores_positions_p_is_permutation_invariant_and_invalid_rows_are_ignored():
    seed_everything(3)
    b = batch()
    a, p = SpatialUNet("A", SpatialConfig(base_channels=4)), SpatialUNet("P", SpatialConfig(base_channels=4))
    changed = {k: v.clone() for k, v in b.items()}
    for key in ("src_pos", "mic_pos"):
        changed[key][:, 2:] = float("nan")
    with torch.no_grad():
        reference = p(b)["occupancy_logits"]
        assert torch.equal(reference, p(changed)["occupancy_logits"])
        order = torch.tensor([1, 0, 7, 6, 5, 4, 3, 2])
        for key in ("src_pos", "mic_pos", "rir_valid"):
            changed[key] = changed[key][:, order]
        assert torch.allclose(reference, p(changed)["occupancy_logits"], atol=1e-6)
        moved = {k: v.clone() for k, v in b.items()}
        moved["src_pos"][:, :2, 0] += 0.4
        assert torch.equal(a(b)["occupancy_logits"], a(moved)["occupancy_logits"])
        assert not torch.allclose(reference, p(moved)["occupancy_logits"])
        moved["rir_valid"][:] = 0
        assert torch.isfinite(p(moved)["occupancy_logits"]).all()


def test_b0_extrapolates_only_observed_runs_and_stops_at_observed_free():
    free, wall = np.zeros((64, 64)), np.zeros((64, 64))
    wall[20, 20:24] = 1
    free[20, 27] = 1
    occupancy, boundary = extrapolate_walls(free, wall, WallConfig(extension_cells=8))
    assert boundary[20, 26] == 1 and boundary[20, 27] == 0 and boundary[20, 28] == 0
    assert boundary[20, 12] == 1 and boundary[20, 11] == 0
    assert occupancy[20, 27] == 1
    assert not boundary[19].any()


def test_frozen_content_checksums_overwrites_and_split_leakage(tmp_path):
    path = fixture_freeze(tmp_path / "ok")
    data = FrozenDataset(path, "test")
    assert len(data) == 1
    _ = data[0]
    entry = data.entries[0]
    file = path.parent / entry["path"]
    with np.load(file) as npz:
        arrays = dict(npz)
    arrays["target_boundary"][0, 0] = 1
    np.savez_compressed(file, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        _ = data[0]
    with pytest.raises(ValueError, match="empty"):
        fixture_freeze(path.parent)
    with pytest.raises(ValueError, match="crosses"):
        freeze_samples(tmp_path / "bad", {"train": [item(0)], "val": [item(0, 1)], "test": [item(2)]}, {})


def test_checkpoint_early_stopping_reload_and_reproducibility(tmp_path):
    path = fixture_freeze(tmp_path / "frozen")
    config = TrainConfig(epochs=4, patience=1, min_delta=100, batch_size=2, amp=True)
    first = run_baseline("A", path, tmp_path / "first", config, SpatialConfig(base_channels=4))
    second = run_baseline("A", path, tmp_path / "second", config, SpatialConfig(base_channels=4))
    assert first["stop_reason"] == "early_stopping" and first["best_epoch"] == 1
    assert len(first["history"]) == 2 and first["history"] == second["history"]
    assert not first["amp_enabled"]  # explicit CPU fallback
    model, payload = load_checkpoint(tmp_path / "first/best.pt", first["frozen_checksum"])
    last = torch.load(tmp_path / "first/last.pt", weights_only=True)
    assert payload["epoch"] == 1 and last["epoch"] == 2 and "optimizer" in payload and "torch_rng_state" in payload
    sample = FrozenDataset(path, "test")[0]
    with torch.no_grad():
        prediction = model({k: v[None] for k, v in sample.items()})["occupancy_logits"].sigmoid().numpy()[0]
    with np.load(tmp_path / "first/predictions_test/000000.npz") as result:
        assert np.allclose(prediction, result["occupancy_probability"])
    assert first["split_sample_ids"] == second["split_sample_ids"]
    with pytest.raises(ValueError, match="another"):
        load_checkpoint(tmp_path / "first/best.pt", "wrong")


def test_test_targets_do_not_select_checkpoint(tmp_path):
    path1 = fixture_freeze(tmp_path / "frozen1")
    changed = item(2)
    changed["target_occupancy"] = 1 - changed["target_occupancy"]
    changed["target_boundary"] = 1 - changed["target_boundary"]
    path2 = freeze_samples(tmp_path / "frozen2", {"train": [item(0), item(0, 1)], "val": [item(1)], "test": [changed]}, {})
    config = TrainConfig(epochs=1, batch_size=2)
    a = run_baseline("P", path1, tmp_path / "run1", config, SpatialConfig(base_channels=4))
    b = run_baseline("P", path2, tmp_path / "run2", config, SpatialConfig(base_channels=4))
    assert a["history"] == b["history"]
    m1, _ = load_checkpoint(tmp_path / "run1/best.pt")
    m2, _ = load_checkpoint(tmp_path / "run2/best.pt")
    assert all(torch.equal(m1.state_dict()[k], m2.state_dict()[k]) for k in m1.state_dict())
    assert a["test_metrics"]["loss"] != b["test_metrics"]["loss"]
