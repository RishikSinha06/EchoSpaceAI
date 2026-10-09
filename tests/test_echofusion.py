"""P8 EchoFusion: model guarantees, audio freeze, random K and the runner (synthetic only)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from echospace.models.echofusion import VARIANTS, AcousticEncoder, EchoFusion, FusionConfig, build  # noqa: E402
from echospace.training.audio_frozen import FrozenAudioDataset, write_freeze  # noqa: E402
from echospace.training.fusion_runner import (  # noqa: E402
    FusionTrainConfig,
    load_fusion_checkpoint,
    mean_shape_baseline,
    run_fusion,
    schedule,
)
from echospace.training.synthetic_audio import synthetic_room, synthetic_splits  # noqa: E402

SMALL = {"width": 32, "channels": (8, 16, 32, 32), "audio_channels": (8, 16, 32, 32), "fusion_layers": 1,
         "heads": 4, "feedforward": 64, "dropout": 0.0}


def batch_of(items: list[dict]) -> dict[str, torch.Tensor]:
    keys = ("observed_free", "observed_wall", "observed_cells", "rir", "src_pos", "mic_pos", "rir_valid")
    return {k: torch.as_tensor(np.stack([it[k] for it in items])).float() for k in keys}


@pytest.fixture(scope="module")
def batch() -> dict[str, torch.Tensor]:
    items = [synthetic_room(r, 0, k=k) for r, k in ((1, 8), (2, 3), (3, 1))]
    return batch_of(items)


def test_default_echofusion_size_and_shapes(batch: dict[str, torch.Tensor]) -> None:
    model = EchoFusion("D").eval()
    assert 2_000_000 <= sum(p.numel() for p in model.parameters()) <= 4_000_000  # plan: 2-4 M
    out = model(batch)
    assert out["occupancy_logits"].shape == out["boundary_logits"].shape == (3, 64, 64)
    assert torch.isfinite(out["occupancy_logits"]).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_invalid_rows_and_measurement_order_never_change_the_output(variant: str, batch: dict[str, torch.Tensor]) -> None:
    torch.manual_seed(0)
    model = build(variant, SMALL).eval()
    with torch.no_grad():
        base = model(batch)["occupancy_logits"]
        noisy = {k: v.clone() for k, v in batch.items()}
        invalid = noisy["rir_valid"] == 0
        noisy["rir"][invalid] = torch.randn_like(noisy["rir"][invalid]) * 9
        noisy["src_pos"][invalid] = 7.0
        assert torch.allclose(model(noisy)["occupancy_logits"], base, atol=1e-5)
        shuffled = {k: v.clone() for k, v in batch.items()}
        order = torch.tensor([2, 0, 1, 3, 4, 5, 6, 7])
        for key in ("rir", "src_pos", "mic_pos", "rir_valid"):
            shuffled[key][0] = batch[key][0][order]
        assert torch.allclose(model(shuffled)["occupancy_logits"], base, atol=1e-4)


def test_each_variant_reads_only_its_inputs(batch: dict[str, torch.Tensor]) -> None:
    torch.manual_seed(0)
    changed_audio = {k: v.clone() for k, v in batch.items()}
    changed_audio["rir"] = torch.where(changed_audio["rir_valid"][..., None].bool(), torch.randn_like(batch["rir"]), batch["rir"])
    moved = {k: v.clone() for k, v in batch.items()}
    moved["src_pos"][..., 2] += 0.5  # heights only: stays in observed free space
    changed_geometry = {k: v.clone() for k, v in batch.items()}
    changed_geometry["observed_wall"][:, 0, :] = 1
    changed_geometry["observed_cells"][:, 0, :] = 1
    reads = {}
    for variant in VARIANTS:
        model = build(variant, SMALL).eval()
        with torch.no_grad():
            base = model(batch)["occupancy_logits"]
            reads[variant] = {name: not torch.allclose(model(other)["occupancy_logits"], base, atol=1e-6)
                              for name, other in (("audio", changed_audio), ("positions", moved), ("geometry", changed_geometry))}
    assert reads["A"] == {"audio": False, "positions": False, "geometry": True}
    assert reads["P"] == {"audio": False, "positions": True, "geometry": True}
    assert reads["S"] == {"audio": True, "positions": True, "geometry": False}
    assert reads["C"] == reads["D"] == {"audio": True, "positions": True, "geometry": True}


def test_audio_encoder_sees_arrival_time() -> None:
    # Regression: conv + global average pooling alone maps an echo at t and at
    # t + 400 samples to the same features, so delay cannot reach the model.
    torch.manual_seed(0)
    early, late = torch.zeros(1, 1, 1280), torch.zeros(1, 1, 1280)
    pulse = torch.exp(-0.5 * (torch.arange(-12, 13) / 3.0) ** 2)
    early[0, 0, 288:313], late[0, 0, 688:713] = pulse, pulse
    with torch.no_grad():
        blind = AcousticEncoder(FusionConfig(time_channel=False)).eval()
        timed = AcousticEncoder(FusionConfig(time_channel=True)).eval()
        assert (blind.encode_audio(early) - blind.encode_audio(late)).abs().max() < 1e-4
        assert (timed.encode_audio(early) - timed.encode_audio(late)).abs().max() > 1e-3


def test_every_sample_needs_one_measurement(batch: dict[str, torch.Tensor]) -> None:
    empty = {k: v.clone() for k, v in batch.items()}
    empty["rir_valid"][1] = 0
    with pytest.raises(ValueError, match="at least one valid"):
        build("D", SMALL)(empty)


# --- frozen audio samples and random K ---------------------------------------------

@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory) -> Path:
    splits = synthetic_splits(train_rooms=6, val_rooms=2, test_rooms=2, masks_per_room=2)
    return write_freeze(tmp_path_factory.mktemp("p8") / "frozen", splits, {"kind": "test"})


def test_random_k_is_a_reproducible_nested_prefix(frozen: Path) -> None:
    train = FrozenAudioDataset(frozen, "train", random_k=(1, 8), seed=3)
    full = FrozenAudioDataset(frozen, "train", k=8)
    ks = []
    for epoch in range(6):
        train.set_epoch(epoch)
        for i in range(len(train)):
            item, whole = train[i], full[i]
            k = int(item["k"])
            ks.append(k)
            assert int(item["rir_valid"].sum()) == k
            assert torch.equal(item["rir"][:k], whole["rir"][:k]) and not item["rir"][k:].any()
            assert not item["src_pos"][k:].any() and not item["mic_pos"][k:].any()
    assert set(ks) == set(range(1, 9))
    again = FrozenAudioDataset(frozen, "train", random_k=(1, 8), seed=3)
    again.set_epoch(5)
    assert [int(again[i]["k"]) for i in range(len(again))] == ks[-len(again):]


def test_frozen_audio_refuses_tampering_and_split_crossing(frozen: Path, tmp_path: Path) -> None:
    manifest = json.loads(frozen.read_text())
    entry = manifest["samples"][0]
    path = frozen.parent / entry["path"]
    original = path.read_bytes()
    with np.load(path) as data:
        arrays = dict(data)
    arrays["rir"] = arrays["rir"].copy()
    arrays["rir"][0, 0] += np.float16(0.5)
    np.savez_compressed(path, **arrays)
    try:
        with pytest.raises(ValueError, match="checksum"):
            FrozenAudioDataset(frozen, "train", k=8)[0]
    finally:
        path.write_bytes(original)
    crossing = synthetic_splits(train_rooms=2, val_rooms=1, test_rooms=1, masks_per_room=1)
    crossing["test"][0] = dict(crossing["test"][0], room_id=crossing["train"][0]["room_id"])
    with pytest.raises(ValueError, match="crosses"):
        write_freeze(tmp_path / "bad", crossing, {"kind": "test"})


# --- runner -------------------------------------------------------------------------

def test_schedule_warms_up_then_decays() -> None:
    factor = schedule(FusionTrainConfig(epochs=10, warmup_epochs=2), steps_per_epoch=5)
    values = [factor(step) for step in range(50)]
    assert values[0] == pytest.approx(0.1) and values[9] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(values[10:], values[11:])) and values[-1] < 0.01
    assert math.isclose(factor(10), 1.0)


def test_run_selects_on_validation_and_scores_test_at_each_k(frozen: Path, tmp_path: Path) -> None:
    config = FusionTrainConfig(epochs=2, batch_size=4, warmup_epochs=1, eval_k=(1, 8), threads=2)
    run = run_fusion("D", frozen, tmp_path / "D", config, SMALL)
    assert run["status"] == "complete" and len(run["history"]) == 2
    assert set(run["test_metrics"]) == {"K=1", "K=8"}
    rows = json.loads((tmp_path / "D" / "test_rows_K1.json").read_text())["rows"]
    assert rows and {r["k"] for r in rows} == {1}
    model, payload = load_fusion_checkpoint(tmp_path / "D" / "best.pt", run["frozen_checksum"])
    assert payload["epoch"] == run["best_epoch"] and model.variant == "D"
    with pytest.raises(ValueError, match="another frozen"):
        load_fusion_checkpoint(tmp_path / "D" / "best.pt", "0" * 64)
    with pytest.raises(ValueError, match="empty"):
        run_fusion("D", frozen, tmp_path / "D", config, SMALL)
    trivial = mean_shape_baseline(FrozenAudioDataset(frozen, "train", k=8), FrozenAudioDataset(frozen, "test", k=8))
    assert 0.0 <= trivial["occupancy_iou"] <= 1.0
