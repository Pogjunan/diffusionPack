from copy import deepcopy
import importlib

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from diffusionpack import diffusion
from diffusionpack.backbone.prediction_unet import PredictionUNet
from diffusionpack.scrd.unet import SCRDUNet
from diffusionpack.main.prediction_data import ForecastDataset
from diffusionpack.main.train import load_checkpoint
from diffusionpack.main.train_prediction import train_prediction
from diffusionpack.scrd.prediction import scrd_prediction_loss, scrd_prediction_sample
from diffusionpack.prediction_config import PredictionInferenceConfig, PredictionTrainConfig

inference = importlib.import_module("diffusionpack.main.inference_prediction")


def write_maps(folder, count=8):
    folder.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pattern = np.linspace(0, .05, 64, dtype=np.float32).reshape(8, 8)
    rows = []
    for index in range(count):
        time = start + pd.Timedelta(minutes=30 * index)
        values = np.stack([pattern + .05 * index, pattern + .05 * index + .02]).astype(np.float32)
        np.save(folder / f"{index}.npy", values)
        rows.append(dict(timestamp=time.isoformat(), path=f"{index}.npy", available_at=time.isoformat()))
    path = folder / "frames.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def options(csv_path, output):
    return dict(csv_path=str(csv_path), backbone="scrd", resolution=8,
                history_steps=3, future_steps=2, map_channels=2,
                value_min=0, value_max=1, units="synthetic units",
                scrd_widths=(8, 16), scrd_window=4, scrd_tcu_depth=1,
                scrd_time_dim=16, scrd_norm_groups=4, scrd_output_norm_groups=4,
                batch_size=2, epochs=None, warmup_steps=0, amp=False,
                device="cpu", progress=False, output=str(output))


@pytest.fixture
def small_scrd_run(tmp_path):
    csv_path = write_maps(tmp_path / "train")
    target_csv = tmp_path / "targets.csv"
    pd.read_csv(csv_path).iloc[3:][["timestamp"]].to_csv(target_csv, index=False)
    config = options(csv_path, tmp_path / "run")
    config["target_csv"] = str(target_csv)
    path = train_prediction(PredictionTrainConfig(steps=1, **config))
    return config, path


def test_scrd_loss_calls_core_ddpm_on_the_correct_multichannel_residual(monkeypatch):
    history = torch.randn(2, 6, 2, 3)
    future = torch.randn(2, 6, 2, 3)
    last = history[:, -2:].repeat(1, 3, 1, 1)
    residual = future - last
    epsilon = torch.linspace(-1, 1, future.numel()).reshape_as(future)
    timesteps = torch.tensor([0, 999])
    calls = {"loss": 0, "q_sample": 0}
    old_loss, old_q = diffusion.ddpm_loss, diffusion.q_sample

    def core_loss(model, clean):
        calls["loss"] += 1
        torch.testing.assert_close(clean, residual)
        return old_loss(model, clean)

    def core_q(clean, t, noise):
        calls["q_sample"] += 1
        torch.testing.assert_close(clean, residual)
        torch.testing.assert_close(noise, epsilon)
        return old_q(clean, t, noise)

    class Denoiser(nn.Module):
        map_channels, future_steps = 2, 3

        def __init__(self):
            super().__init__()
            self.error = nn.Parameter(torch.tensor(.25))

        def forward(self, noisy, t, condition):
            assert condition is history and t.dtype == torch.int64
            assert t.tolist() == [0, 999]
            alpha = diffusion.alpha_bars[t].float().reshape(2, 1, 1, 1)
            torch.testing.assert_close(noisy, alpha.sqrt() * residual + (1 - alpha).sqrt() * epsilon)
            return epsilon + self.error

    monkeypatch.setattr(diffusion, "ddpm_loss", core_loss)
    monkeypatch.setattr(diffusion, "q_sample", core_q)
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: timesteps)
    monkeypatch.setattr(torch, "randn_like", lambda tensor: epsilon)
    model = Denoiser()
    loss = scrd_prediction_loss(model, future, history)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(.0625))
    torch.testing.assert_close(model.error.grad, torch.tensor(.5))
    assert calls == {"loss": 1, "q_sample": 1}


def test_scrd_sample_uses_all_core_steps_with_one_condition_cache(monkeypatch):
    history = torch.tensor([.1, .2, .3, .4, .5, .6]).reshape(1, 6, 1, 1).expand(-1, -1, 2, 3).clone()
    clean = torch.tensor([1.5, -1.5, .7, -.8, 1.2, -1.3]).reshape(1, 6, 1, 1).expand(-1, -1, 2, 3)
    expected = clean + history[:, -2:].repeat(1, 3, 1, 1)
    core_steps = []
    original_step = diffusion.p_sample

    def core_step(model, state, t, **kwargs):
        assert kwargs["clip_denoised"] is False
        core_steps.append(t.item())
        return original_step(model, state, t, **kwargs)

    class Oracle(nn.Module):
        map_channels, future_steps = 2, 3

        def __init__(self):
            super().__init__()
            self.cached = object()
            self.cache_calls = 0
            self.times = []

        def prepare_condition(self, condition):
            assert condition is history
            self.cache_calls += 1
            return self.cached

        def forward(self, noisy, t, condition, features):
            assert condition is history and features is self.cached
            assert not self.training and t.dtype == torch.int64
            self.times.append(t.item())
            alpha = diffusion.alpha_bars[t].to(noisy).reshape(1, 1, 1, 1)
            return (noisy - alpha.sqrt() * clean) / (1 - alpha).sqrt()

    monkeypatch.setattr(diffusion, "p_sample", core_step)
    model = Oracle().train()
    result = scrd_prediction_sample(model, history)
    torch.testing.assert_close(result, expected, atol=5e-6, rtol=5e-6)
    assert core_steps == model.times == list(range(999, -1, -1))
    assert model.cache_calls == 1 and model.training
    assert result.max() > 1


def test_prediction_scrd_train_functional_resume_and_both_weight_routes(small_scrd_run, tmp_path, monkeypatch):
    config, checkpoint = small_scrd_run
    first = load_checkpoint(checkpoint)
    assert first["ddpm_schedule"] == (1000, .0001, .02)
    assert first["training"]["backbone"] == "scrd"
    assert first["training"]["noise_conditioning"] == "scrd_timestep"
    assert first["training"]["prediction_target"] == "residual"
    assert first["training"]["reverse_variance"] == "beta"
    assert first["model_config"]["resolution"] == 8
    checkpoint = train_prediction(PredictionTrainConfig(steps=2, resume=str(checkpoint), **config))
    saved = load_checkpoint(checkpoint)
    assert saved["step"] == 2 and saved["optimizer"]["state"]
    assert any(not torch.equal(first["model"][key], value) for key, value in saved["model"].items())
    assert all(torch.isfinite(value).all() for value in saved["model"].values())

    history_csv = write_maps(tmp_path / "history", count=3)
    for weights in ("model", "ema"):
        seen = []

        def route(model, history, **kwargs):
            assert isinstance(model, SCRDUNet)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, saved[weights][key], rtol=0, atol=0)
            last = history[:, -2:].repeat(1, 2, 1, 1)
            epsilon = model(torch.zeros_like(last), torch.zeros(len(last), dtype=torch.long), history)
            assert epsilon.shape == last.shape and torch.isfinite(epsilon).all()
            seen.append(True)
            return last

        monkeypatch.setattr(inference, "scrd_prediction_sample", route)
        output = tmp_path / weights
        records = inference.infer_prediction(PredictionInferenceConfig(
            checkpoint=str(checkpoint), csv_path=str(history_csv), output=str(output),
            target_csv=config["target_csv"], samples=2, weights=weights, device="cpu", progress=False))
        assert len(seen) == 2 and len(records) == 4
        for _, _, path, units in records:
            result = np.load(output / path)
            assert result.shape == (2, 8, 8) and np.isfinite(result).all()
            assert units == "synthetic units"


def test_prediction_resume_rejects_backbone_gn_and_target_selection_changes(small_scrd_run, tmp_path):
    config, checkpoint = small_scrd_run
    new_targets = tmp_path / "other_targets.csv"
    pd.read_csv(config["target_csv"]).to_csv(new_targets, index=False)
    for changed, message in [({"backbone": "ddpm"}, "same prediction backbone"),
                             ({"scrd_output_norm_groups": 8}, "same model"),
                             ({"target_csv": str(new_targets)}, "same target CSV")]:
        options_now = dict(config, steps=2, resume=str(checkpoint))
        options_now.update(changed)
        with pytest.raises(ValueError, match=message):
            train_prediction(PredictionTrainConfig(**options_now))


def test_prediction_scrd_checkpoint_cannot_silently_change_noise_or_target(small_scrd_run, tmp_path):
    _, checkpoint = small_scrd_run
    saved = load_checkpoint(checkpoint)
    for field, value in [("noise_conditioning", "gamma_adagn"),
                         ("prediction_target", "future"), ("reverse_variance", "posterior")]:
        altered = deepcopy(saved)
        altered["training"][field] = value
        path = tmp_path / f"wrong_{field}.pt"
        torch.save(altered, path)
        with pytest.raises(ValueError, match="integer time, residual targets and core beta variance"):
            inference.infer_prediction(PredictionInferenceConfig(checkpoint=str(path), device="cpu"))


def test_default_gamma_prediction_keeps_legacy_checkpoint_schema(tmp_path, monkeypatch):
    csv_path = write_maps(tmp_path / "train")
    config = options(csv_path, tmp_path / "gamma")
    config.update(backbone="ddpm", base_channels=32, channel_mult=(1, 2))
    checkpoint = train_prediction(PredictionTrainConfig(steps=1, **config))
    saved = load_checkpoint(checkpoint)
    assert saved["training"]["noise_conditioning"] == "gamma_adagn"
    assert "backbone" not in saved["training"] and "target_csv" not in saved["training"]
    checkpoint = train_prediction(PredictionTrainConfig(steps=2, resume=str(checkpoint), **config))
    assert load_checkpoint(checkpoint)["step"] == 2
    calls = []

    def route(model, history, **kwargs):
        assert isinstance(model, PredictionUNet)
        calls.append(True)
        return history[:, -2:].repeat(1, 2, 1, 1)

    monkeypatch.setattr(inference, "prediction_sample", route)
    records = inference.infer_prediction(PredictionInferenceConfig(
        checkpoint=str(checkpoint), csv_path=str(write_maps(tmp_path / "history", 3)),
        output=str(tmp_path / "inferred"), weights="model", device="cpu", progress=False))
    assert calls == [True] and len(records) == 2


def test_target_selection_checks_every_future_and_optional_resolution(tmp_path):
    csv_path = write_maps(tmp_path / "maps")
    frame_rows = pd.read_csv(csv_path)
    targets = tmp_path / "selected.csv"
    frame_rows.iloc[[3, 4]][["timestamp"]].to_csv(targets, index=False)
    config = dict(history_steps=3, future_steps=2, map_channels=2, resolution=8,
                  value_min=0, value_max=1, target_csv=targets)
    source = ForecastDataset(csv_path, **config)
    assert len(source) == 1
    assert source[0][2] == "2026-01-01T01:00:00+00:00"
    assert source[0][3] == ("2026-01-01T01:30:00+00:00", "2026-01-01T02:00:00+00:00")
    frame_rows.iloc[[3, 5]][["timestamp"]].to_csv(targets, index=False)
    with pytest.raises(ValueError, match="target_outside_split"):
        ForecastDataset(csv_path, **config)
    model = SCRDUNet(history_steps=3, future_steps=2, map_channels=2, resolution=8,
                     widths=(8, 16), window=4, tcu_depth=1, time_dim=16,
                     norm_groups=4, output_norm_groups=4)
    with pytest.raises(ValueError, match="resolution"):
        model.prepare_condition(torch.randn(1, 6, 9, 9))


def test_scrd_sample_restores_model_mode_after_an_error():
    class Broken(nn.Module):
        map_channels, future_steps = 1, 2

        def prepare_condition(self, condition):
            raise RuntimeError("condition failed")

    model = Broken().train()
    with pytest.raises(RuntimeError, match="condition failed"):
        scrd_prediction_sample(model, torch.zeros(1, 3, 4, 4))
    assert model.training
