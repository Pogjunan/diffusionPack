from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.nn.functional import silu

from diffusionpack import diffusion
from diffusionpack.backbone.ddpm_unet import DDPMUNet, ResBlock
from diffusionpack.backbone.prediction_unet import PredictionUNet
from diffusionpack.backbone.restore_unet import RestoreUNet
from diffusionpack.config import InferenceConfig, TrainConfig
from diffusionpack.main import train
from diffusionpack.main.inference import run_inference
from diffusionpack.main.inference_prediction import infer_prediction
from diffusionpack.main.inference_restore import infer_restore
from diffusionpack.main.prediction_data import ForecastDataset, inverse_scale
from diffusionpack.main.train_prediction import train_prediction
from diffusionpack.main.train_restore import train_restore
from diffusionpack.prediction_config import PredictionInferenceConfig, PredictionTrainConfig
from diffusionpack.restore import restore_loss, restore_sample
from diffusionpack.restore_config import RestoreInferenceConfig, RestoreTrainConfig


def test_gamma_sampling(monkeypatch):
    target = torch.ones(3, 1, 2, 2)
    condition = torch.randn(3, 5, 2, 2)
    t = torch.tensor([0, 300, 999])
    low = diffusion.alpha_bars[t].float()
    high = diffusion.alpha_bars_prev[t].float()
    expected = low + 0.4 * (high - low)
    monkeypatch.setattr(torch, 'randint', lambda *args, **kwargs: t)
    monkeypatch.setattr(torch, 'rand_like', lambda x: torch.full_like(x, 0.4))
    monkeypatch.setattr(torch, 'randn_like', lambda x: torch.full_like(x, 0.25))

    def model(noisy, gamma, c):
        assert c is condition
        assert gamma.shape == (3,)
        torch.testing.assert_close(gamma, expected)
        g = expected[:, None, None, None]
        torch.testing.assert_close(noisy, g.sqrt() * target + (1 - g).sqrt() * 0.25)
        return torch.full_like(noisy, 0.25)

    assert restore_loss(model, target, condition) == 0


def test_fixed_gamma_during_reverse():
    condition = torch.randn(2, 5, 2, 2)
    seen = []

    class Denoiser:
        target_channels = 4

        def __call__(self, noisy, gamma, c):
            assert c is condition
            assert gamma.shape == (2,)
            seen.append(gamma.clone())
            return torch.zeros_like(noisy)

    result = restore_sample(Denoiser(), condition)
    expected = diffusion.alpha_bars.flip(0).float()[:, None].expand(-1, 2)
    torch.testing.assert_close(torch.stack(seen), expected)
    assert result.shape == (2, 4, 2, 2)
    assert torch.isfinite(result).all()


def test_adagn_formula():
    block = ResBlock(32, 32, 128, 0, use_adagn=True)
    x = torch.randn(2, 32, 4, 4)
    temb = torch.randn(2, 128)
    h = block.conv1(silu(block.norm1(x)))
    scale, shift = block.emb_proj(silu(temb)).chunk(2, dim=1)
    h = block.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
    torch.testing.assert_close(block(x, temb), x + block.conv2(silu(h)))


def test_conditioning_modes():
    settings = dict(resolution=8, base_channels=32, num_res_blocks=1, channel_mult=(1, 2))
    base = DDPMUNet(**settings)
    restored = RestoreUNet(**settings)
    for model, factor in [(base, 1), (restored, 2)]:
        for block in model.modules():
            if isinstance(block, ResBlock):
                assert block.emb_proj.out_features == factor * block.conv1.out_channels
    with pytest.raises(ValueError, match='integer'):
        base(torch.randn(2, 3, 8, 8), torch.tensor([0.2, 0.8]))
    with pytest.raises(ValueError, match='floating-point'):
        restored(torch.randn(2, 3, 8, 8), torch.tensor([0, 1]), torch.randn(2, 3, 8, 8))


@pytest.mark.parametrize('history,future,channels', [(5, 1, 1), (3, 3, 2), (6, 4, 1)])
@pytest.mark.parametrize('resolution', [32, 512, 1024])
def test_joint_prediction_shape(history, future, channels, resolution):
    with torch.device('meta'):
        model = PredictionUNet(history_steps=history, future_steps=future, map_channels=channels,
                               resolution=resolution, base_channels=32, num_res_blocks=1)
        target = torch.empty(2, future * channels, resolution, resolution)
        condition = torch.empty(2, history * channels, resolution, resolution)
        result = model(target, torch.full((2,), 0.5), condition)
    assert result.shape == target.shape
    assert model.conv_in.in_channels == (history + future) * channels
    assert model.conv_out.out_channels == future * channels


def write_csv(folder, indices, *, late_index=None, channels=2):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'maps.csv'
    start = pd.Timestamp('2026-01-01T00:00:00Z')
    rows = []
    for i in indices:
        moment = start + timedelta(minutes=30 * i)
        available = moment + timedelta(minutes=45 if i == late_index else 5)
        name = f'{i}.npy'
        grid = np.stack([np.full((8, 8), 10 * i + c, dtype=np.float32) for c in range(channels)])
        np.save(folder / name, grid)
        rows.append([moment.isoformat(), name, available.isoformat()])
    pd.DataFrame(rows, columns=['timestamp', 'path', 'available_at']).to_csv(path, index=False)
    return path


def test_forecast_window_and_channel_order(tmp_path):
    csv_path = write_csv(tmp_path, range(8))
    source = ForecastDataset(csv_path, history_steps=3, future_steps=3, map_channels=2, resolution=8,
                             cadence_minutes=30, history_lag_minutes=30, lead_minutes=30,
                             value_min=0, value_max=100, origin_time='2026-01-01T01:30:00Z')
    condition, future, origin, valid = source[0]
    assert condition.shape == future.shape == (6, 8, 8)
    torch.testing.assert_close(inverse_scale(condition[:, 0, 0], 0, 100), torch.tensor([0., 1., 10., 11., 20., 21.]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(inverse_scale(future[:, 0, 0], 0, 100), torch.tensor([40., 41., 50., 51., 60., 61.]))
    assert origin == '2026-01-01T01:30:00+00:00'
    assert valid == ('2026-01-01T02:00:00+00:00', '2026-01-01T02:30:00+00:00', '2026-01-01T03:00:00+00:00')


def test_no_future_required_at_inference(tmp_path):
    csv_path = write_csv(tmp_path, range(3))
    options = dict(history_steps=3, future_steps=4, map_channels=2, resolution=8,
                   history_lag_minutes=30, value_min=0, value_max=100,
                   origin_time='2026-01-01T01:30:00Z')
    source = ForecastDataset(csv_path, require_target=False, **options)
    condition, future, _, valid = source[0]
    assert condition.shape == (6, 8, 8)
    assert future.numel() == 0 and len(valid) == 4
    with pytest.raises(ValueError, match="missing_target"):
        ForecastDataset(csv_path, require_target=True, **options)
    (tmp_path / '2.npy').unlink()
    with pytest.raises(ValueError, match="missing_history"):
        ForecastDataset(csv_path, require_target=False, **options)


def test_late_history_rejected(tmp_path):
    csv_path = write_csv(tmp_path, range(8), late_index=2)
    with pytest.raises(ValueError, match="late_history"):
        ForecastDataset(csv_path, history_steps=3, future_steps=3, map_channels=2, resolution=8,
                        history_lag_minutes=30, value_min=0, value_max=100,
                        origin_time='2026-01-01T01:30:00Z')


def test_csv_time_conversion(tmp_path):
    csv_path = write_csv(tmp_path, range(3))
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul").astype(str)
    df.to_csv(csv_path, index=False)
    source = ForecastDataset(csv_path, history_steps=3, future_steps=2, map_channels=2,
                             resolution=8, history_lag_minutes=30, value_min=0, value_max=100,
                             require_target=False, origin_time="2026-01-01T10:30:00+09:00")
    assert source[0][2] == "2026-01-01T01:30:00+00:00"


@pytest.mark.parametrize("problem,message", [
    ("duplicate", "Duplicate timestamps"),
    ("missing", "Missing timestamp"),
    ("early", "Availability predates observation"),
])
def test_invalid_csv(tmp_path, problem, message):
    csv_path = write_csv(tmp_path, range(3))
    df = pd.read_csv(csv_path)
    if problem == "duplicate":
        df.loc[1, "timestamp"] = "2026-01-01T09:00:00+09:00"
    elif problem == "missing":
        df.loc[0, "available_at"] = None
    else:
        df.loc[0, "available_at"] = "2025-12-31T23:59:00Z"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match=message):
        ForecastDataset(csv_path, history_steps=3, map_channels=2, resolution=8,
                        value_min=0, value_max=100, require_target=False)


@pytest.fixture
def short_schedule(monkeypatch):
    betas = torch.linspace(0.0001, 0.02, 4, dtype=torch.float64)
    alphas = 1 - betas
    bars = alphas.cumprod(0)
    prev = torch.cat([torch.ones(1, dtype=bars.dtype), bars[:-1]])
    for name, value in dict(num_diff_steps=4, betas=betas, alphas=alphas,
                            alpha_bars=bars, alpha_bars_prev=prev,
                            sqrt_alpha_bars=bars.sqrt(),
                            sqrt_one_minus_alpha_bars=(1 - bars).sqrt(),
                            posterior_var=betas * (1 - prev) / (1 - bars)).items():
        monkeypatch.setattr(diffusion, name, value)
    monkeypatch.setattr(train, 'ddpm_schedule', (4, betas[0].item(), betas[-1].item()))


def test_prediction_train_resume_inference(tmp_path, short_schedule):
    csv_path = write_csv(tmp_path / 'train', range(8))
    inference_csv_path = write_csv(tmp_path / 'inference', range(3))
    options = dict(csv_path=str(csv_path), resolution=8, base_channels=32,
                   channel_mult=(1, 2), history_steps=3, future_steps=3, map_channels=2,
                   history_lag_minutes=30, value_min=0, value_max=100,
                   batch_size=2, epochs=None, warmup_steps=0, device='cpu', amp=False,
                   progress=False, output=str(tmp_path / 'run'))
    checkpoint = train_prediction(PredictionTrainConfig(steps=1, **options))
    state = train.load_checkpoint(checkpoint)
    assert state['step'] == 1
    assert state['model_config']['future_steps'] == 3
    assert state['training']['noise_conditioning'] == 'gamma_adagn'
    checkpoint = train_prediction(PredictionTrainConfig(steps=2, resume=str(checkpoint), **options))
    assert train.load_checkpoint(checkpoint)['step'] == 2
    output = tmp_path / 'output'
    records = infer_prediction(PredictionInferenceConfig(
        checkpoint=str(checkpoint), csv_path=str(inference_csv_path), output=str(output),
        origin_time='2026-01-01T01:30:00Z', samples=2, device='cpu', progress=False))
    assert len(records) == 6
    for i, (origin, valid, path, units) in enumerate(records):
        assert origin == '2026-01-01T01:30:00+00:00'
        assert path.endswith(f'{30 * (i % 3 + 1)}min_sample_{i // 3 + 1:03d}.npy')
        assert valid == (pd.Timestamp(origin) + timedelta(minutes=30 * (i % 3 + 1))).isoformat()
        grid = np.load(output / path)
        assert grid.shape == (2, 8, 8) and np.isfinite(grid).all()
    with pytest.raises(ValueError, match='same model'):
        train_prediction(PredictionTrainConfig(steps=3, resume=str(checkpoint), **dict(options, future_steps=2)))


def test_restore_train_inference(tmp_path, short_schedule):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    target.mkdir()
    for i in range(2):
        Image.fromarray(np.full((8, 8, 3), 50 + 20 * i, dtype=np.uint8)).save(source / f'{i}.png')
        Image.fromarray(np.full((8, 8, 3), 100 + 20 * i, dtype=np.uint8)).save(target / f'{i}.png')
    checkpoint = train_restore(RestoreTrainConfig(
        dataset='paired', condition_dir=str(source), target_dir=str(target),
        resolution=8, base_channels=32, batch_size=2, steps=1, epochs=None,
        warmup_steps=0, device='cpu', amp=False, progress=False, output=str(tmp_path / 'run')))
    assert train.load_checkpoint(checkpoint)['training']['noise_conditioning'] == 'gamma_adagn'
    output = tmp_path / 'output'
    count = infer_restore(RestoreInferenceConfig(checkpoint=str(checkpoint), condition_path=str(source),
                                                  output=str(output), device='cpu', progress=False))
    assert count == 2
    assert Image.open(output / '0.png').size == (8, 8)


def test_unconditional_train_inference(tmp_path, short_schedule):
    checkpoint = train.run_training(TrainConfig(dataset='fake', resolution=8, base_channels=32,
                                                max_images=2, batch_size=2, steps=1, epochs=None,
                                                warmup_steps=0, device='cpu', amp=False, progress=False,
                                                output=str(tmp_path / 'run')))
    output = tmp_path / 'sample.png'
    run_inference(InferenceConfig(checkpoint=str(checkpoint), samples=1, batch_size=1,
                                  device='cpu', progress=False, output=str(output)))
    assert Image.open(output).size == (8, 8)


@pytest.mark.parametrize('task,config,run', [
    ('prediction', PredictionInferenceConfig, infer_prediction),
    ('restore', RestoreInferenceConfig, infer_restore),
])
def test_old_conditional_checkpoint_rejected(tmp_path, task, config, run):
    path = tmp_path / 'old.pt'
    torch.save({'ddpm_schedule': train.ddpm_schedule, 'training': {'task': task}}, path)
    with pytest.raises(ValueError, match='Retrain old conditional checkpoints'):
        run(config(checkpoint=str(path), device='cpu'))
