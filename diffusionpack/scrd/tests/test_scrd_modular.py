import math

import numpy as np
import pandas as pd
import pytest
import torch

from diffusionpack.main.check_prediction import check_prediction
from diffusionpack.main.train_prediction import train_prediction
from diffusionpack.prediction_config import PredictionTrainConfig
from diffusionpack.scrd import SCRDUNet, STCG, TCU, SWCIBlock, SWCrossAttention


@pytest.mark.parametrize('height,width,shift', [(8, 8, 0), (8, 8, 2), (7, 11, 2), (3, 9, 2)])
def test_swci_against_independent_spatial_windows(height, width, shift):
    torch.manual_seed(5)
    model = SWCrossAttention(3, window=4, shift=shift).double()
    noise = torch.randn(2, 3, height, width, dtype=torch.float64)
    cues = torch.randn_like(noise)
    effective_shift = shift if min(height, width) > 4 else 0
    rows, cols = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    groups = torch.stack([(rows - effective_shift) // 4, (cols - effective_shift) // 4], dim=-1).reshape(-1, 2)
    allowed = (groups[:, None] == groups[None, :]).all(dim=-1)
    q = model.query(noise).flatten(2).transpose(1, 2)
    k = model.key(cues).flatten(2).transpose(1, 2)
    v = model.value(noise).flatten(2).transpose(1, 2)
    logits = (q @ k.transpose(1, 2)) / math.sqrt(3)
    expected = logits.masked_fill(~allowed, float('-inf')).softmax(-1) @ v
    expected = model.proj_out(expected.transpose(1, 2).reshape_as(noise))
    torch.testing.assert_close(model(noise, cues), expected, atol=1e-12, rtol=1e-12)
    wrong_v = model.value(cues).flatten(2).transpose(1, 2)
    wrong = logits.masked_fill(~allowed, float('-inf')).softmax(-1) @ wrong_v
    wrong = model.proj_out(wrong.transpose(1, 2).reshape_as(noise))
    assert not torch.allclose(expected, wrong)
    wrong = model.proj_out((logits.softmax(-1) @ v).transpose(1, 2).reshape_as(noise))
    assert not torch.allclose(expected, wrong)


def test_tcu_broadcast_gate_with_hand_computable_values():
    model = TCU(2).double()
    with torch.no_grad():
        for conv in (model.dw_conv, model.dwd_conv):
            conv.weight.zero_()
            conv.weight[:, 0, 1, 1] = 1
            conv.bias.zero_()
        model.conv1x1.weight.zero_()
        model.conv1x1.bias.copy_(torch.tensor([2., 3.]))
        model.fc.weight.copy_(torch.eye(2))
        model.fc.bias.zero_()
    x = torch.tensor([[[[1., 3.]], [[2., 6.]]]], dtype=torch.float64)
    expected = torch.tensor([[[[4., 12.]], [[24., 72.]]]], dtype=torch.float64)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert not torch.equal(model(x), x * torch.tensor([2., 3.])[None, :, None, None])


def test_module_imports_and_cached_features():
    torch.manual_seed(17)
    model = SCRDUNet(history_steps=5, future_steps=4, resolution=16)
    assert isinstance(model.stcg, STCG)
    assert isinstance(model.unet.down_blocks[0], SWCIBlock)
    history, noisy, t = torch.randn(2, 5, 16, 16), torch.randn(2, 4, 16, 16), torch.tensor([0, 999])
    features = model.prepare_condition(history)
    assert [tuple(f.shape) for f in features] == [(2, 32, 16, 16), (2, 64, 8, 8), (2, 128, 4, 4), (2, 256, 2, 2)]
    actual = model(noisy, t, history)
    torch.testing.assert_close(actual, model(noisy, t, history, features), rtol=0, atol=0)
    changed = model(noisy, t, history, model.prepare_condition(history + 1))
    assert not torch.allclose(actual, changed)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def write_maps(folder, start):
    folder.mkdir()
    rows = []
    for i, moment in enumerate(pd.date_range(start, periods=8, freq='30min')):
        name = f'{i}.npy'
        np.save(folder / name, np.full((8, 8), i / 10, dtype=np.float32))
        rows.append(dict(timestamp=moment.isoformat(), path=name, available_at=moment.isoformat()))
    path = folder / 'maps.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def settings(csv_path, output):
    return PredictionTrainConfig(csv_path=csv_path, output=str(output), backbone='scrd',
                                 resolution=8, history_steps=2, future_steps=2,
                                 value_min=0, value_max=1, units='synthetic',
                                 scrd_widths=(8, 16), scrd_window=4, scrd_tcu_depth=1,
                                 scrd_time_dim=16, scrd_norm_groups=4, scrd_output_norm_groups=4,
                                 device='cpu', amp=False, epochs=None, steps=1)


def test_preflight_uses_training_batch_and_does_not_write_output(tmp_path):
    train = write_maps(tmp_path / 'train', '2026-01-01T00:00:00Z')
    valid = write_maps(tmp_path / 'valid', '2026-02-01T00:00:00Z')
    test = write_maps(tmp_path / 'test', '2026-03-01T00:00:00Z')
    output = tmp_path / 'run'
    config = settings(train, output)
    assert math.isfinite(check_prediction(config, valid_csv=valid, test_csv=test, check_all=True))
    assert not output.exists()
    with pytest.raises(ValueError, match='targets overlap'):
        check_prediction(config, valid_csv=train)
    with pytest.raises(ValueError, match='chronologically separated'):
        check_prediction(settings(valid, output), valid_csv=train)


def test_training_does_not_overwrite_existing_checkpoint(tmp_path):
    output = tmp_path / 'run'
    output.mkdir()
    checkpoint = output / 'latest.pt'
    checkpoint.write_bytes(b'prior checkpoint')
    config = settings('not-loaded.csv', output)
    with pytest.raises(FileExistsError, match='empty output folder'):
        train_prediction(config)
    assert checkpoint.read_bytes() == b'prior checkpoint'
