import pytest
import torch

from diffusionpack.backbone.ddpm_unet import DDPMUNet, default_channel_mult
from diffusionpack.backbone.prediction_unet import PredictionUNet
from diffusionpack.backbone.restore_unet import RestoreUNet


@pytest.mark.parametrize("resolution,levels", [(32, 4), (64, 4), (128, 4),
                                                 (256, 5), (512, 6), (1024, 7)])
def test_resolution_levels(resolution, levels):
    assert len(default_channel_mult(resolution)) == levels
    assert resolution // 2 ** (levels - 1) <= 16 or resolution == 32


@pytest.mark.parametrize("resolution", [32, 128, 512, 1024])
@pytest.mark.parametrize("kind", ["generation", "restoration", "prediction"])
def test_model_shapes(resolution, kind):
    # Meta tensors validate high-resolution shapes without allocating activations.
    with torch.device("meta"):
        if kind == "generation":
            model = DDPMUNet(resolution=resolution, base_channels=32, num_res_blocks=1)
            result = model(torch.empty(2, 3, resolution, resolution, device="meta"),
                           torch.zeros(2, dtype=torch.long, device="meta"))
            channels = 3
        elif kind == "restoration":
            model = RestoreUNet(resolution=resolution, base_channels=32, num_res_blocks=1)
            result = model(torch.empty(2, 3, resolution, resolution, device="meta"),
                           torch.full((2,), 0.5, device="meta"),
                           torch.empty(2, 3, resolution, resolution, device="meta"))
            channels = 3
        else:
            model = PredictionUNet(resolution=resolution, base_channels=32, num_res_blocks=1)
            result = model(torch.empty(2, 1, resolution, resolution, device="meta"),
                           torch.full((2,), 0.5, device="meta"),
                           torch.empty(2, 5, resolution, resolution, device="meta"))
            channels = 1
    assert result.shape == (2, channels, resolution, resolution)


def test_legacy_default_structure():
    defaults = DDPMUNet(resolution=32, base_channels=32, num_res_blocks=1)
    explicit = DDPMUNet(resolution=32, base_channels=32, num_res_blocks=1,
                       channel_mult=(1, 2, 2, 2))
    assert {key: tuple(value.shape) for key, value in defaults.state_dict().items()} == {
        key: tuple(value.shape) for key, value in explicit.state_dict().items()
    }
    explicit.load_state_dict(defaults.state_dict(), strict=True)


def test_shallow_large_resolution_fails():
    with pytest.raises(ValueError, match="attention exceeds"):
        DDPMUNet(resolution=1024, base_channels=32, num_res_blocks=1,
                 channel_mult=(1, 2, 2, 2))


def test_unconditional_backward():
    model = DDPMUNet(resolution=32, base_channels=32, num_res_blocks=1)
    x = torch.randn(2, 3, 32, 32)
    y = model(x, torch.tensor([0, 999]))
    y.square().mean().backward()
    assert y.shape == x.shape
    assert model.conv_in.weight.grad is not None


def test_restore_backward():
    model = RestoreUNet(resolution=32, base_channels=32, num_res_blocks=1)
    x = torch.randn(2, 3, 32, 32)
    y = model(x, torch.tensor([0.2, 0.8]), x)
    y.square().mean().backward()
    assert y.shape == x.shape
    assert model.conv_in.weight.grad is not None


def test_prediction_backward():
    model = PredictionUNet(resolution=32, base_channels=32, num_res_blocks=1)
    x = torch.randn(2, 1, 32, 32)
    y = model(x, torch.tensor([0.2, 0.8]), torch.randn(2, 5, 32, 32))
    y.square().mean().backward()
    assert y.shape == x.shape
    assert model.conv_in.weight.grad is not None
