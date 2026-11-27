import pytest
import torch
from torch import nn

from diffusionpack import diffusion as baseline
from diffusionpack.scrd import SCRDUNet, TCU
from diffusionpack.scrd import ForecastDiffusion, make_diffusion
from diffusionpack.scrd.config import SCRDConfig


def test_schedule_and_forward_match_ddpm_1000():
    engine = ForecastDiffusion(nn.Identity())
    torch.testing.assert_close(engine.betas, baseline.betas, atol=0, rtol=0)
    torch.testing.assert_close(engine.alpha_bars, baseline.alpha_bars, atol=0, rtol=0)
    assert engine.alpha_bars[-1] < 0.00005
    clean = torch.linspace(-1.5, 1.5, 48, dtype=torch.float64).reshape(3, 4, 2, 2)
    noise = clean.flip(-1) + 0.75
    t = torch.tensor([0, 499, 999])
    noisy = engine.q_sample(clean, t, noise)
    torch.testing.assert_close(noisy, baseline.q_sample(clean, t, noise), atol=0, rtol=0)
    torch.testing.assert_close(engine.predict_x0(noisy, t, noise), clean, atol=5e-14, rtol=5e-14)
    wrong = (1 - engine.betas[t]).sqrt()[:, None, None, None] * clean
    wrong += engine.betas[t].sqrt()[:, None, None, None] * noise
    assert not torch.allclose(noisy, wrong)


@pytest.mark.parametrize("variance", ["beta", "posterior"])
def test_reverse_step_matches_independent_posterior(variance):
    engine = ForecastDiffusion(nn.Identity(), reverse_variance=variance)
    t = torch.tensor([0, 1, 499, 999])
    clean = torch.linspace(-2, 2, 64, dtype=torch.float64).reshape(4, 4, 2, 2)
    epsilon = clean.flip(-1) + 0.1
    noise = clean.flip(-2) + 0.3
    noisy = baseline.q_sample(clean, t, epsilon)
    bars, previous, alpha, beta = (values[t, None, None, None] for values in
                                  (baseline.alpha_bars, baseline.alpha_bars_prev,
                                   baseline.alphas, baseline.betas))
    mean = previous.sqrt() * beta / (1 - bars) * clean
    mean += alpha.sqrt() * (1 - previous) / (1 - bars) * noisy
    var = beta if variance == "beta" else beta * (1 - previous) / (1 - bars)
    expected = mean + (t > 0)[:, None, None, None] * var.sqrt() * noise
    actual = engine.p_step(noisy, t, epsilon, noise)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    if variance == "beta":
        original = baseline.p_sample(lambda state, steps: epsilon, noisy, t, noise=noise)
        torch.testing.assert_close(actual, original, atol=1e-11, rtol=1e-11)


def test_last_step_has_no_noise_or_residual_clipping():
    engine = ForecastDiffusion(nn.Identity(), future_steps=1)
    clean = torch.tensor([[[[-1.5, 1.5]]]], dtype=torch.float64)
    epsilon = torch.tensor([[[[0.25, -0.5]]]], dtype=torch.float64)
    t = torch.zeros(1, dtype=torch.long)
    noisy = engine.q_sample(clean, t, epsilon)
    actual = engine.p_step(noisy, t, epsilon, torch.full_like(noisy, 1e6))
    torch.testing.assert_close(actual, clean, atol=1e-12, rtol=1e-12)
    assert actual.min() < -1 and actual.max() > 1


@pytest.mark.parametrize("target", ["future", "residual"])
def test_multichannel_future_order_and_last_frame(target):
    engine = ForecastDiffusion(nn.Identity(), future_steps=3, map_channels=2, target=target)
    condition = torch.tensor([1., 2., 3., 4., 5., 6.]).reshape(1, 6, 1, 1)
    future = torch.tensor([10., 20., 30., 40., 50., 60.]).reshape(1, 6, 1, 1)
    expected_last = torch.tensor([5., 6., 5., 6., 5., 6.]).reshape(1, 6, 1, 1)
    torch.testing.assert_close(engine.last_frame(condition), expected_last)
    expected_target = future - expected_last if target == "residual" else future
    torch.testing.assert_close(engine.make_target(future, condition), expected_target)


@pytest.mark.parametrize("target", ["future", "residual"])
def test_full_1000_step_oracle_caches_condition_and_reconstructs(target):
    condition = torch.linspace(-0.5, 0.5, 48, dtype=torch.float64).reshape(2, 6, 2, 2)
    clean = torch.linspace(-1.5, 1.5, 64, dtype=torch.float64).reshape(2, 8, 2, 2)
    beta = torch.linspace(0.0001, 0.02, 1000, dtype=torch.float64)
    bars = (1 - beta).cumprod(0)

    class Oracle(nn.Module):
        def __init__(self):
            super().__init__()
            self.preparations = 0
            self.seen = []
            self.features = object()

        def prepare_condition(self, value):
            assert value is condition
            self.preparations += 1
            return self.features

        def forward(self, state, t, value, features):
            assert value is condition and features is self.features
            assert not self.training
            self.seen.append(int(t[0]))
            a = bars[t, None, None, None]
            return (state - a.sqrt() * clean) / (1 - a).sqrt()

    model = Oracle()
    engine = ForecastDiffusion(model, future_steps=4, map_channels=2, target=target)
    initial = torch.ones_like(clean)
    forecast = engine.sample(condition, initial_noise=initial,
                             generator=torch.Generator().manual_seed(13))
    last = condition[:, -2:].repeat(1, 4, 1, 1)
    expected = clean + last if target == "residual" else clean
    torch.testing.assert_close(forecast, expected, atol=1e-12, rtol=1e-12)
    assert model.preparations == 1
    assert model.seen == list(range(999, -1, -1))
    assert model.training
    torch.testing.assert_close(initial, torch.ones_like(initial), atol=0, rtol=0)


def test_discrete_loss_uses_fixed_noise_level_and_epsilon_target(monkeypatch):
    future = torch.linspace(-1, 1, 32).reshape(2, 4, 2, 2)
    condition = torch.linspace(1, 2, 40).reshape(2, 5, 2, 2)
    epsilon = torch.full_like(future, 0.25)
    t = torch.tensor([0, 999])
    expected = baseline.q_sample(future - condition[:, -1:].repeat(1, 4, 1, 1), t, epsilon)

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.tensor(0.25))

        def forward(self, noisy, time, value):
            assert value is condition
            torch.testing.assert_close(time, t, atol=0, rtol=0)
            assert time.dtype == torch.long
            torch.testing.assert_close(noisy, expected, atol=3e-6, rtol=3e-6)
            return epsilon + self.offset

    def no_interpolation(*args, **kwargs):
        raise AssertionError("Discrete training must not sample an interpolated gamma")

    monkeypatch.setattr(torch, "rand_like", no_interpolation)
    model = Probe()
    loss = ForecastDiffusion(model).loss(future, condition, t=t, noise=epsilon)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.0625))
    torch.testing.assert_close(model.offset.grad, torch.tensor(0.5))


def test_gamma_loss_uses_the_same_interpolated_level_for_noising_and_model(monkeypatch):
    future = torch.ones(3, 4, 2, 2)
    condition = torch.zeros(3, 5, 2, 2)
    epsilon = torch.full_like(future, 0.25)
    t = torch.tensor([0, 400, 999])
    low, high = baseline.alpha_bars[t].float(), baseline.alpha_bars_prev[t].float()
    gamma = low + 0.25 * (high - low)
    expected = gamma[:, None, None, None].sqrt() * future
    expected += (1 - gamma[:, None, None, None]).sqrt() * epsilon

    class Probe(nn.Module):
        def forward(self, noisy, time, value):
            assert value is condition
            torch.testing.assert_close(time, gamma, atol=0, rtol=0)
            torch.testing.assert_close(noisy, expected, atol=0, rtol=0)
            return epsilon

    monkeypatch.setattr(torch, "rand_like", lambda value: torch.full_like(value, 0.25))
    engine = ForecastDiffusion(Probe(), target="future", noise_conditioning="gamma")
    assert engine.loss(future, condition, t=t, noise=epsilon).item() == 0


@pytest.mark.parametrize("conditioning,time_input", [("gamma", "index"), ("timestep", "fraction")])
def test_sampling_uses_fixed_gamma_or_normalized_discrete_time(conditioning, time_input):
    seen = []

    class Probe(nn.Module):
        def forward(self, noisy, time, condition):
            seen.append(time.clone())
            return torch.zeros_like(noisy)

    engine = ForecastDiffusion(Probe(), future_steps=1, steps=4, target="future",
                               noise_conditioning=conditioning, time_input=time_input)
    output = engine.sample(torch.zeros(1, 3, 2, 2), generator=torch.Generator().manual_seed(2))
    expected = engine.alpha_bars.flip(0).float() if conditioning == "gamma" else torch.tensor([1., .75, .5, .25])
    torch.testing.assert_close(torch.cat(seen), expected, atol=0, rtol=0)
    assert torch.isfinite(output).all()


def test_wrong_epsilon_shape_is_rejected_before_broadcasting():
    class Wrong(nn.Module):
        def forward(self, noisy, time, condition):
            return noisy[:, :1]

    engine = ForecastDiffusion(Wrong())
    with pytest.raises(ValueError, match="matching shapes"):
        engine.loss(torch.ones(2, 4, 2, 2), torch.zeros(2, 5, 2, 2))
    with pytest.raises(ValueError, match="matching shapes"):
        engine.sample(torch.zeros(2, 5, 2, 2))


def test_model_mode_restored_after_sampling_failure():
    class Broken(nn.Module):
        def forward(self, *args):
            raise RuntimeError("injected failure")

    model = Broken().train()
    with pytest.raises(RuntimeError, match="injected failure"):
        ForecastDiffusion(model).sample(torch.zeros(1, 5, 2, 2))
    assert model.training


def test_nonfinite_sample_is_not_hidden():
    class Broken(nn.Module):
        def forward(self, noisy, time, condition):
            return torch.full_like(noisy, torch.nan)

    with pytest.raises(FloatingPointError, match="Non-finite forecast"):
        ForecastDiffusion(Broken(), steps=2).sample(torch.zeros(1, 5, 2, 2))


@pytest.mark.parametrize("use_tcu", [True, False])
def test_actual_scrd_backward_reaches_condition_encoder(use_tcu):
    torch.manual_seed(3)
    model = SCRDUNet(history_steps=3, future_steps=4, map_channels=2,
                     widths=(8, 16), window=4, tcu_depth=1, time_dim=8,
                     norm_groups=4, output_norm_groups=4, use_tcu=use_tcu)
    engine = ForecastDiffusion(model, future_steps=4, map_channels=2)
    future, condition = torch.randn(2, 8, 16, 16), torch.randn(2, 6, 16, 16)
    loss = engine.loss(future, condition, t=torch.tensor([200, 999]), noise=torch.randn_like(future))
    loss.backward()
    encoder_grad = model.stcg.encoder.blocks[0][0].weight.grad
    key_grad = model.unet.down_blocks[0].attention1.key.weight.grad
    for gradient in (encoder_grad, key_grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    if use_tcu:
        gradient = model.stcg.tcu_blocks[0][0].dw_conv.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    else:
        assert not any(isinstance(module, TCU) for module in model.modules())


@pytest.mark.parametrize("conditioning", ["timestep", "gamma"])
def test_existing_ddpm_backbone_comparison_paths(conditioning):
    config = SCRDConfig(backbone="ddpm", target="future", noise_conditioning=conditioning,
                        resolution=8, base_channels=32, channel_mult=(1, 2),
                        num_res_blocks=1, attention_resolutions=(), dropout=0,
                        history_steps=3, future_steps=2, map_channels=1)
    engine = make_diffusion(config)
    future, condition = torch.randn(2, 2, 8, 8), torch.randn(2, 3, 8, 8)
    loss = engine.loss(future, condition, t=torch.tensor([0, 999]))
    loss.backward()
    assert torch.isfinite(loss)
    assert engine.model.conv_in.weight.grad is not None
    assert engine.model.use_adagn == (conditioning == "gamma")


def test_scrd_rejects_continuous_gamma_as_paper_default():
    with pytest.raises(ValueError, match="discrete timesteps"):
        make_diffusion(SCRDConfig(noise_conditioning="gamma"))
