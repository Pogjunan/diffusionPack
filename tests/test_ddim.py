import pytest
import torch

from diffusionpack import diffusion
from diffusionpack.backbone.ddpm_unet import DDPMUNet
from diffusionpack.backbone.prediction_unet import PredictionUNet
from diffusionpack.backbone.restore_unet import RestoreUNet
from diffusionpack.config import InferenceConfig
from diffusionpack.ddim import ddim_sample
from diffusionpack.main.sampling import p_sample_loop
from diffusionpack.prediction import prediction_sample
from diffusionpack.prediction_config import PredictionInferenceConfig
from diffusionpack.restore import restore_sample
from diffusionpack.restore_config import RestoreInferenceConfig


@pytest.mark.parametrize("t,s", [(999, 974), (300, 10), (100, 99), (0, -1)])
def test_ddim_marginal_with_exact_noise(t, s):
    x0 = torch.randn(2, 3, 4, 4, dtype=torch.float64)
    eps = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, torch.full((2,), t), eps)
    result = ddim_sample(lambda x, step: eps, x_t, t, s)
    expected = x0 if s == -1 else diffusion.q_sample(x0, torch.full((2,), s), eps)
    torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_ddim_equation(eta):
    t, s = 600, 420
    x_t = torch.randn(2, 3, 4, 4, dtype=torch.float64)
    eps = torch.randn_like(x_t)
    a, b = diffusion.alpha_bars[t], diffusion.alpha_bars[s]
    x0 = (x_t - (1 - a).sqrt() * eps) / a.sqrt()
    sigma2 = eta**2 * (1 - b) / (1 - a) * (1 - a / b)
    torch.manual_seed(4)
    expected = b.sqrt() * x0 + (1 - b - sigma2).sqrt() * eps
    if eta:
        expected = expected + sigma2.sqrt() * torch.randn_like(x_t)
    torch.manual_seed(4)
    actual = ddim_sample(lambda x, step: eps, x_t, t, s, eta=eta)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("clip", [False, True])
def test_adjacent_eta_one(clip):
    t = 400
    x_t = torch.randn(2, 3, 4, 4, dtype=torch.float64)
    eps = torch.randn_like(x_t)
    batch_t = torch.full((2,), t)
    x0 = diffusion.predict_x0(x_t, batch_t, eps)
    if clip:
        x0 = x0.clamp(-1, 1)
    mean = diffusion.posterior_mean(x0, x_t, batch_t)
    torch.manual_seed(17)
    expected = mean + diffusion.posterior_var[t].sqrt() * torch.randn_like(x_t)
    torch.manual_seed(17)
    actual = ddim_sample(lambda x, step: eps, x_t, t, t - 1, eta=1, clip_denoised=clip)
    torch.testing.assert_close(actual, expected)


def test_clipped_x0_direction():
    t, s = 800, 450
    x_t = torch.full((1, 1, 2, 2), 2.0, dtype=torch.float64)
    a, b = diffusion.alpha_bars[t], diffusion.alpha_bars[s]
    expected = b.sqrt() + (1 - b).sqrt() * (x_t - a.sqrt()) / (1 - a).sqrt()
    actual = ddim_sample(lambda x, step: torch.zeros_like(x), x_t, t, s, clip_denoised=True)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("steps", [1, 2, 40, 50, 1000])
def test_sampling_trajectory_and_rng(steps):
    seen = []
    x_t = torch.randn(2, 1, 2, 2)

    def model(x, t):
        assert t.shape == (2,) and t.dtype == torch.long
        assert t[0] == t[1]
        seen.append(t[0].item())
        return torch.zeros_like(x)

    rng = torch.random.get_rng_state()
    result = p_sample_loop(model, x_t, sampler="ddim", sampling_steps=steps, clip_denoised=False)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert len(seen) == steps and seen[0] == 999
    assert all(a > b for a, b in zip(seen, seen[1:]))
    assert steps == 1 or seen[-1] == 0
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_clean_endpoint(eta):
    x_t = torch.randn(2, 1, 2, 2)
    eps = torch.randn_like(x_t)
    expected = diffusion.predict_x0(x_t, torch.full((2,), 35), eps)
    rng = torch.random.get_rng_state()
    actual = ddim_sample(lambda x, t: eps, x_t, 35, -1, eta=eta)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_ddpm_unchanged():
    model = lambda x, t: torch.zeros_like(x)
    x_t = torch.randn(2, 1, 2, 2)
    expected = x_t.clone()
    torch.manual_seed(11)
    for step in range(diffusion.num_diff_steps - 1, -1, -1):
        expected = diffusion.p_sample(model, expected, torch.full((len(x_t),), step))
    torch.manual_seed(11)
    actual = p_sample_loop(model, x_t, clip_denoised=False)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("kind", ["generation", "restoration", "prediction"])
def test_real_unet_sampling(kind):
    settings = dict(resolution=8, base_channels=32, channel_mult=(1, 2), num_res_blocks=1)
    options = dict(sampler="ddim", sampling_steps=4, eta=0.0)
    if kind == "generation":
        model = DDPMUNet(**settings).eval()
        x = torch.randn(2, 3, 8, 8)
        result = p_sample_loop(model, x, clip_denoised=True, **options)
        channels = 3
    elif kind == "restoration":
        model = RestoreUNet(**settings).eval()
        x = torch.randn(2, 3, 8, 8)
        original = x.clone()
        result = restore_sample(model, x, clip_denoised=True, **options)
        assert torch.equal(x, original)
        channels = 3
    else:
        model = PredictionUNet(history_steps=5, future_steps=4, map_channels=1, **settings).eval()
        x = torch.randn(2, 5, 8, 8)
        original = x.clone()
        result = prediction_sample(model, x, **options)
        assert torch.equal(x, original)
        channels = 4
    assert result.shape == (2, channels, 8, 8)
    assert torch.isfinite(result).all()


def test_fixed_gamma_for_selected_steps():
    condition = torch.randn(2, 5, 4, 4)
    seen = []

    class Denoiser:
        target_channels = 4

        def __call__(self, x, gamma, c):
            assert c is condition
            seen.append(gamma.clone())
            return torch.zeros_like(x)

    prediction_sample(Denoiser(), condition, sampler="ddim", sampling_steps=4)
    indices = torch.tensor([999, 666, 333, 0])
    expected = diffusion.alpha_bars[indices].float()[:, None].expand(-1, 2)
    torch.testing.assert_close(torch.stack(seen), expected)


@pytest.mark.parametrize("config", [InferenceConfig, RestoreInferenceConfig, PredictionInferenceConfig])
def test_inference_config(config):
    assert config().sampler == "ddpm"
    options = config(sampler="ddim", sampling_steps=40, eta=0.0)
    assert (options.sampler, options.sampling_steps, options.eta) == ("ddim", 40, 0.0)


@pytest.mark.parametrize("options", [dict(sampler="ddim", sampling_steps=0),
                                    dict(sampler="ddim", sampling_steps=1001),
                                    dict(sampler="ddim", sampling_steps=2, eta=1.1),
                                    dict(sampler="ddpm", sampling_steps=50),
                                    dict(sampler="unknown")])
def test_invalid_sampling_options(options):
    with pytest.raises(ValueError):
        p_sample_loop(lambda x, t: torch.zeros_like(x), torch.randn(1, 1, 2, 2),
                      clip_denoised=False, **options)


@pytest.mark.parametrize("t,s", [(1.5, -1), (True, -1), (4.0, 2), (4, 2.0), (4, True), (1000, 0), (5, 5), (4, -2)])
def test_invalid_ddim_indices(t, s):
    with pytest.raises(ValueError, match="integer timesteps"):
        ddim_sample(lambda x, t: torch.zeros_like(x), torch.randn(1, 1, 2, 2), t, s)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_trajectory_independent_of_default_dtype(dtype):
    previous = torch.get_default_dtype()
    seen = []

    def model(x, t):
        seen.append(t.item())
        return torch.zeros_like(x)

    try:
        torch.set_default_dtype(dtype)
        p_sample_loop(model, torch.ones(1, 1, 2, 2, dtype=torch.float32),
                      sampler="ddim", sampling_steps=40, clip_denoised=True)
    finally:
        torch.set_default_dtype(previous)
    assert seen == torch.linspace(999, 0, 40, dtype=torch.float64).round().long().tolist()


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("t,s", [(999, 250), (500, 499), (25, 0)])
def test_oracle_joint_covariance(t, s, eta):
    # Check the new marginal and cross-time covariance.
    generator = torch.Generator().manual_seed(42)
    n = 32768
    x0 = torch.full((n, 1, 1, 1), 0.7, dtype=torch.float64)
    eps = torch.randn(x0.shape, dtype=x0.dtype, generator=generator)
    x_t = diffusion.q_sample(x0, torch.full((n,), t), eps)
    torch.manual_seed(73)
    x_s = ddim_sample(lambda x, step: eps, x_t, t, s, eta=eta)
    at, a_s = diffusion.alpha_bars[t], diffusion.alpha_bars[s]
    residual_t = (x_t - at.sqrt() * x0) / (1 - at).sqrt()
    residual_s = (x_s - a_s.sqrt() * x0) / (1 - a_s).sqrt()
    corr = (1 - eta**2 * (1 - at / a_s) / (1 - at)).sqrt()
    torch.testing.assert_close(residual_s.mean(), torch.zeros((), dtype=x0.dtype), atol=0.02, rtol=0)
    torch.testing.assert_close(residual_s.var(), torch.ones((), dtype=x0.dtype), atol=0.03, rtol=0)
    torch.testing.assert_close((residual_t * residual_s).mean(), corr, atol=0.03, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_sampler_state_dtype(dtype, eta):
    x = torch.randn(2, 3, 4, 4, dtype=dtype)
    result = p_sample_loop(lambda x, t: torch.zeros_like(x), x, sampler="ddim",
                           sampling_steps=1000, eta=eta, clip_denoised=True)
    assert result.dtype == dtype and torch.isfinite(result).all()
    assert result.abs().max() <= 1


def test_ddpm_rejects_noninteger_step_count():
    with pytest.raises(ValueError, match="integer"):
        p_sample_loop(lambda x, t: torch.zeros_like(x), torch.randn(1, 1, 2, 2),
                      sampler="ddpm", sampling_steps=1000.0, clip_denoised=False)


@pytest.mark.parametrize("kind", ["generation", "restoration", "prediction"])
def test_trained_weights_reused_by_both_samplers(tmp_path, kind):
    settings = dict(resolution=8, base_channels=32, channel_mult=(1, 2), num_res_blocks=1)
    if kind == "generation":
        make_model = lambda: DDPMUNet(**settings)
        condition = None
        channels = 3
        loss_function = diffusion.ddpm_loss
    elif kind == "restoration":
        from diffusionpack.restore import restore_loss
        make_model = lambda: RestoreUNet(**settings)
        condition = torch.randn(1, 3, 8, 8)
        channels = 3
        loss_function = restore_loss
    else:
        from diffusionpack.prediction import prediction_loss
        make_model = lambda: PredictionUNet(history_steps=5, future_steps=4, **settings)
        condition = torch.randn(1, 5, 8, 8)
        channels = 4
        loss_function = prediction_loss
    torch.manual_seed(13)
    model = make_model()
    target = torch.randn(1, channels, 8, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss = loss_function(model, target) if condition is None else loss_function(model, target, condition)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    model = make_model().eval()
    model.load_state_dict(torch.load(path, weights_only=True))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    x_t = torch.randn_like(target)

    def denoiser(state, t):
        if condition is None:
            return model(state, t)
        gamma = diffusion.step_value(diffusion.alpha_bars, t, state).reshape(-1)
        return model(state, gamma, condition)

    # Use the same trained weights and initial noise.
    ddpm = p_sample_loop(denoiser, x_t.clone(), clip_denoised=kind != "prediction")
    ddim = p_sample_loop(denoiser, x_t.clone(), sampler="ddim", sampling_steps=40,
                         clip_denoised=kind != "prediction")
    again = p_sample_loop(denoiser, x_t.clone(), sampler="ddim", sampling_steps=40,
                          clip_denoised=kind != "prediction")
    assert ddpm.shape == ddim.shape == target.shape
    assert torch.isfinite(ddpm).all() and torch.isfinite(ddim).all()
    assert torch.equal(ddim, again)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
