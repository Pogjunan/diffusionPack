import torch
from .config import num_diff_steps, beta_start, beta_end

betas = torch.linspace(beta_start, beta_end, num_diff_steps, dtype=torch.float64)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])

sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
log_one_minus_alpha_bars = torch.log(1 - alpha_bars)
sqrt_recip_alpha_bars = torch.sqrt(1 / alpha_bars)
sqrt_recipm1_alpha_bars = torch.sqrt(1 / alpha_bars - 1)

posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
posterior_log_var = torch.log(torch.cat([posterior_var[1:2], posterior_var[1:]]))
reverse_var = betas
reverse_log_var = torch.log(torch.cat([posterior_var[1:2], betas[1:]]))


def step_value(schedule, t, x):
    batch_size = x.shape[0]
    if t.shape != (batch_size,):
        raise ValueError("Each sample needs one timestep")
    if not x.is_floating_point():
        raise ValueError("Diffusion inputs must be floating point")
    schedule = schedule.to(device=x.device, dtype=x.dtype)
    values = schedule.index_select(0, t.to(x.device))
    shape = (batch_size,) + (1,) * (x.ndim - 1)
    return values.reshape(shape)


def check_shape(first, second):
    if first.shape != second.shape:
        raise ValueError(f"Expected matching shapes, got {first.shape} and {second.shape}")


def q_sample(x_0, t, eps):
    check_shape(x_0, eps)
    return (
        step_value(sqrt_alpha_bars, t, x_0) * x_0
        + step_value(sqrt_one_minus_alpha_bars, t, x_0) * eps
    )


def posterior_mean(x_0, x_t, t):
    check_shape(x_0, x_t)
    denom = 1 - alpha_bars
    return (
        step_value(alpha_bars_prev.sqrt() * betas / denom, t, x_t) * x_0
        + step_value(alphas.sqrt() * (1 - alpha_bars_prev) / denom, t, x_t) * x_t
    )


def predict_x0(x_t, t, eps_pred):
    check_shape(x_t, eps_pred)
    return (
        step_value(sqrt_recip_alpha_bars, t, x_t) * x_t
        - step_value(sqrt_recipm1_alpha_bars, t, x_t) * eps_pred
    )


def reverse_mean(x_t, t, eps_pred):
    check_shape(x_t, eps_pred)
    eps_scale = step_value(betas / sqrt_one_minus_alpha_bars, t, x_t)
    return (x_t - eps_scale * eps_pred) / step_value(alphas.sqrt(), t, x_t)


@torch.no_grad()
def p_sample(model, x_t, t, noise=None, *, clip_denoised=True):
    eps_pred = model(x_t, t)
    x_0 = predict_x0(x_t, t, eps_pred)
    if clip_denoised:
        x_0 = x_0.clamp(-1, 1)
    mean = posterior_mean(x_0, x_t, t)
    std = step_value(reverse_log_var, t, x_t).mul(0.5).exp()
    if noise is None:
        noise = torch.randn_like(x_t)
    elif noise.shape != x_t.shape or noise.device != x_t.device or noise.dtype != x_t.dtype:
        raise ValueError("noise must match the shape, device and dtype of x_t")
    add_noise = (t > 0).reshape((t.shape[0],) + (1,) * (x_t.ndim - 1))
    return mean + add_noise * std * noise


@torch.no_grad()
def p_sample_loop(model, shape, *, device, dtype=torch.float32, clip_denoised=True):
    x_t = torch.randn(shape, device=device, dtype=dtype)
    for step in reversed(range(num_diff_steps)):
        t = torch.full((x_t.shape[0],), step, device=x_t.device, dtype=torch.long)
        x_t = p_sample(model, x_t, t, clip_denoised=clip_denoised)
    return x_t


def ddpm_loss(model, x_0):
    t = torch.randint(0, num_diff_steps, (x_0.shape[0],), device=x_0.device)
    eps = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, eps)
    eps_pred = model(x_t, t)
    check_shape(eps_pred, eps)
    return torch.mean((eps_pred - eps).square())
