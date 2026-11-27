import torch
from .config import num_diff_steps, beta_start, beta_end

betas = torch.linspace(beta_start, beta_end, num_diff_steps, dtype=torch.float64)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])

sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)

posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)


def step_value(schedule, t, x):
    values = schedule.to(device=x.device, dtype=x.dtype)[t.to(x.device)]
    shape = (len(t),) + (1,) * (x.ndim - 1)
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
    return (
        step_value(alpha_bars_prev.sqrt() * betas / (1 - alpha_bars), t, x_t) * x_0
        + step_value(alphas.sqrt() * (1 - alpha_bars_prev) / (1 - alpha_bars), t, x_t) * x_t
    )


def predict_x0(x_t, t, eps_pred):
    check_shape(x_t, eps_pred)
    return (
        x_t - step_value(sqrt_one_minus_alpha_bars, t, x_t) * eps_pred
    ) / step_value(sqrt_alpha_bars, t, x_t)


def reverse_mean(x_t, t, eps_pred):
    check_shape(x_t, eps_pred)
    eps_scale = step_value(betas / sqrt_one_minus_alpha_bars, t, x_t)
    return (x_t - eps_scale * eps_pred) / step_value(alphas.sqrt(), t, x_t)


@torch.no_grad()
def p_sample(model, x_t, t, noise=None, *, clip_denoised=False):
    t = t.to(device=x_t.device)
    eps_pred = model(x_t, t)
    x_0 = predict_x0(x_t, t, eps_pred)
    if clip_denoised:
        x_0 = x_0.clamp(-1, 1)
    mean = posterior_mean(x_0, x_t, t)
    std = step_value(betas, t, x_t).sqrt()
    if noise is None:
        noise = torch.randn_like(x_t)
    elif noise.shape != x_t.shape or noise.device != x_t.device or noise.dtype != x_t.dtype:
        raise ValueError("noise must match the shape, device and dtype of x_t")
    add_noise = (t > 0).reshape((t.shape[0],) + (1,) * (x_t.ndim - 1))
    return mean + add_noise * std * noise


def ddpm_loss(model, x_0):
    t = torch.randint(0, num_diff_steps, (x_0.shape[0],), device=x_0.device)
    eps = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, eps)
    eps_pred = model(x_t, t)
    check_shape(eps_pred, eps)
    return torch.mean((eps_pred - eps).square())
