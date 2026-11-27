"""DDPM equations (2), (4), (7), (11), (14), (15).

A draft using the original fixedlarge CIFAR-10 variance.
Training and reverse sampling loops have not yet been integrated.
"""

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


def q_sample(x_0, t, eps):
    """Equation (4), with code t=0 as the first diffusion step."""
    if x_0.shape != eps.shape:
        raise ValueError("x_0 and eps must have identical shapes")
    if t.ndim != 1 or t.shape[0] != x_0.shape[0] or t.dtype != torch.long:
        raise ValueError("t must be a long tensor with shape (batch_size,)")
    if torch.any(t < 0) or torch.any(t >= num_diff_steps):
        raise ValueError("t is outside the diffusion schedule")
    alpha_bar = alpha_bars.to(device=x_0.device, dtype=x_0.dtype)[t].reshape(
        -1, *([1] * (x_0.ndim - 1))
    )
    return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps


def posterior_mean(x_0, x_t, t):
    """Equation (7), without precomputed posterior mean coefficients."""
    if x_0.shape != x_t.shape:
        raise ValueError("x_0 and x_t must have identical shapes")
    shape = (-1, *([1] * (x_0.ndim - 1)))
    a = alphas.to(device=x_t.device, dtype=x_t.dtype)[t].reshape(shape)
    b = betas.to(device=x_t.device, dtype=x_t.dtype)[t].reshape(shape)
    ab = alpha_bars.to(device=x_t.device, dtype=x_t.dtype)[t].reshape(shape)
    ab_prev = alpha_bars_prev.to(device=x_t.device, dtype=x_t.dtype)[t].reshape(shape)
    return (torch.sqrt(ab_prev) * b * x_0 + torch.sqrt(a) * (1 - ab_prev) * x_t) / (1 - ab)


def predict_x0(x_t, t, eps_pred):
    """Equation (15), without clipping."""
    if x_t.shape != eps_pred.shape:
        raise ValueError("x_t and eps_pred must have identical shapes")
    shape = (-1, *([1] * (x_t.ndim - 1)))
    ab = alpha_bars.to(device=x_t.device, dtype=x_t.dtype)[t].reshape(shape)
    return (x_t - torch.sqrt(1 - ab) * eps_pred) / torch.sqrt(ab)


def reverse_mean(x_t, t, eps_pred):
    """Equation (11), without clipping."""
    if x_t.shape != eps_pred.shape:
        raise ValueError("x_t and eps_pred must have identical shapes")
    shape = (-1, *([1] * (x_t.ndim - 1)))
    a = alphas.to(device=x_t.device, dtype=x_t.dtype)[t].reshape(shape)
    b = betas.to(device=x_t.device, dtype=x_t.dtype)[t].reshape(shape)
    ab = alpha_bars.to(device=x_t.device, dtype=x_t.dtype)[t].reshape(shape)
    return (x_t - b / torch.sqrt(1 - ab) * eps_pred) / torch.sqrt(a)


def ddpm_loss(model, x_0):
    """Equation (14) and Algorithm 1, unweighted epsilon MSE."""
    t = torch.randint(0, num_diff_steps, (x_0.shape[0],), device=x_0.device)
    eps = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, eps)
    eps_pred = model(x_t, t)
    if eps_pred.shape != eps.shape:
        raise ValueError("eps_pred and eps must have identical shapes")
    return torch.mean((eps_pred - eps).square())
