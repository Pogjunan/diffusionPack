import torch

from . import diffusion
from .main.sampling import p_sample_loop


def restore_loss(model, target, condition):
    t = torch.randint(0, diffusion.num_diff_steps, (target.shape[0],), device=target.device)
    low = diffusion.step_value(diffusion.alpha_bars, t, target)
    high = diffusion.step_value(diffusion.alpha_bars_prev, t, target)
    gamma = torch.lerp(low, high, torch.rand_like(low))
    noise = torch.randn_like(target)
    noisy = gamma.sqrt() * target + (1 - gamma).sqrt() * noise
    pred = model(noisy, gamma.reshape(-1), condition)
    diffusion.check_shape(pred, noise)
    return (pred - noise).square().mean()


@torch.no_grad()
def restore_sample(model, condition, *, clip_denoised=False, progress=False):
    height, width = condition.shape[-2:]
    shape = (condition.shape[0], model.target_channels, height, width)
    x_t = torch.randn(shape, device=condition.device, dtype=condition.dtype)

    def denoiser(x_t, t):
        gamma = diffusion.step_value(diffusion.alpha_bars, t, x_t).reshape(-1)
        return model(x_t, gamma, condition)

    return p_sample_loop(denoiser, x_t, clip_denoised=clip_denoised, progress=progress)
