from functools import partial

import torch

from .diffusion import ddpm_loss
from .main.sampling import p_sample_loop


def restore_loss(model, target, condition):
    # L = mean((εθ(y_t,t,c) − ε)²)
    denoiser = partial(model, condition=condition)
    return ddpm_loss(denoiser, target)


@torch.no_grad()
def restore_sample(model, condition, *, clip_denoised=False, progress=False):
    height, width = condition.shape[-2:]
    shape = (condition.shape[0], model.target_channels, height, width)
    denoiser = partial(model, condition=condition)
    x_t = torch.randn(shape, device=condition.device, dtype=condition.dtype)
    return p_sample_loop(denoiser, x_t, clip_denoised=clip_denoised, progress=progress)
