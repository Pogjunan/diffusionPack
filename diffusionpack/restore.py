from functools import partial

import torch
from tqdm.auto import tqdm

from .config import num_diff_steps
from .diffusion import ddpm_loss, p_sample


def restore_loss(model, target, condition):
    denoiser = partial(model, condition=condition)
    return ddpm_loss(denoiser, target)


@torch.no_grad()
def restore_sample(model, condition, *, clip_denoised=False, progress=False):
    height, width = condition.shape[-2:]
    shape = (condition.shape[0], model.target_channels, height, width)
    denoiser = partial(model, condition=condition)
    x_t = torch.randn(shape, device=condition.device, dtype=condition.dtype)
    for step in tqdm(reversed(range(num_diff_steps)), total=num_diff_steps,
                     desc="DDPM sampling", dynamic_ncols=True, disable=not progress):
        t = torch.full((shape[0],), step, device=x_t.device, dtype=torch.long)
        x_t = p_sample(denoiser, x_t, t, clip_denoised=clip_denoised)
    return x_t
