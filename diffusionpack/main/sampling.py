import torch
from tqdm.auto import tqdm

from .. import diffusion


@torch.no_grad()
def p_sample_loop(model, x_t, *, clip_denoised, progress=False):
    steps = range(diffusion.num_diff_steps - 1, -1, -1)
    for step in tqdm(steps, desc="DDPM sampling", dynamic_ncols=True, disable=not progress):
        t = torch.full((len(x_t),), step, device=x_t.device, dtype=torch.long)
        x_t = diffusion.p_sample(model, x_t, t, clip_denoised=clip_denoised)
    if not torch.isfinite(x_t).all():
        raise FloatingPointError("DDPM sampling produced a non-finite value")
    return x_t
