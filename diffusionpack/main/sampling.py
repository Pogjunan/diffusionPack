import torch
from tqdm.auto import tqdm

from .. import diffusion
from ..ddim import ddim_sample


@torch.no_grad()
def p_sample_loop(model, x_t, *, clip_denoised, progress=False,
                  sampler="ddpm", sampling_steps=None, eta=0.0):
    if sampling_steps is not None and (type(sampling_steps) is not int or not 1 <= sampling_steps <= diffusion.num_diff_steps):
        raise ValueError("sampling_steps must be an integer between 1 and num_diff_steps")
    if sampler == "ddpm":
        if sampling_steps not in (None, diffusion.num_diff_steps) or eta != 0:
            raise ValueError("DDPM uses all training steps; eta is for DDIM")
        steps = range(diffusion.num_diff_steps - 1, -1, -1)
        for step in tqdm(steps, desc="DDPM sampling", dynamic_ncols=True, disable=not progress):
            t = torch.full((len(x_t),), step, device=x_t.device, dtype=torch.long)
            x_t = diffusion.p_sample(model, x_t, t, clip_denoised=clip_denoised)
    elif sampler == "ddim":
        if sampling_steps is None:
            sampling_steps = diffusion.num_diff_steps
        steps = torch.linspace(diffusion.num_diff_steps - 1, 0, sampling_steps,
                               dtype=torch.float64, device="cpu").round().long().tolist()
        for t, s in tqdm(zip(steps, steps[1:] + [-1]), total=len(steps),
                         desc="DDIM sampling", dynamic_ncols=True, disable=not progress):
            x_t = ddim_sample(model, x_t, t, s, eta=eta, clip_denoised=clip_denoised)
    else:
        raise ValueError("Choose ddpm or ddim")
    if not torch.isfinite(x_t).all():
        raise FloatingPointError("Sampling produced a non-finite value")
    return x_t
