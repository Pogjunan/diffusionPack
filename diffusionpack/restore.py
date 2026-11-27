from functools import partial

from .diffusion import ddpm_loss, p_sample_loop


def restore_loss(model, target, condition):
    denoiser = partial(model, condition=condition)
    return ddpm_loss(denoiser, target)


def restore_sample(model, condition, *, clip_denoised=True, progress=False):
    height, width = condition.shape[-2:]
    shape = (condition.shape[0], model.target_channels, height, width)
    denoiser = partial(model, condition=condition)
    return p_sample_loop(
        denoiser, shape, device=condition.device, dtype=condition.dtype,
        clip_denoised=clip_denoised, progress=progress,
    )
