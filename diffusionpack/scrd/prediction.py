import torch

from .. import diffusion
from ..main.sampling import p_sample_loop


def scrd_prediction_loss(model, future, history):
    last = history[:, -model.map_channels:].repeat(1, model.future_steps, 1, 1)
    diffusion.check_shape(future, last)
    residual = future - last

    def denoiser(noisy, t):
        return model(noisy, t, history)

    return diffusion.ddpm_loss(denoiser, residual)


@torch.no_grad()
def scrd_prediction_sample(model, history, *, progress=False):
    was_training = model.training
    model.eval()
    try:
        condition_features = model.prepare_condition(history)
        last = history[:, -model.map_channels:].repeat(1, model.future_steps, 1, 1)
        state = torch.randn_like(last)

        def denoiser(noisy, t):
            return model(noisy, t, history, condition_features)

        residual = p_sample_loop(denoiser, state, clip_denoised=False, progress=progress)
        future = residual + last
        if not torch.isfinite(future).all():
            raise FloatingPointError("Non-finite SCRD forecast")
        return future
    finally:
        model.train(was_training)
