import torch

from . import diffusion


@torch.no_grad()
def ddim_sample(model, x_t, t, s, *, eta=0.0, clip_denoised=False):
    if type(t) is not int or type(s) is not int or not -1 <= s < t < len(diffusion.alpha_bars):
        raise ValueError("Expected integer timesteps with -1 <= s < t < num_diff_steps")
    if not 0 <= eta <= 1:
        raise ValueError("eta must be between 0 and 1")

    t_batch = torch.full((len(x_t),), t, device=x_t.device, dtype=torch.long)
    eps_pred = model(x_t, t_batch)
    x0_pred = diffusion.predict_x0(x_t, t_batch, eps_pred)
    if clip_denoised:
        x0_pred = x0_pred.clamp(-1, 1)
    if s == -1:
        return x0_pred

    alpha_bar_t = diffusion.alpha_bars[t]
    alpha_bar_s = diffusion.alpha_bars[s]
    if clip_denoised:
        eps_pred = (x_t - alpha_bar_t.sqrt() * x0_pred) / (1 - alpha_bar_t).sqrt()

    # DDIM Eq. (16)
    sigma = eta * ((1 - alpha_bar_s) / (1 - alpha_bar_t)).sqrt() * (1 - alpha_bar_t / alpha_bar_s).sqrt()

    # DDIM Eq. (12)
    x_s = alpha_bar_s.sqrt() * x0_pred + (1 - alpha_bar_s - sigma.square()).sqrt() * eps_pred
    if eta > 0:
        x_s = x_s + sigma * torch.randn_like(x_t)
    return x_s
