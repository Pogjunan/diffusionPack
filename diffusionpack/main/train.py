from torch.nn.utils import clip_grad_norm_

from ..diffusion import ddpm_loss


def train_step(model, x_0, optimizer, *, grad_clip=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = ddpm_loss(model, x_0)
    loss.backward()
    clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()
