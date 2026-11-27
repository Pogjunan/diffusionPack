from copy import deepcopy

import torch
from torch.nn.utils import clip_grad_norm_

from ..config import ema_decay, learning_rate, warmup_steps
from ..diffusion import ddpm_loss


def init_training(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step / warmup_steps, 1.0) if warmup_steps else 1.0
    )
    ema_model = deepcopy(model).requires_grad_(False).eval()
    return optimizer, scheduler, ema_model


def train_step(model, x_0, optimizer, scheduler=None, ema_model=None, *, grad_clip=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = ddpm_loss(model, x_0)
    loss.backward()
    clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    if ema_model is not None:
        with torch.no_grad():
            for average, weight in zip(ema_model.parameters(), model.parameters(), strict=True):
                average.lerp_(weight, 1 - ema_decay)
    if scheduler is not None:
        scheduler.step()
    return loss.detach()
