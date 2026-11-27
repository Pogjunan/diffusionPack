from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..backbone.restore_unet import RestoreUNet
from ..config import ema_decay, learning_rate, warmup_steps
from ..restore import restore_loss
from .restore_data import CifarJpeg, PairedImages
from .train import init_training, load_checkpoint, model_settings, save_checkpoint, start_training


def train_restore(config):
    if (config.steps is None) == (config.epochs is None):
        raise ValueError("Choose either epochs or steps")
    if min(config.batch_size, config.save_every, config.steps or config.epochs or 0) < 1:
        raise ValueError("Batch size, training length and save interval must be positive")

    device = start_training(config)
    settings = model_settings(config.model)
    del settings["in_channels"]
    settings.update(resolution=config.resolution, condition_channels=config.condition_channels,
                    target_channels=config.target_channels)
    if config.dataset == "paired":
        dataset = PairedImages(config.condition_dir, config.target_dir, config.resolution,
                               config.condition_channels, config.target_channels)
    elif config.dataset == "cifar10_jpeg":
        if config.target_channels != 3 or config.condition_channels != 3:
            raise ValueError("CIFAR JPEG needs three condition and three target channels")
        dataset = CifarJpeg(config.data_root, config.resolution, quality=config.jpeg_quality)
    else:
        raise ValueError("Choose cifar10_jpeg or paired")

    saved = load_checkpoint(config.resume) if config.resume else None
    training = {"task": "restore"}
    if config.dataset == "cifar10_jpeg":
        training["jpeg_quality"] = config.jpeg_quality
    previous = saved["training"] if saved else {}
    if saved and (saved["dataset"] != config.dataset or saved["model_config"] != settings
                  or any(previous.get(key) != value for key, value in training.items())):
        raise ValueError("Resume with the same restoration model and dataset settings")
    lr = previous.get("lr", learning_rate) if config.lr is None else config.lr
    warmup = previous.get("warmup", warmup_steps) if config.warmup_steps is None else config.warmup_steps
    training.update(lr=lr, warmup=warmup)
    if lr <= 0 or warmup < 0 or (saved and training != previous):
        raise ValueError("Use a positive learning rate and nonnegative warmup matching the checkpoint")

    model = RestoreUNet(**settings).to(device)
    optimizer, scheduler, ema = init_training(model, lr, warmup)
    amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step = 0
    if saved:
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if amp and saved["scaler"]:
            scaler.load_state_dict(saved["scaler"])
        step = saved["step"]
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.workers, pin_memory=device.type == "cuda")
    total = config.steps if config.steps is not None else config.epochs * len(loader)
    if total <= step:
        raise ValueError(f"Checkpoint is already at step {step}, requested {total}")

    path = Path(config.output) / "latest.pt"
    with tqdm(total=total, initial=step, desc="Restore training", disable=not config.progress) as progress:
        while step < total:
            for condition, target in loader:
                condition = condition.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = restore_loss(model, target, condition)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not amp)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale:
                    scheduler.step()
                    with torch.no_grad():
                        for average, weight in zip(ema.parameters(), model.parameters(), strict=True):
                            average.lerp_(weight, 1 - ema_decay)

                step += 1
                progress.update()
                if config.progress and (step == 1 or step % 50 == 0 or step == total):
                    progress.set_postfix(loss=f"{loss.item():.4f}")
                if step % config.save_every == 0 or step == total:
                    save_checkpoint(path, step, model, ema, optimizer, scheduler, scaler,
                                    settings, config.dataset, training)
                if step == total:
                    break
    return path
