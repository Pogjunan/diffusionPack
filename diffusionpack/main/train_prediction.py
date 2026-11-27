from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..backbone.prediction_unet import PredictionUNet
from ..config import ema_decay, learning_rate, warmup_steps
from ..prediction import prediction_loss
from .prediction_data import ForecastDataset, manifest_hash
from .train import choose_device, init_training, model_settings, save_checkpoint


def prediction_settings(config):
    settings = model_settings(config.model)
    del settings["in_channels"]
    settings.update(
        history_steps=config.history_steps,
        map_channels=config.map_channels,
        resolution=config.resolution,
        channel_mult=tuple(config.channel_mult),
    )
    if config.resolution >= 16:
        resolutions = [config.resolution // (2 ** level) for level in range(len(config.channel_mult))]
        if 16 not in resolutions:
            raise ValueError("channel_mult must include a 16x16 resolution level")
    return settings


def train_prediction(config):
    if (config.steps is None) == (config.epochs is None):
        raise ValueError("Choose exactly one of steps and epochs")
    if (config.steps is not None and config.steps < 1 or
        config.epochs is not None and config.epochs < 1 or
        config.batch_size < 1 or config.save_every < 1 or config.log_every < 1):
        raise ValueError("Training length, batch and logging intervals must be positive")
    torch.manual_seed(config.seed)
    device = choose_device(config.device)
    dataset = ForecastDataset(
        config.manifest,
        history_steps=config.history_steps,
        map_channels=config.map_channels,
        resolution=config.resolution,
        cadence_minutes=config.cadence_minutes,
        history_lag_minutes=config.history_lag_minutes,
        lead_minutes=config.lead_minutes,
        value_min=config.value_min,
        value_max=config.value_max,
        require_target=True,
    )
    if not config.units:
        raise ValueError("Set physical units in the prediction config")
    settings = prediction_settings(config)
    current = dict(
        task="prediction",
        manifest_sha256=manifest_hash(config.manifest),
        cadence_minutes=config.cadence_minutes,
        history_lag_minutes=config.history_lag_minutes,
        lead_minutes=config.lead_minutes,
        value_min=config.value_min,
        value_max=config.value_max,
        units=config.units,
    )
    checkpoint = torch.load(config.resume, map_location="cpu", weights_only=True) if config.resume else None
    previous = {"history_lag_minutes": 0, **checkpoint.get("training", {})} if checkpoint else {}
    start_epoch = previous.pop("epoch", None) if checkpoint else 0.0
    lr = previous.get("lr", learning_rate) if config.lr is None else config.lr
    warmup = previous.get("warmup", warmup_steps) if config.warmup_steps is None else config.warmup_steps
    current.update(lr=lr, warmup=warmup)
    if lr <= 0 or warmup < 0:
        raise ValueError("Learning rate must be positive and warmup nonnegative")
    if checkpoint and (checkpoint["model_config"] != settings or checkpoint["dataset"] != "forecast" or previous != current):
        raise ValueError("Resume requires identical model, manifest, timing, scale and optimizer settings")
    model = PredictionUNet(**settings).to(device)
    optimizer, scheduler, ema_model = init_training(model, lr=lr, warmup=warmup)
    amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_step = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if amp and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"]
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=device.type == "cuda",
    )
    target_steps = config.steps if config.steps is not None else config.epochs * len(loader)
    if target_steps <= start_step:
        raise ValueError("Target steps must exceed the saved checkpoint step")
    save_path = Path(config.output) / "latest.pt"
    print(f"train_windows={len(dataset)} exclusions={dataset.rejections} target_steps={target_steps}", flush=True)
    step = start_step
    with tqdm(total=target_steps, initial=step, desc="Prediction training", disable=not config.progress) as bar:
        while step < target_steps:
            for history, target, _, _ in loader:
                history = history.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = prediction_loss(model, target, history)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not amp)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                    with torch.no_grad():
                        for average, weight in zip(ema_model.parameters(), model.parameters(), strict=True):
                            average.lerp_(weight, 1 - ema_decay)
                step += 1
                bar.update(1)
                if step == 1 or step % config.log_every == 0:
                    bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                if step % config.save_every == 0 or step == target_steps:
                    epoch = None if start_epoch is None else start_epoch + (step - start_step) / len(loader)
                    save_checkpoint(
                        save_path, step, model, ema_model, optimizer,
                        scheduler, scaler, settings, "forecast", dict(current, epoch=epoch),
                    )
                if step == target_steps:
                    break
    return save_path
