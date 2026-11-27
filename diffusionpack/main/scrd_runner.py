import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from ..scrd.comparison import make_diffusion
from ..scrd_config import SCRDConfig
from .prediction_data import ForecastDataset, inverse_scale
from .train import choose_device


def dataset(config, split, require_target=True):
    source = ForecastDataset(
        getattr(config, f"{split}_csv"), history_steps=config.history_steps,
        future_steps=config.future_steps, map_channels=config.map_channels,
        resolution=config.resolution, cadence_minutes=config.cadence_minutes,
        lead_minutes=config.lead_minutes, history_lag_minutes=config.history_lag_minutes,
        value_min=config.value_min, value_max=config.value_max, require_target=require_target,
        origin_time=config.origin_time if split == "test" or not require_target else None,
        target_csv=getattr(config, f"{split}_targets"),
    )
    print(f"{split}: {len(source)} origins; exclusions={source.rejections}")
    return source


def check_splits(train, valid):
    train_targets = {time for _, _, times in train.windows for time in times}
    valid_targets = {time for _, _, times in valid.windows for time in times}
    train_inputs = {time for _, times, _ in train.windows for time in times}
    valid_inputs = {time for _, times, _ in valid.windows for time in times}
    overlap = train_targets & valid_targets
    if overlap:
        raise ValueError(f"Training and validation targets overlap: {min(overlap)}")
    print(f"Shared history times: {len(train_inputs & valid_inputs)}; "
          f"validation targets in training history: {len(valid_targets & train_inputs)}; "
          f"training targets in validation history: {len(train_targets & valid_inputs)}")


def save_checkpoint(path, diffusion, optimizer, config, step, epoch, cursor, best, train_origins, valid_origins):
    state = dict(format="scrd_v1", config=vars(config), model=diffusion.state_dict(),
                 optimizer=optimizer.state_dict(), step=step, epoch=epoch, cursor=cursor,
                 best=best, train_origins=train_origins, valid_origins=valid_origins, rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary = Path(path).with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_run(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "scrd_v1":
        raise ValueError("Use a checkpoint from this SCRD runner; old checkpoints require separate review")
    config = SCRDConfig(**saved["config"])
    diffusion = make_diffusion(config).to(device)
    diffusion.load_state_dict(saved["model"], strict=True)
    return config, diffusion, saved


def selected_indices(source, count, seed):
    if count is None:
        return list(range(len(source)))
    if type(count) is not int or not 1 <= count <= len(source):
        raise ValueError(f"Requested {count} origins; available {len(source)}")
    return sorted(torch.randperm(len(source), generator=torch.Generator().manual_seed(seed))[:count].tolist())


@torch.no_grad()
def forecast(config, diffusion, source, *, count, members, output=None, persistence=False):
    if members < 1:
        raise ValueError("Use at least one ensemble member")
    device = choose_device(config.device) if persistence else next(diffusion.parameters()).device
    generator = torch.Generator(device=device).manual_seed(config.seed)
    rows, outputs = [], []
    indices = selected_indices(source, count, config.seed)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps(vars(config), indent=2))
    for index in indices:
        condition, target, origin, valid_times = source[index]
        condition = condition.unsqueeze(0).to(device)
        predictions = []
        for member in range(members):
            sample = (condition[:, -config.map_channels:].repeat(1, config.future_steps, 1, 1)
                      if persistence else diffusion.sample(condition, generator=generator, progress=config.progress))
            physical = inverse_scale(sample[0].float().cpu(), config.value_min, config.value_max)
            if not torch.isfinite(physical).all():
                raise FloatingPointError(f"Non-finite physical forecast at {origin}")
            predictions.append(physical.reshape(config.future_steps, config.map_channels, *physical.shape[-2:]))
        predictions = torch.stack(predictions)  # [members, future, channels, height, width]
        point = predictions.mean(0)
        stamp = origin.replace(":", "").replace("+0000", "Z")
        if output is not None:
            for member in range(members):
                for k, valid in enumerate(valid_times):
                    name = f"{stamp}_{k + 1:02d}_sample_{member + 1:03d}.npy"
                    np.save(output / name, predictions[member, k].numpy())
                    outputs.append(dict(origin=origin, timestamp=valid, member=member + 1,
                                        path=name, units=config.units))
        if target.numel():
            truth = inverse_scale(target, config.value_min, config.value_max).reshape_as(point)
            for k, valid in enumerate(valid_times):
                error = point[k] - truth[k]
                bias = error.mean()
                rows.append(dict(origin=origin, timestamp=valid,
                                 lead_minutes=config.lead_minutes + k * config.cadence_minutes,
                                 mse=error.square().mean().item(), mae=error.abs().mean().item(),
                                 bias=bias.item(), centered_mse=(error - bias).square().mean().item(),
                                 sample_mse=(predictions[:, k] - truth[k]).square().mean().item()))
    scores = pd.DataFrame(rows)
    if not scores.empty and not np.isfinite(scores[["mse", "mae", "bias", "centered_mse", "sample_mse"]]).all().all():
        raise FloatingPointError("Non-finite evaluation metric")
    if output is not None:
        pd.DataFrame(outputs).to_csv(output / "forecasts.csv", index=False)
        if not scores.empty:
            scores.to_csv(output / "errors.csv", index=False)
            scores.groupby("lead_minutes")[["mse", "mae", "bias", "centered_mse", "sample_mse"]].mean().to_csv(
                output / "metrics.csv")
    return scores


def train(config, check_data=False):
    positive = (config.steps, config.batch_size, config.save_every, config.log_every, config.cpu_threads)
    if any(type(value) is not int or value < 1 for value in positive):
        raise ValueError("Training steps, batch size, intervals and CPU threads must be positive integers")
    if config.workers < 0 or config.validation_every < 0 or not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("Invalid workers, validation interval or learning rate")
    if config.precision not in ("float32", "bfloat16") or not config.units:
        raise ValueError("Choose float32/bfloat16 and set the physical units")
    if config.grad_clip is not None and (not np.isfinite(config.grad_clip) or config.grad_clip <= 0):
        raise ValueError("grad_clip must be None or finite and positive")
    torch.set_num_threads(config.cpu_threads)
    train_set = dataset(config, "train")
    valid_set = dataset(config, "valid") if config.valid_csv else None
    if valid_set is not None:
        check_splits(train_set, valid_set)
        selected_indices(valid_set, config.validation_origins, config.seed)
    if check_data:
        for source in [train_set] + ([] if valid_set is None else [valid_set]):
            for index in range(len(source)):
                source[index]
        print("Every usable training/validation window loaded. No model trained.")
        return
    device = choose_device(config.device)
    if config.precision == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 training requires a supported CUDA GPU")
    torch.manual_seed(config.seed)
    diffusion = make_diffusion(config).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=config.learning_rate)
    output = Path(config.output)
    step = epoch = cursor = 0
    best = float("inf")
    train_origins = [origin.isoformat() for origin, _, _ in train_set.windows]
    valid_origins = [] if valid_set is None else [origin.isoformat() for origin, _, _ in valid_set.windows]
    if config.resume:
        saved = torch.load(config.resume, map_location="cpu", weights_only=True)
        if saved.get("format") != "scrd_v1":
            raise ValueError("Resume requires a checkpoint from this runner")
        runtime = {"steps", "output", "resume", "device", "workers", "cpu_threads", "save_every", "log_every",
                   "progress", "checkpoint", "samples", "max_origins", "origin_time", "test_csv", "test_targets"}
        differences = [key for key, value in vars(config).items()
                       if key not in runtime and saved["config"].get(key) != value]
        if differences or saved["train_origins"] != train_origins or saved["valid_origins"] != valid_origins:
            raise ValueError(f"Resume data/model/training settings differ: {differences}")
        if output.exists() and any(output.iterdir()):
            checkpoint_path = Path(config.resume).resolve()
            if checkpoint_path.parent != output.resolve() or checkpoint_path.name != "latest.pt":
                raise ValueError("Resume latest.pt in its own run folder, or use a new empty output folder")
            log = output / "loss.csv"
            if not log.exists() or int(pd.read_csv(log)["step"].iloc[-1]) != saved["step"]:
                raise ValueError("Checkpoint and loss log disagree; resume into a new empty folder")
        diffusion.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch, cursor, best = saved["step"], saved["epoch"], saved["cursor"], saved["best"]
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output folder or resume explicitly: {output}")
    if step >= config.steps:
        raise ValueError("The requested total optimizer steps have already been reached")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(config), indent=2))
    records = []
    print(f"Device={device}; parameters={sum(p.numel() for p in diffusion.parameters()):,}; "
          f"alpha_bar_T={diffusion.alpha_bars[-1].item():.8g}; updates={step}->{config.steps}")
    while step < config.steps:
        # A deterministic epoch order and cursor allow resuming in the middle of an epoch.
        order = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(config.seed + epoch)).tolist()
        loader = DataLoader(Subset(train_set, order[cursor:]), batch_size=config.batch_size,
                            num_workers=config.workers, shuffle=False, pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(config.seed + epoch))
        for condition, future, _, _ in loader:
            diffusion.train()
            optimizer.zero_grad(set_to_none=True)
            condition, future = condition.to(device), future.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=config.precision == "bfloat16"):
                loss = diffusion.loss(future, condition)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at update {step + 1}")
            loss.backward()
            limit = float("inf") if config.grad_clip is None else config.grad_clip
            gradient_norm = torch.nn.utils.clip_grad_norm_(diffusion.parameters(), limit, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            cursor += len(condition)
            if cursor == len(train_set):
                epoch, cursor = epoch + 1, 0
            records.append(dict(step=step, loss=loss.item(), gradient_norm=gradient_norm.item()))
            if step == 1 or step % config.log_every == 0:
                print(f"step={step} epsilon_mse={loss.item():.6f}", flush=True)
            validate = valid_set is not None and config.validation_every and (
                step % config.validation_every == 0 or step == config.steps)
            if validate:
                scores = forecast(config, diffusion, valid_set, count=config.validation_origins,
                                  members=config.validation_samples)
                value = scores["mse"].mean()
                scores.to_csv(output / f"validation_{step:07d}.csv", index=False)
                if value < best:
                    best = value
                    save_checkpoint(output / "best.pt", diffusion, optimizer, config, step, epoch, cursor, best,
                                    train_origins, valid_origins)
                print(f"validation ensemble-mean MSE={value:.6g} {config.units}^2; best={best:.6g}")
            if step % config.save_every == 0 or step == config.steps:
                save_checkpoint(output / "latest.pt", diffusion, optimizer, config, step, epoch, cursor, best,
                                train_origins, valid_origins)
                path = output / "loss.csv"
                pd.DataFrame(records).to_csv(path, mode="a", header=not path.exists(), index=False)
                records.clear()
            if step == config.steps:
                break
    return output / "latest.pt"


def infer(config, *, evaluate=False, persistence=False):
    torch.set_num_threads(config.cpu_threads)
    device = choose_device(config.device)
    if persistence:
        if not config.units:
            raise ValueError("Set the physical units")
        source = dataset(config, "test", require_target=True)
        return forecast(config, None, source, count=config.max_origins, members=1,
                        output=config.output, persistence=True)
    trained, diffusion, saved = load_run(config.checkpoint, device)
    # Architecture, scale and cadence come from the checkpoint, never current defaults.
    for key in ("test_csv", "test_targets", "samples", "max_origins", "origin_time", "seed", "progress", "output", "checkpoint"):
        setattr(trained, key, getattr(config, key))
    source = dataset(trained, "test", require_target=evaluate)
    if evaluate:
        test_times = {time for _, _, times in source.windows for time in times}
        for split in ("train", "valid"):
            origins = pd.to_datetime(saved[f"{split}_origins"], utc=True, format="ISO8601")
            previous_targets = {origin + pd.Timedelta(minutes=trained.lead_minutes + k * trained.cadence_minutes)
                                for origin in origins for k in range(trained.future_steps)}
            if test_times & previous_targets:
                raise ValueError(f"Test targets overlap the checkpoint's {split} targets")
    scores = forecast(trained, diffusion, source, count=config.max_origins, members=config.samples,
                      output=config.output, persistence=persistence)
    if not scores.empty:
        print(scores.groupby("lead_minutes")[["mse", "mae", "bias", "centered_mse"]].mean().to_string())
    return scores


@torch.no_grad()
def diagnose(config):
    torch.set_num_threads(config.cpu_threads)
    device = choose_device(config.device)
    trained, diffusion, _ = load_run(config.checkpoint, device)
    trained.valid_csv, trained.valid_targets = config.valid_csv, config.valid_targets
    source = dataset(trained, "valid")
    diffusion.eval()
    generator = torch.Generator(device=device).manual_seed(config.seed)
    rows = []
    levels = sorted({0, *(round((diffusion.steps - 1) * fraction) for fraction in (.05, .2, .5, .8, 1.0))})
    for index in selected_indices(source, config.max_origins, config.seed):
        condition, future, origin, _ = source[index]
        condition, future = condition[None].to(device), future[None].to(device)
        clean = diffusion.make_target(future, condition)
        for level in levels:
            t = torch.tensor([level], device=device)
            noise = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
            noisy = diffusion.q_sample(clean, t, noise)
            predicted = diffusion.predict_noise(noisy, t, condition)
            error = predicted - noise
            reconstructed = diffusion.predict_x0(noisy, t, predicted)
            if not torch.isfinite(reconstructed).all():
                raise FloatingPointError(f"Non-finite reconstruction at {origin}, t={level}")
            rows.append(dict(origin=origin, t=level, alpha_bar=diffusion.alpha_bars[level].item(),
                             epsilon_mse=error.square().mean().item(), epsilon_bias=error.mean().item(),
                             clean_mse=(reconstructed - clean).square().mean().item(),
                             clean_bias=(reconstructed - clean).mean().item()))
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "noise_levels.csv", index=False)
    print(scores.groupby("t")[["epsilon_mse", "epsilon_bias", "clean_mse", "clean_bias"]].mean().to_string())
    return scores
