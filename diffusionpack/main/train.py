import argparse
from copy import deepcopy
from math import ceil, isfinite
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from tqdm.auto import tqdm
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset

from ..backbone import DDPMUNet
from ..backbone.ddpm_unet import default_channel_mult
from ..config import ema_decay, learning_rate, random_seed, warmup_steps
from ..diffusion import betas, ddpm_loss


ddpm_schedule = (len(betas), betas[0].item(), betas[-1].item())


class FolderImages(Dataset):
    def __init__(self, folder, transform):
        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        self.paths = sorted(path for path in Path(folder).rglob("*")
                            if path.suffix.lower() in extensions and path.is_file())
        if not self.paths:
            raise ValueError(f"No images found in {folder}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def model_settings(preset="light", channels=3, resolution=32, base_channels=None):
    if preset not in ("light", "original"):
        raise ValueError("preset must be light or original")
    if base_channels is None:
        base_channels = 64 if preset == "light" else 128
    if base_channels < 32 or base_channels % 32:
        raise ValueError("base_channels must be a multiple of 32")
    return dict(
        resolution=resolution,
        in_channels=channels,
        base_channels=base_channels,
        channel_mult=default_channel_mult(resolution),
        num_res_blocks=1 if preset == "light" else 2,
        attention_resolutions=(16,),
        dropout=0.1,
    )


def init_training(model, lr=learning_rate, warmup=warmup_steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step / warmup, 1.0) if warmup else 1.0
    )
    ema_model = deepcopy(model).requires_grad_(False).eval()
    return optimizer, scheduler, ema_model


def train_step(model, x_0, optimizer, scheduler=None, ema_model=None, *, grad_clip=1.0,
               scaler=None, loss_function=ddpm_loss, condition=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    with torch.autocast(device_type=x_0.device.type, enabled=scaler.is_enabled()):
        if condition is None:
            loss = loss_function(model, x_0)
        else:
            loss = loss_function(model, x_0, condition)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=not scaler.is_enabled())
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale = scaler.get_scale()
    if scale == 0:
        raise FloatingPointError("AMP scale reached zero after non-finite gradients")
    if scale < previous_scale:
        return loss.detach()

    if ema_model is not None:
        # EMA ← d·EMA + (1−d)·θ
        with torch.no_grad():
            for average, weight in zip(ema_model.parameters(), model.parameters(), strict=True):
                average.lerp_(weight, 1 - ema_decay)
    if scheduler is not None:
        scheduler.step()
    return loss.detach()


def get_dataset(args, channels, resolution=32):
    from torchvision import datasets, transforms

    ops = []
    if args.dataset == "folder" or (args.dataset == "cifar10" and resolution != 32):
        ops.extend([transforms.Resize(resolution), transforms.CenterCrop(resolution)])
    if args.flip:
        ops.append(transforms.RandomHorizontalFlip())
    if channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * channels, [0.5] * channels),
    ])
    transform = transforms.Compose(ops)
    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
    elif args.dataset == "folder":
        dataset = FolderImages(args.data_root, transform)
    elif args.dataset == "fake":
        dataset = datasets.FakeData(size=256, image_size=(3, resolution, resolution), transform=transform)
    else:
        raise ValueError("Choose cifar10, folder or fake")
    if args.max_images > 0:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    return dataset


def save_checkpoint(path, step, model, ema_model, optimizer, scheduler, scaler, config, dataset,
                    training=None, *, batches=None, epoch=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "ddpm_schedule": ddpm_schedule,
        "step": step,
        "model_config": config,
        "dataset": dataset,
        "training": training or {},
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if batches is not None:
        state.update(batches=batches, epoch=epoch)
    with TemporaryDirectory(dir=path.parent) as folder:
        temporary = Path(folder) / path.name
        torch.save(state, temporary)
        temporary.replace(path)


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    trained_schedule = saved.get("ddpm_schedule", (1000, 0.0001, 0.02))
    if trained_schedule != ddpm_schedule:
        raise ValueError(f"Checkpoint DDPM schedule {trained_schedule} differs from {ddpm_schedule}")
    return saved


def choose_device(value="auto"):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install CUDA-enabled PyTorch and check the NVIDIA driver")
    return device


def start_training(config):
    if config.steps is not None and config.epochs is not None:
        raise ValueError("Choose either epochs or steps")
    counts = (config.steps, config.epochs, config.batch_size, config.save_every, getattr(config, "log_every", 1))
    if any(not isinstance(n, Integral) or n < 1 for n in counts if n is not None):
        raise ValueError("Training length, batch size and intervals must be positive integers")
    if config.lr is not None and not (isfinite(config.lr) and config.lr > 0):
        raise ValueError("Learning rate must be finite and positive")
    if config.warmup_steps is not None and not (isinstance(config.warmup_steps, Integral) and config.warmup_steps >= 0):
        raise ValueError("Warmup steps must be a nonnegative integer")
    if not config.resume:
        torch.manual_seed(random_seed)
    return choose_device(config.device)


def train_batches(config, model, dataset, settings, dataset_name, training, checkpoint, loss_function=ddpm_loss):
    task = training.get("task", "unconditional")
    log_every = getattr(config, "log_every", 50)
    previous = dict(checkpoint.get("training", {})) if checkpoint else {}
    default_lr = checkpoint["scheduler"]["base_lrs"][0] if checkpoint else learning_rate
    saved_lr = previous.get("lr", default_lr)
    saved_warmup = previous.get("warmup", warmup_steps)
    lr = saved_lr if config.lr is None else config.lr
    warmup = saved_warmup if config.warmup_steps is None else config.warmup_steps
    if checkpoint:
        if task == "prediction":
            previous.setdefault("history_lag_minutes", 0)
        if (checkpoint["model_config"] != settings or checkpoint["dataset"] != dataset_name
            or any(previous.get(key) != value for key, value in training.items())
            or lr != saved_lr or warmup != saved_warmup):
            raise ValueError("Resume with the same model, dataset and training settings")
    training = dict(training, lr=lr, warmup=warmup)

    device = next(model.parameters()).device
    optimizer, scheduler, ema = init_training(model, lr, warmup)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers,
                        pin_memory=device.type == "cuda")
    batches = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler.is_enabled() and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        batches = checkpoint.get("batches", checkpoint["step"])
    step = scheduler.last_epoch
    epoch = checkpoint.get("epoch", previous.get("epoch")) if checkpoint else 0
    epoch_offset = 0 if epoch is None else epoch - batches / len(loader)
    if config.epochs is not None:
        total = ceil((config.epochs - epoch_offset) * len(loader))
        completed = batches
    else:
        total = 30000 if config.steps is None else config.steps
        completed = step
    if completed >= total:
        raise ValueError("Training target has already been reached")

    path = Path(config.output) / "latest.pt"
    with tqdm(total=total, initial=completed, desc=f"{task} training", disable=not config.progress) as bar:
        while completed < total:
            for batch in loader:
                condition = None
                if task == "unconditional":
                    target = batch[0] if isinstance(batch, (list, tuple)) else batch
                else:
                    condition, target = batch[:2]
                    condition = condition.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                lr_used = optimizer.param_groups[0]["lr"]
                loss = train_step(model, target, optimizer, scheduler, ema, scaler=scaler,
                                  loss_function=loss_function, condition=condition)
                batches += 1
                updated = scheduler.last_epoch > step
                step = scheduler.last_epoch
                count = batches if config.epochs is not None else step
                bar.update(count - completed)
                completed = count
                epoch = epoch_offset + batches / len(loader)
                if updated and (step == 1 or step % log_every == 0):
                    message = f"step={step} epoch={epoch:.2f} loss={loss.item():.5f} lr={lr_used:.2e}"
                    if device.type == "cuda":
                        message += f" vram={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}GiB"
                    if config.progress:
                        bar.set_postfix_str(message)
                    else:
                        print(message, flush=True)
                if completed == total or (updated and step % config.save_every == 0):
                    if task == "prediction":
                        training["epoch"] = epoch
                    save_checkpoint(path, step, model, ema, optimizer, scheduler, scaler,
                                    settings, dataset_name, training, batches=batches, epoch=epoch)
                if completed == total:
                    break
    return path


def run_training(args):
    device = start_training(args)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    settings = (checkpoint["model_config"] if checkpoint else
                model_settings(args.model, args.channels, getattr(args, "resolution", 32),
                               getattr(args, "base_channels", None)))
    if checkpoint and checkpoint["dataset"] != args.dataset:
        raise ValueError("The resume checkpoint uses a different dataset type")
    if not args.output:
        args.output = str(Path(args.resume).parent) if checkpoint else f"runs/{args.model}_{args.dataset}"
    dataset = get_dataset(args, settings["in_channels"], settings["resolution"])
    model = DDPMUNet(**settings).to(device)
    return train_batches(args, model, dataset, settings, args.dataset, {}, checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("light", "original"), default="light")
    parser.add_argument("--dataset", choices=("cifar10", "folder", "fake"), default="cifar10")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--steps", type=int)
    length.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
