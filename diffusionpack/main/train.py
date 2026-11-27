import argparse
from copy import deepcopy
from pathlib import Path

import torch
from tqdm.auto import tqdm
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset

from ..backbone import DDPMUNet
from ..config import ema_decay, learning_rate, warmup_steps
from ..diffusion import ddpm_loss


class FolderImages(Dataset):
    def __init__(self, folder, transform):
        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        self.paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in extensions)
        if not self.paths:
            raise ValueError(f"No images found in {folder}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def model_settings(preset="light", channels=3):
    if preset not in ("light", "original"):
        raise ValueError("preset must be light or original")
    return dict(
        resolution=32,
        in_channels=channels,
        base_channels=64 if preset == "light" else 128,
        channel_mult=(1, 2, 2, 2),
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


def train_step(model, x_0, optimizer, scheduler=None, ema_model=None, *, grad_clip=1.0, scaler=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp = scaler is not None and scaler.is_enabled()
    with torch.autocast(device_type=x_0.device.type, enabled=amp):
        loss = ddpm_loss(model, x_0)
    if amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), grad_clip)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updated = scaler.get_scale() >= previous_scale
    else:
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        updated = True
    if updated:
        if ema_model is not None:
            with torch.no_grad():
                for average, weight in zip(ema_model.parameters(), model.parameters(), strict=True):
                    average.lerp_(weight, 1 - ema_decay)
        if scheduler is not None:
            scheduler.step()
    return loss.detach()


def get_dataset(args, channels):
    from torchvision import datasets, transforms

    ops = []
    if args.dataset == "folder":
        ops.extend([transforms.Resize(32), transforms.CenterCrop(32)])
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
    else:
        dataset = datasets.FakeData(size=256, image_size=(3, 32, 32), transform=transform)
    if args.max_images > 0:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    return dataset


def save_checkpoint(path, step, model, ema_model, optimizer, scheduler, scaler, config, dataset, training=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
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
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def choose_device(value="auto"):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install CUDA-enabled PyTorch and check the NVIDIA driver")
    return device


def run_training(args):
    if args.steps is not None and args.epochs is not None:
        raise ValueError("Specify either steps or epochs, not both")
    if (args.steps is not None and args.steps < 1 or
        args.epochs is not None and args.epochs < 1 or
        args.batch_size < 1 or args.save_every < 1 or args.log_every < 1 or
        args.lr is not None and args.lr <= 0 or
        args.warmup_steps is not None and args.warmup_steps < 0):
        parser.error("Invalid steps, epochs, batch size, learning rate or warmup")

    args.output = args.output or (str(Path(args.resume).parent) if args.resume else f"runs/{args.model}_{args.dataset}")
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    config = checkpoint["model_config"] if checkpoint else model_settings(args.model, args.channels)
    if checkpoint and checkpoint["dataset"] != args.dataset:
        raise ValueError("The resume checkpoint uses a different dataset type")
    previous_training = checkpoint.get("training", {}) if checkpoint else {}
    previous_lr = previous_training.get(
        "lr", checkpoint["scheduler"]["base_lrs"][0] if checkpoint else learning_rate
    )
    previous_warmup = previous_training.get("warmup", warmup_steps)
    if checkpoint and args.lr is not None and args.lr != previous_lr:
        raise ValueError("Resume must use the saved learning rate")
    if checkpoint and args.warmup_steps is not None and args.warmup_steps != previous_warmup:
        raise ValueError("Resume must use the saved warmup steps")
    lr = previous_lr if checkpoint else (args.lr if args.lr is not None else learning_rate)
    warmup = previous_warmup if checkpoint else (
        args.warmup_steps if args.warmup_steps is not None else warmup_steps
    )
    training = {"lr": lr, "warmup": warmup}
    model = DDPMUNet(**config).to(device)
    optimizer, scheduler, ema_model = init_training(model, lr=lr, warmup=warmup)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_step = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if use_amp and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"]

    dataset = get_dataset(args, config["in_channels"])
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    steps_per_epoch = len(loader)
    target_steps = args.steps if args.steps is not None else (
        args.epochs * steps_per_epoch if args.epochs is not None else 30000
    )
    if target_steps <= start_step:
        raise ValueError(f"Already trained {start_step} steps; target is {target_steps}")
    iterator = iter(loader)
    progress = tqdm(
        range(start_step + 1, target_steps + 1), total=target_steps, initial=start_step,
        desc="DDPM training", dynamic_ncols=True, disable=not args.progress,
    )
    print(f"target_steps={target_steps} steps_per_epoch={steps_per_epoch} learning_rate={lr} warmup={warmup}", flush=True)
    print(f"device={device} channels={config['in_channels']} base={config['base_channels']} images={len(dataset)} parameters={sum(p.numel() for p in model.parameters()):,} amp={use_amp}", flush=True)
    checkpoint_path = Path(args.output) / "latest.pt"
    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device, non_blocking=True)
        applied_lr = optimizer.param_groups[0]["lr"]
        loss = train_step(model, images, optimizer, scheduler, ema_model, scaler=scaler)
        if step == 1 or step % args.log_every == 0:
            vram = f" vram={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}GiB" if device.type == "cuda" else ""
            status = f"step={step}/{target_steps} epoch={step / steps_per_epoch:.2f} loss={loss.item():.6f} lr={applied_lr:.8f}{vram}"
            if args.progress:
                progress.set_postfix_str(f"loss={loss.item():.4f} lr={applied_lr:.2e}{vram}")
            else:
                print(status, flush=True)
        if step % args.save_every == 0 or step == target_steps:
            save_checkpoint(checkpoint_path, step, model, ema_model, optimizer, scheduler, scaler, config, args.dataset, training=training)
            if args.progress:
                tqdm.write(f"saved={checkpoint_path}")
            else:
                print(f"saved={checkpoint_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("light", "original"), default="light")
    parser.add_argument("--dataset", choices=("cifar10", "folder", "fake"), default="cifar10")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
