import argparse
from copy import deepcopy
from pathlib import Path

import torch
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


def init_training(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step / warmup_steps, 1.0) if warmup_steps else 1.0
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


def save_checkpoint(path, step, model, ema_model, optimizer, scheduler, scaler, config, dataset):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model_config": config,
        "dataset": dataset,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("light", "original"), default="light")
    parser.add_argument("--dataset", choices=("cifar10", "folder", "fake"), default="cifar10")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=30000)
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
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.save_every < 1 or args.log_every < 1:
        parser.error("steps, batch-size, save-every and log-every must be positive")

    args.output = args.output or (str(Path(args.resume).parent) if args.resume else f"runs/{args.model}_{args.dataset}")
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    config = checkpoint["model_config"] if checkpoint else model_settings(args.model, args.channels)
    if checkpoint and checkpoint["dataset"] != args.dataset:
        raise ValueError("The resume checkpoint uses a different dataset type")
    model = DDPMUNet(**config).to(device)
    optimizer, scheduler, ema_model = init_training(model)
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
    iterator = iter(loader)
    print(f"device={device} channels={config['in_channels']} base={config['base_channels']} images={len(dataset)} parameters={sum(p.numel() for p in model.parameters()):,} amp={use_amp}", flush=True)
    checkpoint_path = Path(args.output) / "latest.pt"
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device, non_blocking=True)
        loss = train_step(model, images, optimizer, scheduler, ema_model, scaler=scaler)
        if step == 1 or step % args.log_every == 0:
            vram = f" vram={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}GiB" if device.type == "cuda" else ""
            print(f"step={step} loss={loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.8f}{vram}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(checkpoint_path, step, model, ema_model, optimizer, scheduler, scaler, config, args.dataset)
            print(f"saved={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
