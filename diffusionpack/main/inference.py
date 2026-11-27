import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from ..backbone import DDPMUNet
from ..diffusion import p_sample_loop
from .train import choose_device


def run_inference(args):
    if args.samples < 1 or args.batch_size < 1:
        parser.error("samples and batch-size must be positive")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    model = DDPMUNet(**config).to(device)
    model.load_state_dict(checkpoint[args.weights])
    model.eval()
    generated = []
    for start in range(0, args.samples, args.batch_size):
        count = min(args.batch_size, args.samples - start)
        result = p_sample_loop(
            model, (count, config["in_channels"], config["resolution"], config["resolution"]),
            device=device, progress=args.progress
        )
        generated.append(result.clamp(-1, 1).add(1).div(2).cpu())
        print(f"generated={start + count}/{args.samples}", flush=True)
    images = torch.cat(generated)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, path, nrow=min(4, args.samples))
    print(f"saved={path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="samples.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    run_inference(parser.parse_args())


if __name__ == "__main__":
    main()
