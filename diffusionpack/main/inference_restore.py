from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image

from ..backbone.restore_unet import RestoreUNet
from ..restore import restore_sample
from .restore_data import CifarJpeg, ConditionImages
from .train import choose_device


@torch.inference_mode()
def infer_restore(config):
    if config.batch_size < 1 or (config.samples is not None and config.samples < 1):
        raise ValueError("Batch size and sample count must be positive")
    if config.weights not in ("ema", "model"):
        raise ValueError("Choose ema or model weights")
    torch.manual_seed(config.seed)
    device = choose_device(config.device)
    saved = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
    training = saved.get("training", {})
    if training.get("task") != "restore":
        raise ValueError("Choose a restoration checkpoint")
    settings = saved["model_config"]
    model = RestoreUNet(**settings).to(device)
    model.load_state_dict(saved[config.weights])
    model.eval()

    output = Path(config.output)
    if config.condition_path is not None:
        source = Path(config.condition_path).resolve()
        if source.is_dir() and output.resolve().is_relative_to(source):
            raise ValueError("Save restored images outside the condition folder")
        dataset = ConditionImages(source, settings["resolution"], settings["condition_channels"])
        count = len(dataset) if config.samples is None else config.samples
    elif saved["dataset"] == "cifar10_jpeg":
        dataset = CifarJpeg(config.data_root, settings["resolution"], train=False,
                            quality=training["jpeg_quality"])
        count = 8 if config.samples is None else config.samples
    else:
        raise ValueError("Set condition_path to a degraded image or a folder")
    selection = Subset(dataset, range(min(count, len(dataset))))
    loader = DataLoader(selection, batch_size=config.batch_size)
    written = 0
    for condition, reference in loader:
        result = restore_sample(model, condition.to(device), progress=config.progress)
        for index, image in enumerate(result):
            name = reference[index] if config.condition_path is not None else f"{written:06d}.png"
            path = output / name
            if config.condition_path is not None and path.resolve().is_relative_to(source):
                raise ValueError("Choose an output path outside the input")
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = image.clamp(-1, 1).add(1).mul(127.5).round().byte().cpu()
            to_pil_image(pixels).save(path)
            written += 1
    return written
