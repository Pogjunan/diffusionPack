from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def image_files(folder):
    root = Path(folder)
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    files = {}
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        relative = Path(path.name) if root.is_file() else path.relative_to(root)
        key = relative.with_suffix("")
        if key in files:
            raise ValueError(f"Multiple images named {key} in {root}")
        files[key] = path
    if not files:
        raise ValueError(f"No images found in {root}")
    return files


def prepare_image(image, resolution, channels):
    modes = {1: "L", 3: "RGB"}
    if channels not in modes:
        raise ValueError("Use 1 or 3 image channels")
    image = image.convert(modes[channels])
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.BILINEAR)
    return image


def image_tensor(image):
    pixels = torch.from_numpy(np.array(image))
    pixels = pixels.unsqueeze(0) if pixels.ndim == 2 else pixels.permute(2, 0, 1)
    return pixels.float().div_(127.5).sub_(1)


class PairedImages(Dataset):
    def __init__(self, condition_dir, target_dir, resolution, condition_channels=3, target_channels=3):
        conditions = image_files(condition_dir)
        targets = image_files(target_dir)
        unmatched = conditions.keys() ^ targets.keys()
        if unmatched:
            raise ValueError(f"Missing image pair for {min(unmatched)}")
        self.pairs = [(path, targets[key]) for key, path in conditions.items()]
        self.resolution = resolution
        self.condition_channels = condition_channels
        self.target_channels = target_channels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = self.pairs[index]
        with Image.open(source) as condition, Image.open(target) as clean:
            if condition.size != clean.size:
                raise ValueError(f"Image sizes differ for {source} and {target}")
            condition = prepare_image(condition, self.resolution, self.condition_channels)
            clean = prepare_image(clean, self.resolution, self.target_channels)
        return image_tensor(condition), image_tensor(clean)


class ConditionImages(Dataset):
    def __init__(self, condition_dir, resolution, channels=3):
        self.images = list(image_files(condition_dir).items())
        self.resolution = resolution
        self.channels = channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        key, path = self.images[index]
        with Image.open(path) as image:
            condition = prepare_image(image, self.resolution, self.channels)
        return image_tensor(condition), f"{key}.png"


class CifarJpeg(Dataset):
    def __init__(self, data_root, resolution=32, channels=3, *, train=True, quality=10, download=True):
        from torchvision.datasets import CIFAR10

        if channels != 3 or not 1 <= quality <= 95:
            raise ValueError("CIFAR JPEG uses RGB and a JPEG quality from 1 to 95")
        self.images = CIFAR10(data_root, train=train, download=download)
        self.resolution = resolution
        self.quality = quality

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index][0]
        clean = prepare_image(image, self.resolution, 3)
        with BytesIO() as stream:
            clean.save(stream, format="JPEG", quality=self.quality)
            stream.seek(0)
            with Image.open(stream) as image:
                condition = image.convert("RGB")
        return image_tensor(condition), image_tensor(clean)
