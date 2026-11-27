from dataclasses import dataclass


@dataclass
class RestoreTrainConfig:
    dataset: str = "cifar10_jpeg"
    data_root: str = "data"
    condition_dir: str = "data/train/condition"
    target_dir: str = "data/train/target"
    jpeg_quality: int = 10
    model: str = "light"
    resolution: int = 32
    condition_channels: int = 3
    target_channels: int = 3
    batch_size: int = 8
    epochs: int | None = 30
    steps: int | None = None
    lr: float | None = None
    warmup_steps: int | None = None
    workers: int = 0
    save_every: int = 1000
    output: str = "runs/restore"
    resume: str | None = None
    device: str = "auto"
    amp: bool = True
    progress: bool = True
    seed: int = 42


@dataclass
class RestoreInferenceConfig:
    checkpoint: str = "runs/restore/latest.pt"
    condition_path: str | None = None
    data_root: str = "data"
    output: str = "runs/restore/images"
    samples: int | None = None
    batch_size: int = 2
    device: str = "auto"
    weights: str = "ema"
    progress: bool = True
    seed: int = 42
