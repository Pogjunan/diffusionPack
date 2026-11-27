from dataclasses import dataclass

num_diff_steps = 1000
beta_start = 0.0001
beta_end = 0.02

learning_rate = 0.0002
warmup_steps = 5000
ema_decay = 0.9999


@dataclass
class TrainConfig:
    model: str = "light"
    dataset: str = "cifar10"
    data_root: str = "data"
    channels: int = 3
    batch_size: int = 8
    steps: int | None = None
    epochs: int | None = 30
    lr: float | None = None
    warmup_steps: int | None = None
    workers: int = 0
    max_images: int = 0
    output: str | None = "runs/cifar_light"
    resume: str | None = None
    save_every: int = 1000
    log_every: int = 100
    device: str = "auto"
    seed: int = 42
    amp: bool = True
    flip: bool = True
    progress: bool = True


@dataclass
class InferenceConfig:
    checkpoint: str = "runs/cifar_light/latest.pt"
    samples: int = 8
    batch_size: int = 2
    output: str = "runs/cifar_light/samples.png"
    device: str = "auto"
    seed: int = 42
    weights: str = "ema"
    progress: bool = True
