from .config import Config


class RestoreTrainConfig(Config):
    dataset = "cifar10_jpeg"
    data_root = "data"
    condition_dir = "data/train/condition"
    target_dir = "data/train/target"
    jpeg_quality = 10
    model = "light"
    resolution = 32
    condition_channels = 3
    target_channels = 3
    batch_size = 8
    epochs = 30
    steps = None
    lr = None
    warmup_steps = None
    workers = 0
    save_every = 1000
    output = "runs/restore"
    resume = None
    device = "auto"
    amp = True
    progress = True
    base_channels = None


class RestoreInferenceConfig(Config):
    checkpoint = "runs/restore/latest.pt"
    condition_path = None
    data_root = "data"
    output = "runs/restore/images"
    samples = None
    batch_size = 2
    device = "auto"
    weights = "ema"
    progress = True
    sampler = "ddpm"
    sampling_steps = None
    eta = 0.0
