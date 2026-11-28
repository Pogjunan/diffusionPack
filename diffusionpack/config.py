random_seed = 10000
num_diff_steps = 1000
beta_start = 0.0001
beta_end = 0.02

learning_rate = 0.0002
warmup_steps = 5000
ema_decay = 0.9999


class Config:
    def __init__(self, *values, **options):
        fields = [name for name in vars(type(self)) if not name.startswith("_")]
        if len(values) > len(fields):
            raise TypeError("Too many positional settings")
        for name, value in zip(fields, values):
            if name in options:
                raise TypeError(f"Setting supplied twice: {name}")
            options[name] = value
        for name in options:
            if name not in fields:
                raise TypeError(f"Unknown setting: {name}")
        for name in fields:
            setattr(self, name, options.get(name, getattr(self, name)))


class TrainConfig(Config):
    model = "light"
    dataset = "cifar10"
    data_root = "data"
    channels = 3
    batch_size = 8
    steps = None
    epochs = 30
    lr = None
    warmup_steps = None
    workers = 0
    max_images = 0
    output = "runs/cifar_light"
    resume = None
    save_every = 1000
    log_every = 100
    device = "auto"
    amp = True
    flip = True
    progress = True
    resolution = 32
    base_channels = None


class InferenceConfig(Config):
    checkpoint = "runs/cifar_light/latest.pt"
    samples = 8
    batch_size = 2
    output = "runs/cifar_light/samples.png"
    device = "auto"
    weights = "ema"
    progress = True
    sampler = "ddpm"
    sampling_steps = None
    eta = 0.0
