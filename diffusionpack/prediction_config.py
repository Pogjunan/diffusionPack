from .config import Config


class PredictionTrainConfig(Config):
    manifest = "data/forecast/train.csv"
    model = "light"
    resolution = 32
    channel_mult = (1, 2, 2, 2)
    history_steps = 5
    map_channels = 1
    cadence_minutes = 30
    lead_minutes = 30
    value_min = None
    value_max = None
    units = "W/m2"
    batch_size = 8
    steps = None
    epochs = 30
    lr = None
    warmup_steps = None
    workers = 0
    output = "runs/prediction"
    resume = None
    save_every = 1000
    log_every = 100
    device = "auto"
    amp = True
    progress = True
    history_lag_minutes = 0


class PredictionInferenceConfig(Config):
    checkpoint = "runs/prediction/latest.pt"
    manifest = "data/forecast/test.csv"
    output = "runs/prediction/forecasts"
    origin_time = None
    max_origins = 1
    device = "auto"
    weights = "ema"
    progress = True
    samples = 1
