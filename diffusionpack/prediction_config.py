from .config import Config


class PredictionTrainConfig(Config):
    csv_path = "data/forecast/train.csv"
    model = "light"
    resolution = 32
    channel_mult = None
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
    base_channels = None
    future_steps = 1
    backbone = "ddpm"
    target_csv = None
    scrd_widths = (32, 64, 128, 256)
    scrd_window = 8
    scrd_tcu_depth = 3
    scrd_time_dim = 128
    scrd_norm_groups = 32
    scrd_output_norm_groups = 8
    scrd_use_tcu = True


class PredictionInferenceConfig(Config):
    checkpoint = "runs/prediction/latest.pt"
    csv_path = "data/forecast/test.csv"
    output = "runs/prediction/forecasts"
    origin_time = None
    max_origins = 1
    device = "auto"
    weights = "ema"
    progress = True
    samples = 1
    target_csv = None
