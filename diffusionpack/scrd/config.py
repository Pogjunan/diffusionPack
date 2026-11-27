from ..config import Config


class SCRDConfig(Config):
    train_csv = "data/forecast/train.csv"
    valid_csv = "data/forecast/valid.csv"
    test_csv = "data/forecast/test.csv"
    train_targets = None
    valid_targets = None
    test_targets = None
    history_steps = 5
    future_steps = 4
    map_channels = 1
    resolution = 512
    cadence_minutes = 30
    lead_minutes = 30
    history_lag_minutes = 0
    value_min = None
    value_max = None
    units = None

    backbone = "scrd"
    target = "residual"
    noise_conditioning = "timestep"
    time_input = "index"
    diffusion_steps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    reverse_variance = "posterior"

    widths = (32, 64, 128, 256)
    window = 8
    tcu_depth = 3
    time_dim = 128
    norm_groups = 32
    output_norm_groups = 8
    use_tcu = True

    base_channels = 64
    channel_mult = None
    num_res_blocks = 1
    attention_resolutions = (16,)
    dropout = 0.1

    learning_rate = 0.0002
    steps = 250000
    batch_size = 16
    seed = 1
    workers = 0
    device = "auto"
    cpu_threads = 4
    precision = "float32"
    grad_clip = None
    save_every = 1000
    log_every = 100
    validation_every = 10000
    validation_origins = 30
    validation_samples = 1
    output = "runs/scrd"
    resume = None

    checkpoint = "runs/scrd/latest.pt"
    samples = 8
    max_origins = 1
    origin_time = None
    progress = True
