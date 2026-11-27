from .config import Config


class SCRDConfig(Config):
    # Current SSI task. The paper used 6 -> 6, 128x128, every 10 minutes.
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
    units = None  # Set the actual units of the stored arrays.

    # Matched comparisons: ddpm/future, ddpm/residual, scrd/residual.
    backbone = "scrd"
    target = "residual"
    noise_conditioning = "timestep"  # gamma is available for the DDPM baseline only.
    time_input = "index"  # index: 0..999 (old code); fraction: (t+1)/1000 (Fig. 1).
    diffusion_steps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    reverse_variance = "posterior"  # beta is the diffusionPack baseline choice.

    # SCRD implementation choices; these numbers are not specified in the paper.
    widths = (32, 64, 128, 256)
    window = 8
    tcu_depth = 3
    time_dim = 128
    norm_groups = 32
    output_norm_groups = 8
    use_tcu = True  # False removes the TCU gates, not the entire condition encoder.

    # Existing DDPM U-Net settings for the matched control.
    base_channels = 64
    channel_mult = None
    num_res_blocks = 1
    attention_resolutions = (16,)
    dropout = 0.1

    # Paper: Adam, initial LR 2e-4, 250,000 optimizer updates.
    learning_rate = 0.0002
    steps = 250000
    batch_size = 16
    seed = 1
    workers = 0
    device = "auto"
    cpu_threads = 4
    precision = "float32"  # float32 or bfloat16; start debugging in float32.
    grad_clip = None
    save_every = 1000
    log_every = 100
    validation_every = 10000
    validation_origins = 30
    validation_samples = 1
    output = "runs/scrd"
    resume = None

    # Sampling/evaluation are explicit separate commands.
    checkpoint = "runs/scrd/latest.pt"
    samples = 8
    max_origins = 1  # None evaluates every usable origin.
    origin_time = None
    progress = True
