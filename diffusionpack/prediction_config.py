from dataclasses import dataclass


@dataclass
class PredictionTrainConfig:
    manifest: str = "data/forecast/train.csv"
    model: str = "light"
    resolution: int = 32
    channel_mult: tuple[int, ...] = (1, 2, 2, 2)
    history_steps: int = 5
    map_channels: int = 1
    cadence_minutes: int = 30
    lead_minutes: int = 30
    value_min: float | None = None
    value_max: float | None = None
    units: str = "W/m2"
    batch_size: int = 8
    steps: int | None = None
    epochs: int | None = 30
    lr: float | None = None
    warmup_steps: int | None = None
    workers: int = 0
    output: str = "runs/prediction"
    resume: str | None = None
    save_every: int = 1000
    log_every: int = 100
    device: str = "auto"
    amp: bool = True
    progress: bool = True
    seed: int = 42
    history_lag_minutes: int = 0


@dataclass
class PredictionInferenceConfig:
    checkpoint: str = "runs/prediction/latest.pt"
    manifest: str = "data/forecast/test.csv"
    output: str = "runs/prediction/forecasts"
    origin_time: str | None = None
    max_origins: int = 1
    device: str = "auto"
    weights: str = "ema"
    progress: bool = True
    seed: int = 42
    samples: int = 1
