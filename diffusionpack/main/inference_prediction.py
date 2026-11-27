from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ..backbone.prediction_unet import PredictionUNet
from ..scrd.unet import SCRDUNet
from ..prediction import prediction_sample
from ..scrd.prediction import scrd_prediction_sample
from .prediction_data import ForecastDataset, inverse_scale
from .train import choose_device, load_checkpoint


@torch.inference_mode()
def infer_prediction(config):
    if min(config.max_origins, config.samples) < 1 or config.weights not in ("ema", "model"):
        raise ValueError("Choose positive max_origins and samples, and ema or model weights")
    device = choose_device(config.device)
    saved = load_checkpoint(config.checkpoint)
    training = saved.get("training", {})
    if training.get("task") != "prediction":
        raise ValueError("Use a prediction checkpoint")
    if training.get("backbone", "ddpm") == "scrd":
        expected = dict(noise_conditioning="scrd_timestep", prediction_target="residual", reverse_variance="beta")
        if any(training.get(key) != value for key, value in expected.items()):
            raise ValueError("SCRD checkpoint must use integer time, residual targets and core beta variance")
        settings = saved["model_config"]
        model = SCRDUNet(**settings).to(device)
        sample_function = scrd_prediction_sample
    else:
        if training.get("backbone", "ddpm") != "ddpm":
            raise ValueError("Unknown prediction backbone in checkpoint")
        if training.get("noise_conditioning") != "gamma_adagn":
            raise ValueError("Retrain old conditional checkpoints for gamma conditioning and AdaGN")
        settings = saved["model_config"]
        model = PredictionUNet(**settings).to(device)
        sample_function = prediction_sample
    model.load_state_dict(saved[config.weights])
    model.eval()
    source = ForecastDataset(
        config.csv_path,
        history_steps=settings["history_steps"],
        future_steps=settings["future_steps"],
        map_channels=settings["map_channels"],
        resolution=settings["resolution"],
        cadence_minutes=training["cadence_minutes"],
        lead_minutes=training["lead_minutes"],
        history_lag_minutes=training.get("history_lag_minutes", 0),
        value_min=training["value_min"],
        value_max=training["value_max"],
        require_target=False,
        origin_time=config.origin_time,
        target_csv=config.target_csv,
    )
    selected = range(max(0, len(source) - config.max_origins), len(source))
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with TemporaryDirectory(prefix=".forecast-", dir=output) as temporary:
        staging = Path(temporary)
        run = output / staging.name.removeprefix(".")
        for index in selected:
            history, _, origin, valid_times = source[index]
            history = history.unsqueeze(0).to(device)
            stamp = origin.removesuffix("+00:00").replace(":", "") + "Z"
            for sample in range(1, config.samples + 1):
                predicted = sample_function(model, history, progress=config.progress)
                grid = inverse_scale(predicted[0].float().cpu(), source.value_min, source.value_max)
                if not torch.isfinite(grid).all():
                    raise ValueError(f"Non-finite forecast for {origin}, sample {sample}")
                # Split the joint forecast into F maps, keeping the same sample index.
                for k, (frame, valid) in enumerate(zip(grid.split(settings["map_channels"]), valid_times, strict=True)):
                    lead = training["lead_minutes"] + k * training["cadence_minutes"]
                    filename = f"forecast_{stamp}_{lead}min_sample_{sample:03d}.npy"
                    np.save(staging / filename, frame.numpy())
                    records.append((origin, valid, str(Path(run.name) / filename), training["units"]))
        staging.rename(run)
    return records
