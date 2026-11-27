import csv
from pathlib import Path

import numpy as np
import torch

from ..backbone.prediction_unet import PredictionUNet
from ..prediction import prediction_sample
from .prediction_data import ForecastDataset, inverse_scale
from .train import choose_device


@torch.inference_mode()
def infer_prediction(config):
    if config.max_origins < 1 or config.weights not in ("ema", "model"):
        raise ValueError("Choose positive max_origins and ema or model weights")
    torch.manual_seed(config.seed)
    device = choose_device(config.device)
    saved = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
    training = saved.get("training", {})
    if training.get("task") != "prediction":
        raise ValueError("Use a prediction checkpoint")
    settings = saved["model_config"]
    model = PredictionUNet(**settings).to(device)
    model.load_state_dict(saved[config.weights])
    model.eval()
    source = ForecastDataset(
        config.manifest,
        history_steps=settings["history_steps"],
        map_channels=settings["map_channels"],
        resolution=settings["resolution"],
        cadence_minutes=training["cadence_minutes"],
        lead_minutes=training["lead_minutes"],
        value_min=training["value_min"],
        value_max=training["value_max"],
        require_target=False,
        origin_time=config.origin_time,
    )
    selected = list(range(max(0, len(source) - config.max_origins), len(source)))
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index in selected:
        history, _, origin, valid = source[index]
        predicted = prediction_sample(model, history.unsqueeze(0).to(device), progress=config.progress)
        restored = inverse_scale(predicted[0].cpu(), training["value_min"], training["value_max"])
        filename = f"forecast_{origin.replace(':', '').replace('+0000', 'Z').replace('+00:00', 'Z')}_{training['lead_minutes']}min.npy"
        path = output / filename
        np.save(path, restored.numpy().astype(np.float32, copy=False))
        records.append((origin, valid, filename, training["units"]))
    manifest_path = output / "predictions.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["origin_time", "target_time", "path", "units"])
        writer.writerows(records)
    return records
