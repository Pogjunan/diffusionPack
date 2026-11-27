import csv
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ..backbone.prediction_unet import PredictionUNet
from ..prediction import prediction_sample
from .prediction_data import ForecastDataset, inverse_scale
from .train import choose_device


@torch.inference_mode()
def infer_prediction(config):
    if min(config.max_origins, config.samples) < 1 or config.weights not in ("ema", "model"):
        raise ValueError("Choose positive max_origins and samples, and ema or model weights")
    device = choose_device(config.device)
    digest = hashlib.sha256()
    with Path(config.checkpoint).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
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
        history_lag_minutes=training.get("history_lag_minutes", 0),
        value_min=training["value_min"],
        value_max=training["value_max"],
        require_target=False,
        origin_time=config.origin_time,
    )
    selected = range(max(0, len(source) - config.max_origins), len(source))
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    epoch = training.get("epoch")
    epoch_label = f"epoch_{int(epoch)}_" if epoch is not None else ""
    run_label = f"{epoch_label}step_{saved['step']}_{config.weights}_seed_{config.seed}"
    records = []
    with TemporaryDirectory(dir=output) as temporary:
        staging = Path(temporary)
        run = output / f"{run_label}_{staging.name}"
        with (staging / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["origin_time", "target_time", "path", "units", "last_observation",
                             "sample", "seed", "sample_seed", "weights", "step", "epoch", "checkpoint_sha256"])
            for index in selected:
                history, _, origin, valid = source[index]
                history = history.unsqueeze(0).to(device)
                last_observation = source.windows[index][1][-1].isoformat()
                stamp = origin.removesuffix("+00:00").replace(":", "") + "Z"
                for sample in range(1, config.samples + 1):
                    key = f"{config.seed}/{origin}/{sample}".encode()
                    sample_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
                    torch.manual_seed(sample_seed)
                    predicted = prediction_sample(model, history, progress=config.progress)
                    grid = inverse_scale(predicted[0].cpu(), training["value_min"], training["value_max"])
                    filename = f"forecast_{stamp}_{training['lead_minutes']}min_sample_{sample:03d}.npy"
                    np.save(staging / filename, grid.numpy().astype(np.float32, copy=False))
                    writer.writerow([origin, valid, filename, training["units"], last_observation, sample,
                                     config.seed, sample_seed, config.weights, saved["step"], epoch, digest.hexdigest()])
                    records.append((origin, valid, str(Path(run.name) / filename), training["units"]))
        staging.rename(run)
    print(f"saved={run}", flush=True)
    return records
