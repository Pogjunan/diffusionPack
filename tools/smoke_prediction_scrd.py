"""Exercise SCRD through the existing Prediction trainer and DDPM sampler."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from diffusionpack.main.inference_prediction import infer_prediction
from diffusionpack.main.train import load_checkpoint
from diffusionpack.main.train_prediction import train_prediction
from diffusionpack.prediction_config import PredictionInferenceConfig, PredictionTrainConfig


def main(output):
    torch.set_num_threads(1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / "data"
    data.mkdir()
    axis = np.linspace(-1, 1, 16, dtype=np.float32)
    y, x = np.meshgrid(axis, axis)
    rows = []
    for index, timestamp in enumerate(pd.date_range("2020-01-01T00:00:00Z", periods=14, freq="30min")):
        name = f"{index:02d}.npy"
        np.save(data / name, np.exp(-((x - .025 * index) ** 2 + y ** 2) / .2).astype(np.float32))
        rows.append(dict(timestamp=timestamp.isoformat(), path=name, available_at=timestamp.isoformat()))
    table = pd.DataFrame(rows)
    table.to_csv(data / "frames.csv", index=False)
    table.iloc[5:][["timestamp"]].to_csv(data / "targets.csv", index=False)
    table.iloc[:5].to_csv(data / "history_only.csv", index=False)
    config = PredictionTrainConfig(
        backbone="scrd", csv_path=str(data / "frames.csv"), target_csv=str(data / "targets.csv"),
        history_steps=5, future_steps=4, resolution=16, value_min=0, value_max=1, units="synthetic",
        steps=2, epochs=None, batch_size=2, warmup_steps=0, device="cpu", amp=False,
        save_every=1, log_every=1, progress=False, output=str(output / "train"),
    )
    started = time.perf_counter()
    checkpoint = train_prediction(config)
    config.steps, config.resume = 3, str(checkpoint)
    train_prediction(config)
    records = infer_prediction(PredictionInferenceConfig(
        checkpoint=str(checkpoint), csv_path=str(data / "history_only.csv"),
        target_csv=str(data / "targets.csv"), origin_time="2020-01-01T02:00:00Z",
        output=str(output / "forecasts"), weights="model", samples=1, device="cpu", progress=False,
    ))
    assert len(records) == 4
    for _, _, name, _ in records:
        array = np.load(output / "forecasts" / name)
        assert array.shape == (1, 16, 16) and np.isfinite(array).all()
    saved = load_checkpoint(checkpoint)
    result = dict(kind="synthetic CPU integration only", entry="train_prediction / infer_prediction",
                  backbone=saved["training"]["backbone"], optimizer_steps=saved["step"],
                  diffusion_steps=saved["ddpm_schedule"][0], reverse_variance="beta",
                  history_steps=5, future_steps=4, resolution=16, widths=list(config.scrd_widths),
                  forecast_files=len(records), forecast_shape=[1, 16, 16],
                  resume="functional; not bit-exact continuation", weights="model",
                  seconds=round(time.perf_counter() - started, 2), quality_validated=False,
                  torch_version=str(torch.__version__), cuda_available=torch.cuda.is_available())
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/prediction_scrd_smoke")
    main(parser.parse_args().output)
