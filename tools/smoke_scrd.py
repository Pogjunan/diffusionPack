"""Small synthetic CPU run, with the real four-scale SCRD and all 1000 DDPM steps."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from diffusionpack.main.scrd_runner import diagnose, infer, train
from diffusionpack.scrd_config import SCRDConfig


def main(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / "data"
    data.mkdir()
    axis = np.linspace(-1, 1, 16, dtype=np.float32)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    for split, day in (("train", 1), ("valid", 3), ("test", 5)):
        rows = []
        for index in range(14):
            values = np.exp(-((x - .025 * index) ** 2 + y ** 2) / .2).astype(np.float32)
            path = data / f"{split}_{index:02d}.npy"
            np.save(path, values)
            timestamp = pd.Timestamp(f"2020-01-{day:02d}T00:00:00Z") + pd.Timedelta(minutes=30 * index)
            rows.append(dict(timestamp=timestamp.isoformat(), path=path.name, available_at=timestamp.isoformat()))
        pd.DataFrame(rows).to_csv(data / f"{split}.csv", index=False)
    config = SCRDConfig(train_csv=str(data / "train.csv"), valid_csv=str(data / "valid.csv"),
                        test_csv=str(data / "test.csv"), value_min=0, value_max=1, units="synthetic",
                        resolution=16, steps=3, batch_size=2, device="cpu", cpu_threads=1,
                        validation_every=0, validation_origins=1, samples=1, max_origins=1,
                        save_every=1, log_every=1, progress=False, output=str(output / "train"))
    started = time.perf_counter()
    checkpoint = train(config)
    config.resume, config.steps = str(checkpoint), 4
    train(config)
    config.checkpoint = str(checkpoint)
    config.output = str(output / "diagnose")
    diagnosis = diagnose(config)

    # Only five history files are visible to inference. No future truth is required.
    history = pd.read_csv(data / "test.csv").iloc[:5]
    history.to_csv(data / "history_only.csv", index=False)
    config.test_csv = str(data / "history_only.csv")
    config.output = str(output / "forecasts")
    infer(config)
    paths = sorted(Path(config.output).glob("*.npy"))
    assert len(paths) == 4
    assert all(np.load(path).shape == (1, 16, 16) and np.isfinite(np.load(path)).all() for path in paths)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    result = dict(kind="synthetic CPU execution only", optimizer_steps=saved["step"],
                  history_steps=config.history_steps, future_steps=config.future_steps,
                  resolution=config.resolution, diffusion_steps=config.diffusion_steps,
                  widths=list(config.widths), tcu_depth=config.tcu_depth,
                  forecast_files=len(paths), forecast_shape=[1, 16, 16],
                  diagnosis_levels=diagnosis["t"].tolist(),
                  torch_version=str(torch.__version__), cuda_available=torch.cuda.is_available(),
                  seconds=round(time.perf_counter() - started, 2),
                  quality_validated=False)
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/scrd_smoke")
    main(parser.parse_args().output)
