"""Bounded synthetic audit. Run from the repository with python -m scripts.audit_modes."""
import argparse
import csv
import gc
import json
import platform
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusionpack.backbone import DDPMUNet
from diffusionpack.backbone.prediction_unet import PredictionUNet
from diffusionpack.backbone.restore_unet import RestoreUNet
from diffusionpack.diffusion import ddpm_loss
from diffusionpack.main.inference_prediction import infer_prediction
from diffusionpack.main.inference_restore import infer_restore
from diffusionpack.main.train import model_settings
from diffusionpack.main.train_prediction import train_prediction
from diffusionpack.main.train_restore import train_restore
from diffusionpack.prediction import prediction_loss
from diffusionpack.prediction_config import PredictionInferenceConfig, PredictionTrainConfig
from diffusionpack.restore import restore_loss
from diffusionpack.restore_config import RestoreInferenceConfig, RestoreTrainConfig


def audit(device):
    torch.manual_seed(42)
    result = {"synthetic_only": True, "device": str(device), "torch": torch.__version__,
              "python": platform.python_version(), "cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "architectures": [], "pipelines": {}}
    for preset in ("light", "original"):
        for task in ("unconditional", "restoration", "prediction"):
            settings = model_settings(preset)
            if task == "unconditional":
                model = DDPMUNet(**settings).to(device)
                target = torch.randn(1, 3, 32, 32, device=device)
                loss = ddpm_loss(model, target)
            else:
                del settings["in_channels"]
                if task == "restoration":
                    model = RestoreUNet(target_channels=3, condition_channels=3, **settings).to(device)
                    target = torch.randn(1, 3, 32, 32, device=device)
                    condition = torch.randn_like(target)
                    loss = restore_loss(model, target, condition)
                else:
                    model = PredictionUNet(history_steps=5, map_channels=1, **settings).to(device)
                    target = torch.randn(1, 1, 32, 32, device=device)
                    condition = torch.randn(1, 5, 32, 32, device=device)
                    loss = prediction_loss(model, target, condition)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if task != "unconditional":
                assert model.conv_in.weight.grad[:, model.target_channels:].abs().sum() > 0
            result["architectures"].append({"task": task, "preset": preset,
                "parameters": sum(p.numel() for p in model.parameters()),
                "input_channels": model.conv_in.in_channels,
                "output_channels": model.conv_out.out_channels,
                "finite_loss_and_gradients": True})
            del model, loss
            gc.collect()

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for side in ("condition", "target"):
            (root / side).mkdir()
        rng = np.random.default_rng(12)
        for i in range(2):
            clean = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            degraded = (clean // 32 * 32).astype(np.uint8)
            Image.fromarray(clean).save(root / 'target' / f'{i}.png')
            Image.fromarray(degraded).save(root / 'condition' / f'{i}.png')
        cfg = RestoreTrainConfig(dataset="paired", condition_dir=root/'condition', target_dir=root/'target',
            steps=2, epochs=None, warmup_steps=0, batch_size=2, save_every=1,
            output=root/'restore_run', device=str(device), progress=False)
        cfg.resume = train_restore(cfg)
        cfg.steps = 3
        checkpoint = train_restore(cfg)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['step'] == saved['scheduler']['last_epoch'] == 3
        assert bool(saved['scaler']) == (device.type == 'cuda')
        written = infer_restore(RestoreInferenceConfig(checkpoint=checkpoint, condition_path=root/'condition',
            output=root/'restored', samples=1, device=str(device), progress=False))
        assert written == 1
        with Image.open(root/'restored/0.png') as image:
            assert image.size == (32, 32)
            image.verify()
        result['pipelines']['restoration'] = {'train_resume_steps': 3, 'sampling_steps': 1000, 'png_saved': True}

        manifest = root/'maps.csv'
        with manifest.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['timestamp', 'path', 'available_at'])
            for i in range(10):
                stamp = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=30*i)
                np.save(root/f'{i}.npy', rng.uniform(0, 1000, (32, 32)).astype(np.float32))
                writer.writerow([stamp.isoformat(), f'{i}.npy', (stamp+timedelta(minutes=10)).isoformat()])
        cfg = PredictionTrainConfig(manifest=str(manifest), value_min=0, value_max=1000,
            history_lag_minutes=30, steps=2, epochs=None, warmup_steps=0, batch_size=2,
            output=str(root/'predict_run'), device=str(device), progress=False)
        cfg.resume = str(train_prediction(cfg))
        cfg.steps = 3
        checkpoint = train_prediction(cfg)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['step'] == saved['scheduler']['last_epoch'] == 3
        assert bool(saved['scaler']) == (device.type == 'cuda')
        infer = PredictionInferenceConfig(checkpoint=str(checkpoint), manifest=str(manifest),
            output=str(root/'forecasts'), samples=2, device=str(device), progress=False)
        first = infer_prediction(infer)
        second = infer_prediction(infer)
        assert {r[2] for r in first}.isdisjoint(r[2] for r in second)
        for a, b in zip(first, second):
            x = np.load(root/'forecasts'/a[2]); y = np.load(root/'forecasts'/b[2])
            assert x.shape == (1, 32, 32) and np.isfinite(x).all()
            assert (x >= 0).all() and (x <= 1000).all()
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-5)
        assert not np.array_equal(np.load(root/'forecasts'/first[0][2]), np.load(root/'forecasts'/first[1][2]))
        result['pipelines']['prediction'] = {'train_resume_steps': 3, 'sampling_steps': 1000,
            'samples_per_run': 2, 'runs': 2, 'no_overwrite': True,
            'same_seed_repeat_matches': True, 'ensemble_members_differ': True,
            'observation_delay_minutes': 10, 'history_lag_minutes': 30}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='docs/validation/modes-smoke.json')
    args = parser.parse_args()
    result = audit(torch.device(args.device))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
