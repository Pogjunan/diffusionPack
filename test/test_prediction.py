import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from diffusionpack import diffusion
from diffusionpack.backbone.prediction_unet import PredictionUNet
from diffusionpack.main.inference_prediction import infer_prediction
from diffusionpack.main.prediction_data import ForecastDataset, inverse_scale, parse_time
from diffusionpack.main.train_prediction import prediction_settings, train_prediction
from diffusionpack.prediction import prediction_loss, prediction_sample
from diffusionpack.prediction_config import PredictionInferenceConfig, PredictionTrainConfig


class TestPrediction(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.manifest = self.root / "train.csv"
        self.start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=9)))
        self.make_manifest()

    def make_manifest(self, *, missing=(), late=(), invalid=None):
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "path", "available_at"])
            for index in range(9):
                if index in missing:
                    continue
                stamp = self.start + timedelta(minutes=30 * index)
                available = stamp + timedelta(minutes=20 if index in late else 0)
                path = self.root / f"{index:02d}.npy"
                grid = np.full((8, 8), index * 100.0, dtype=np.float32)
                if index == invalid:
                    grid[0, 0] = float("nan")
                np.save(path, grid)
                writer.writerow([stamp.isoformat(), path.name, available.isoformat()])

    def dataset(self, **kwargs):
        return ForecastDataset(
            self.manifest, history_steps=3, map_channels=1,
            resolution=8, cadence_minutes=30, lead_minutes=60,
            value_min=0, value_max=1000, **kwargs,
        )

    def test_exact_forecast_origin_and_values(self):
        data = self.dataset()
        history, future, origin, valid = data[0]
        self.assertEqual(history.shape, (3, 8, 8))
        self.assertEqual(future.shape, (1, 8, 8))
        self.assertEqual(origin, "2024-12-31T16:00:00+00:00")
        self.assertEqual(valid, "2024-12-31T17:00:00+00:00")
        torch.testing.assert_close(history[:, 0, 0], torch.tensor([-1.0, -0.8, -0.6]))
        torch.testing.assert_close(future[:, 0, 0], torch.tensor([-0.2]))
        torch.testing.assert_close(inverse_scale(future, 0, 1000), torch.full((1, 8, 8), 400.0))
        for s, hist, y in data.windows:
            self.assertTrue(all(x <= s for x in hist))
            self.assertGreater(y, s)
            self.assertEqual(y - s, timedelta(minutes=60))
            self.assertEqual(hist[-1], s)

    def test_bad_512_channel_configuration_rejected(self):
        with self.assertRaisesRegex(ValueError, "16x16"):
            prediction_settings(PredictionTrainConfig(
                resolution=512, channel_mult=(1, 2, 2, 2),
            ))
        config = PredictionTrainConfig(
            resolution=512, channel_mult=(1, 1, 1, 2, 2, 4, 4),
        )
        self.assertEqual(prediction_settings(config)["channel_mult"], config.channel_mult)

    def test_epsilon_training_matches_independent_reference(self):
        future = torch.tensor([[[[0.3, -0.5]]]], dtype=torch.float64)
        history = torch.tensor([[[[-0.2, 0.8]], [[-0.4, 0.5]]]], dtype=torch.float64)
        traces = []

        def denoiser(noisy, t, condition):
            traces.append((noisy.clone(), t.clone(), condition.clone()))
            return condition[:, :1] * 0.2

        torch.manual_seed(24)
        t = torch.randint(0, 1000, (1,))
        eps = torch.randn_like(future)
        gamma = float(diffusion.alpha_bars[t].item())
        expected_xt = (gamma ** 0.5) * future + ((1 - gamma) ** 0.5) * eps
        expected_loss = ((history[:, :1] * 0.2 - eps) ** 2).mean()
        torch.manual_seed(24)
        observed = prediction_loss(denoiser, future, history)
        torch.testing.assert_close(observed, expected_loss, rtol=0, atol=1e-14)
        torch.testing.assert_close(traces[0][0], expected_xt, rtol=0, atol=1e-14)
        torch.testing.assert_close(traces[0][1], t)
        torch.testing.assert_close(traces[0][2], history, rtol=0, atol=0)

    def test_time_zone_equivalence_and_naive_rejection(self):
        self.assertEqual(parse_time("2025-01-01T00:00:00+09:00"), parse_time("2024-12-31T15:00:00Z"))
        with self.assertRaises(ValueError):
            parse_time("2025-01-01T00:00:00")
        original = self.manifest.read_text()
        self.manifest.write_text(original.replace("2025-01-01T00:00:00+09:00", "2024-12-31T15:00:00Z", 1))
        self.assertEqual(len(self.dataset()), 5)

    def test_missing_grid_not_treated_as_adjacent(self):
        self.make_manifest(missing=(3,))
        data = self.dataset()
        self.assertLess(len(data), 5)
        self.assertGreater(data.rejections["missing_history"], 0)
        self.assertGreater(data.rejections["missing_target"], 0)
        with self.assertRaisesRegex(ValueError, "No valid prediction windows"):
            self.dataset(origin_time=(self.start + timedelta(minutes=120)).isoformat())

    def test_late_published_history_rejected(self):
        self.make_manifest(late=(2,))
        data = self.dataset()
        self.assertGreater(data.rejections["late_history"], 0)
        self.assertNotEqual(data.windows[0][0], parse_time((self.start + timedelta(minutes=60)).isoformat()))

    def test_unavailable_future_still_allows_inference(self):
        origin = (self.start + timedelta(minutes=240)).isoformat()
        with self.assertRaises(ValueError):
            self.dataset(origin_time=origin)
        data = self.dataset(require_target=False, origin_time=origin)
        history, future, seen, target = data[0]
        self.assertEqual(history.shape, (3, 8, 8))
        self.assertEqual(future.numel(), 0)
        self.assertEqual(parse_time(target) - parse_time(seen), timedelta(minutes=60))

    def test_nans_and_scaling_are_not_silently_clipped(self):
        with self.assertRaisesRegex(ValueError, "value_min and value_max"):
            ForecastDataset(self.manifest, history_steps=3, resolution=8, value_min=None, value_max=None)
        self.make_manifest(invalid=2)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.dataset()[0]
        self.make_manifest()
        np.save(self.root / "00.npy", np.full((8, 8), 2000, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Out-of-range"):
            self.dataset()[0]

    def test_duplicate_timestamps_detected(self):
        with self.manifest.open("a") as handle:
            writer = csv.writer(handle)
            writer.writerow([self.start.isoformat(), "00.npy", self.start.isoformat()])
        with self.assertRaisesRegex(ValueError, "Duplicate valid"):
            self.dataset()

    def test_model_channels_and_conditional_gradient(self):
        model = PredictionUNet(
            history_steps=3, map_channels=1, resolution=8, base_channels=32,
            channel_mult=(1,), num_res_blocks=1, attention_resolutions=(), dropout=0,
        )
        self.assertEqual(model.conv_in.in_channels, 4)
        self.assertEqual(model.conv_out.out_channels, 1)
        future = torch.randn(1, 1, 8, 8)
        history = torch.randn(1, 3, 8, 8)
        pred = model(future, torch.tensor([5]), history)
        self.assertEqual(pred.shape, future.shape)
        loss = prediction_loss(model, future, history)
        loss.backward()
        self.assertGreater(model.conv_in.weight.grad[:, 1:].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            model(future, torch.tensor([5]), history[:, :2])

    def test_history_is_constant_at_every_reverse_step(self):
        class Recorder:
            target_channels = 1

            def __init__(self):
                self.seen = []

            def __call__(self, noisy, t, condition):
                torch.testing.assert_close(condition, original)
                self.seen.append(t.item())
                return torch.zeros_like(noisy)

        original = torch.randn(1, 3, 2, 2)
        model = Recorder()
        result = prediction_sample(model, original)
        self.assertEqual(result.shape, (1, 1, 2, 2))
        self.assertEqual(model.seen, list(reversed(range(diffusion.num_diff_steps))))
        self.assertTrue(torch.isfinite(result).all())

    def test_condition_can_change_epsilon_prediction(self):
        class Conditioner:
            target_channels = 1

            def __call__(self, noisy, t, condition):
                return condition[:, :1] * 0.2

        a = torch.zeros(1, 3, 2, 2)
        b = torch.ones(1, 3, 2, 2)
        torch.manual_seed(8)
        first = prediction_sample(Conditioner(), a)
        torch.manual_seed(8)
        second = prediction_sample(Conditioner(), b)
        self.assertFalse(torch.allclose(first, second))

    def test_cpu_train_resume_and_inference(self):
        small = dict(base_channels=32, channel_mult=(1,), num_res_blocks=1,
                     attention_resolutions=(), dropout=0)
        original_model = prediction_settings

        def tiny(config):
            settings = original_model(config)
            settings.update(small)
            return settings

        cfg = PredictionTrainConfig(
            manifest=str(self.manifest), resolution=8, channel_mult=(1,),
            history_steps=3, cadence_minutes=30, lead_minutes=60,
            value_min=0.0, value_max=1000.0, units="W/m2", batch_size=1,
            epochs=None, steps=1, warmup_steps=0, save_every=1, log_every=1,
            output=str(self.root / "run"), device="cpu", progress=False,
        )
        with patch("diffusionpack.main.train_prediction.prediction_settings", side_effect=tiny):
            checkpoint = train_prediction(cfg)
            cfg.resume = str(checkpoint)
            cfg.steps = 2
            checkpoint = train_prediction(cfg)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(saved["step"], 2)
        self.assertEqual(saved["training"]["lead_minutes"], 60)
        self.assertEqual(saved["training"]["value_max"], 1000)
        inference = PredictionInferenceConfig(
            checkpoint=str(checkpoint), manifest=str(self.manifest),
            output=str(self.root / "forecasts"), max_origins=1, device="cpu", progress=False,
        )
        with patch.object(diffusion, "num_diff_steps", 3):
            records = infer_prediction(inference)
        self.assertEqual(len(records), 1)
        produced = np.load(self.root / "forecasts" / records[0][2])
        self.assertEqual(produced.shape, (1, 8, 8))
        self.assertTrue(np.isfinite(produced).all())
        self.assertTrue(((produced >= 0) & (produced <= 1000)).all())
        self.assertEqual(records[0][3], "W/m2")
        self.assertTrue((self.root / "forecasts" / "predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
