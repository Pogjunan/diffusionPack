import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from diffusionpack.config import TrainConfig, InferenceConfig
from diffusionpack.main.train import run_training
from diffusionpack.main.inference import run_inference
from diffusionpack import diffusion


class TestPythonLaunch(unittest.TestCase):
    def test_config_entries(self):
        train = TrainConfig()
        sample = InferenceConfig()
        self.assertEqual(train.model, "light")
        self.assertEqual(train.epochs, 30)
        self.assertTrue(train.progress)
        self.assertEqual(sample.weights, "ema")

    def test_python_launch_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "latest.pt"
            args = TrainConfig(
                dataset="fake", max_images=2, batch_size=2,
                steps=1, epochs=None, lr=0.0002,
                save_every=1, log_every=1,
                device="cpu", output=folder, progress=False,
            )
            run_training(args)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(state["step"], 1)
            self.assertEqual(state["training"]["lr"], 0.0002)
            args.resume = str(checkpoint)
            args.steps = 2
            run_training(args)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(state["step"], 2)

    def test_rejects_both_steps_and_epochs(self):
        with self.assertRaises(ValueError):
            run_training(TrainConfig(steps=50, epochs=30))

    def test_sampling_progress_option(self):
        class Zero(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        with patch.object(diffusion, "num_diff_steps", 2):
            result = diffusion.p_sample_loop(
                Zero(), (1, 1, 4, 4), device="cpu", progress=True
            )
        self.assertEqual(result.shape, (1, 1, 4, 4))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
