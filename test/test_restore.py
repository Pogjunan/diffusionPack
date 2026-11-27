import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from diffusionpack.backbone.restore_unet import RestoreUNet
from diffusionpack.config import num_diff_steps
from diffusionpack.diffusion import alpha_bars
from diffusionpack.main.inference_restore import infer_restore
from diffusionpack.main.train_restore import train_restore
from diffusionpack.restore_config import RestoreInferenceConfig, RestoreTrainConfig
from diffusionpack.restore import restore_loss, restore_sample


class TestRestore(unittest.TestCase):
    def test_noise_and_loss_belong_to_target(self):
        target = torch.linspace(-1, 1, 24).reshape(2, 3, 2, 2)
        source = torch.linspace(1, -1, 8).reshape(2, 1, 2, 2)
        torch.manual_seed(41)
        expected_t = torch.randint(0, num_diff_steps, (2,))
        noise = torch.randn_like(target)
        alpha = alpha_bars[expected_t].float().reshape(2, 1, 1, 1)
        expected_input = alpha.sqrt() * target + (1 - alpha).sqrt() * noise

        def denoiser(noisy_target, t, condition):
            torch.testing.assert_close(t, expected_t)
            torch.testing.assert_close(noisy_target, expected_input)
            torch.testing.assert_close(condition, source)
            return condition.expand_as(noisy_target)

        torch.manual_seed(41)
        loss = restore_loss(denoiser, target, source)
        torch.testing.assert_close(loss, (source - noise).square().mean())

    def test_update_uses_condition_channels(self):
        torch.manual_seed(17)
        model = RestoreUNet(
            target_channels=1, condition_channels=3, resolution=4, base_channels=32,
            channel_mult=(1,), num_res_blocks=1, attention_resolutions=(), dropout=0,
        )
        target = torch.randn(2, 1, 4, 4)
        condition = torch.randn(2, 3, 4, 4)
        self.assertEqual(model.conv_in.in_channels, 4)
        self.assertEqual(model.conv_out.out_channels, 1)
        starting_weight = model.conv_out.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
        loss = restore_loss(model, target, condition)
        loss.backward()
        condition_gradient = model.conv_in.weight.grad[:, model.target_channels:]
        self.assertTrue(torch.isfinite(condition_gradient).all())
        self.assertGreater(condition_gradient.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(model.conv_out.weight, starting_weight))

    def test_full_sampling_keeps_condition_fixed(self):
        source = torch.linspace(-0.8, 0.8, 12, dtype=torch.float64).reshape(1, 3, 2, 2)
        original = source.clone()
        steps = []
        torch.manual_seed(73)
        initial_noise = torch.randn((1, 1, 2, 2), dtype=source.dtype)

        def oracle(noisy_target, t, condition):
            if not steps:
                torch.testing.assert_close(noisy_target, initial_noise)
            torch.testing.assert_close(condition, original)
            steps.append(t.item())
            alpha = alpha_bars[t].reshape(1, 1, 1, 1)
            return (noisy_target - alpha.sqrt() * condition[:, :1]) / (1 - alpha).sqrt()

        oracle.target_channels = 1
        torch.manual_seed(73)
        restored = restore_sample(oracle, source)
        self.assertEqual(steps, list(reversed(range(num_diff_steps))))
        torch.testing.assert_close(restored, original[:, :1])
        torch.testing.assert_close(source, original)


    def test_paired_training_resume_and_condition_only_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for side, mode in [("source", "RGB"), ("target", "L")]:
                (root / side).mkdir()
                Image.new(mode, (8, 8), "white").save(root / side / "sample.png")
            config = RestoreTrainConfig(
                dataset="paired", condition_dir=root / "source", target_dir=root / "target",
                condition_channels=3, target_channels=1, resolution=8, batch_size=1,
                steps=1, epochs=None, warmup_steps=0, output=root / "run", device="cpu",
                progress=False,
            )
            small = dict(in_channels=3, base_channels=32, channel_mult=(1,), num_res_blocks=1,
                         attention_resolutions=(), dropout=0)
            with patch("diffusionpack.main.train_restore.model_settings", side_effect=lambda preset: small.copy()):
                config.resume = train_restore(config)
                config.steps = 2
                checkpoint = train_restore(config)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["step"], 2)
            sample = RestoreInferenceConfig(
                checkpoint=checkpoint, condition_path=root / "source", output=root / "out",
                device="cpu", progress=False,
            )
            self.assertEqual(infer_restore(sample), 1)
            with Image.open(root / "out" / "sample.png") as image:
                self.assertEqual((image.size, image.mode), ((8, 8), "L"))


if __name__ == "__main__":
    unittest.main()
