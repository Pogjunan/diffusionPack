import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from diffusionpack import diffusion
from diffusionpack.backbone import DDPMUNet
from diffusionpack.main.inference import main as inference_main
from diffusionpack.main.train import (
    FolderImages, choose_device, get_dataset, init_training,
    model_settings, save_checkpoint, train_step,
)


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, t):
        return self.weight * x


class TestPipeline(unittest.TestCase):
    def test_light_preserves_ddpm_steps(self):
        self.assertEqual(diffusion.num_diff_steps, 1000)
        self.assertEqual(model_settings("light")["base_channels"], 64)
        self.assertEqual(model_settings("light")["num_res_blocks"], 1)
        self.assertEqual(model_settings("original")["base_channels"], 128)
        self.assertEqual(model_settings("original")["num_res_blocks"], 2)
        with torch.device("meta"):
            light = DDPMUNet(**model_settings("light"))
            original = DDPMUNet(**model_settings("original"))
        self.assertLess(sum(p.numel() for p in light.parameters()),
                        sum(p.numel() for p in original.parameters()))
        self.assertEqual(sum(m.__class__.__name__ == "AttnBlock" for m in light.modules()), 4)
        self.assertEqual(sum(m.__class__.__name__ == "AttnBlock" for m in original.modules()), 6)

    def test_folder_images_without_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            nested = Path(folder) / "nested"
            nested.mkdir()
            Image.new("RGB", (64, 64), (180, 50, 10)).save(nested / "image.png")
            transform = transforms.Compose([
                transforms.Resize(32), transforms.CenterCrop(32),
                transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ])
            dataset = FolderImages(folder, transform)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].shape, (3, 32, 32))
            self.assertTrue(torch.all(dataset[0] <= 1))
            self.assertTrue(torch.all(dataset[0] >= -1))

    def test_synthetic_dataset(self):
        args = argparse.Namespace(dataset="fake", data_root="unused", flip=True, max_images=5)
        dataset = get_dataset(args, channels=3)
        self.assertEqual(len(dataset), 5)
        image, label = dataset[0]
        self.assertEqual(image.shape, (3, 32, 32))
        self.assertTrue(torch.all(image >= -1))
        self.assertTrue(torch.all(image <= 1))

    def test_warmup_and_ema_cpu(self):
        model = TinyDenoiser()
        optimizer, scheduler, ema_model = init_training(model)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0)
        original = model.weight.detach().clone()
        images = torch.randn(4, 1, 2, 2)
        disabled_scaler = torch.amp.GradScaler("cuda", enabled=False)
        loss1 = train_step(model, images, optimizer, scheduler, ema_model, scaler=disabled_scaler)
        self.assertTrue(torch.isfinite(loss1))
        torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
        loss2 = train_step(model, images, optimizer, scheduler, ema_model, scaler=disabled_scaler)
        self.assertTrue(torch.isfinite(loss2))
        self.assertNotEqual(model.weight.item(), original.item())
        self.assertTrue(torch.isfinite(ema_model.weight))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 8e-8)

    def test_save_and_restore(self):
        model = TinyDenoiser()
        optimizer, scheduler, ema_model = init_training(model)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "latest.pt"
            config = model_settings("light")
            save_checkpoint(path, 9, model, ema_model, optimizer, scheduler, scaler, config, "fake")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["step"], 9)
            self.assertEqual(checkpoint["model_config"], config)
            self.assertTrue(torch.equal(checkpoint["model"]["weight"], model.weight.detach()))
            self.assertIn("optimizer", checkpoint)
            self.assertIn("scheduler", checkpoint)
            self.assertIn("scaler", checkpoint)

    def test_inference_creates_png(self):
        with tempfile.TemporaryDirectory() as folder:
            config = model_settings("light")
            config["base_channels"] = 32
            model = DDPMUNet(**config).eval()
            checkpoint = Path(folder) / "latest.pt"
            image = Path(folder) / "sample.png"
            torch.save({"model_config": config, "model": model.state_dict(),
                        "ema": model.state_dict()}, checkpoint)
            argv = ["inference", "--checkpoint", str(checkpoint), "--samples", "2",
                    "--batch-size", "2", "--device", "cpu", "--output", str(image)]
            with patch.object(diffusion, "num_diff_steps", 2), patch.object(sys, "argv", argv):
                inference_main()
            self.assertTrue(image.exists())
            with Image.open(image) as output:
                self.assertGreater(output.width, 32)
                self.assertEqual(output.height, 32 + 4)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestLightCUDA(unittest.TestCase):
    def test_amp_training_ema_and_sampling(self):
        device = torch.device("cuda:0")
        config = model_settings("light", channels=3)
        model = DDPMUNet(**config).to(device)
        optimizer, scheduler, ema_model = init_training(model)
        scaler = torch.amp.GradScaler("cuda")
        images = torch.randn(2, 3, 32, 32, device=device)
        first = train_step(model, images, optimizer, scheduler, ema_model, scaler=scaler)
        second = train_step(model, images, optimizer, scheduler, ema_model, scaler=scaler)
        self.assertTrue(torch.isfinite(first))
        self.assertTrue(torch.isfinite(second))
        ema_model.eval()
        output = diffusion.p_sample(ema_model, torch.randn_like(images), torch.tensor([0, 1]))
        self.assertTrue(output.is_cuda)
        self.assertTrue(torch.isfinite(output).all())
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    unittest.main()
