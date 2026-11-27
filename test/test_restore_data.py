import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from diffusionpack.main.restore_data import CifarJpeg, ConditionImages, PairedImages


class TestRestoreData(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def write_image(self, name, color, size=(6, 4)):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path)
        return path

    def test_pairs_and_condition_only_input(self):
        path = self.write_image("source/nested/sample.v1.jpg", "black")
        self.write_image("target/nested/sample.v1.png", "white")
        condition, clean = PairedImages(self.root / "source", self.root / "target", 8, 1, 3)[0]
        torch.testing.assert_close(condition, torch.full((1, 8, 8), -1.0))
        torch.testing.assert_close(clean, torch.ones(3, 8, 8))
        for location, filename in [(self.root / "source", "nested/sample.v1.png"), (path, "sample.v1.png")]:
            image, name = ConditionImages(location, 8, 1)[0]
            self.assertEqual(Path(name), Path(filename))
            torch.testing.assert_close(image, condition)

    def test_ambiguous_missing_and_unaligned_pairs(self):
        source = self.root / "source"
        targets = self.root / "target"
        self.write_image("source/a.jpg", "black")
        self.write_image("target/a.png", "white", size=(8, 4))
        with self.assertRaisesRegex(ValueError, "sizes differ"):
            PairedImages(source, targets, 4)[0]
        self.write_image("target/b.png", "white")
        with self.assertRaisesRegex(ValueError, "Missing image pair"):
            PairedImages(source, targets, 4)
        self.write_image("source/a.png", "white")
        with self.assertRaisesRegex(ValueError, "Multiple images"):
            PairedImages(source, targets, 4)

    def test_cifar_jpeg_quality_changes_only_condition(self):
        pixels = np.random.default_rng(12).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        rows = [(Image.fromarray(pixels), 0)]
        with patch("torchvision.datasets.CIFAR10", return_value=rows) as dataset:
            low, clean = CifarJpeg(self.root, train=False, quality=5, download=False)[0]
            high, same_clean = CifarJpeg(self.root, train=False, quality=30, download=False)[0]
            dataset.assert_called_with(self.root, train=False, download=False)
            with self.assertRaisesRegex(ValueError, "JPEG quality"):
                CifarJpeg(self.root, quality=0, download=False)
        reference = torch.tensor(pixels).permute(2, 0, 1).float() / 127.5 - 1
        torch.testing.assert_close(clean, reference)
        torch.testing.assert_close(clean, same_clean)
        self.assertEqual(low.shape, high.shape)
        self.assertFalse(torch.equal(low, clean))
        self.assertFalse(torch.equal(low, high))
