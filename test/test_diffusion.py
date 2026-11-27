"""Small DDPM tests. No real dataset or GPU required."""
import math
import unittest
import numpy as np
import torch
from diffusionpack import diffusion


class TestDDPM(unittest.TestCase):
    def test_schedule(self):
        expected = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
        np.testing.assert_allclose(diffusion.betas.numpy(), expected, rtol=1e-15, atol=1e-18)
        self.assertEqual(tuple(diffusion.alpha_bars.shape), (1000,))
        self.assertAlmostEqual(diffusion.alpha_bars[0].item(), 0.9999, places=12)
        self.assertAlmostEqual(diffusion.alpha_bars[-1].item(), 4.0358297654e-5, places=12)

    def test_q_sample_fixed_noise(self):
        x_0 = torch.tensor([[[[1.0, 0.3]]], [[[0.2, -0.7]]]], dtype=torch.float64)
        eps = torch.tensor([[[[-0.4, 0.1]]], [[[0.5, -0.3]]]], dtype=torch.float64)
        steps = torch.tensor([0, 2], dtype=torch.long)
        observed = diffusion.q_sample(x_0, steps, eps)
        beta = [0.0001 + j * (0.02 - 0.0001) / 999 for j in range(3)]
        abar = [1 - beta[0], math.prod(1 - value for value in beta)]
        expected = torch.empty_like(x_0)
        for j in range(2):
            expected[j] = math.sqrt(abar[j]) * x_0[j] + math.sqrt(1 - abar[j]) * eps[j]
        torch.testing.assert_close(observed, expected, rtol=0, atol=3e-15)
        self.assertFalse(torch.allclose(observed, torch.stack([expected[1], expected[0]])))

    def test_posterior_and_reverse_mean(self):
        x_0 = torch.tensor([[[[0.2]]], [[[0.7]]]], dtype=torch.float64)
        eps = torch.tensor([[[[-0.8]]], [[[1.2]]]], dtype=torch.float64)
        steps = torch.tensor([1, 500], dtype=torch.long)
        xt = diffusion.q_sample(x_0, steps, eps)
        mu = diffusion.posterior_mean(x_0, xt, steps)
        torch.testing.assert_close(mu, diffusion.reverse_mean(xt, steps, eps), rtol=0, atol=1e-12)
        torch.testing.assert_close(x_0, diffusion.predict_x0(xt, steps, eps), rtol=0, atol=1e-12)
        self.assertEqual(diffusion.posterior_var[0].item(), 0)
        torch.testing.assert_close(diffusion.reverse_var, diffusion.betas)

    def test_wrong_noise_shape_fails(self):
        with self.assertRaises(ValueError):
            diffusion.q_sample(torch.zeros(2, 1, 2, 2),
                               torch.tensor([0, 1]), torch.zeros(2, 1, 2, 1))

    def test_gradient(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.1))
            def forward(self, x, t):
                return self.weight * x
        model = ToyModel()
        loss = diffusion.ddpm_loss(model, torch.randn(3, 1, 4, 4))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.weight.grad))

    def test_unet_shape(self):
        from diffusionpack.backbone import DDPMUNet
        model = DDPMUNet(resolution=32, in_channels=1, base_channels=32,
                         channel_mult=(1, 2, 2, 2), num_res_blocks=1, dropout=0)
        image = torch.randn(1, 1, 32, 32)
        output = model(image, torch.tensor([5], dtype=torch.long))
        self.assertEqual(tuple(output.shape), tuple(image.shape))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
