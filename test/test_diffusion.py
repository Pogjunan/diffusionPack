import math
import unittest
from unittest.mock import patch

import torch

from diffusionpack import diffusion


class TestDiffusion(unittest.TestCase):
    def test_schedule(self):
        self.assertEqual(diffusion.betas.numel(), 1000)
        self.assertAlmostEqual(diffusion.betas[0].item(), 0.0001, places=12)
        self.assertAlmostEqual(diffusion.betas[-1].item(), 0.02, places=12)
        self.assertTrue(torch.all(diffusion.alpha_bars[1:] < diffusion.alpha_bars[:-1]))
        self.assertEqual(diffusion.posterior_var[0].item(), 0.0)
        self.assertTrue(torch.isfinite(diffusion.reverse_log_var).all())
        self.assertAlmostEqual(
            diffusion.reverse_log_var[0].item(),
            math.log(diffusion.posterior_var[1].item()),
            places=12,
        )

    def test_step_value(self):
        schedule = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        t = torch.tensor([2, 0])
        x = torch.tensor([
            [[[1., 2.], [3., 4.]]],
            [[[5., 6.], [7., 8.]]],
        ])
        values = diffusion.step_value(schedule, t, x)
        self.assertEqual(values.shape, (2, 1, 1, 1))
        self.assertEqual(values.dtype, x.dtype)
        torch.testing.assert_close(
            values.reshape(2), torch.tensor([0.3, 0.1]), rtol=0, atol=1e-7
        )
        torch.testing.assert_close(
            values * x,
            torch.tensor([
                [[[0.3, 0.6], [0.9, 1.2]]],
                [[[0.5, 0.6], [0.7, 0.8]]],
            ]),
            rtol=0, atol=2e-7,
        )

    def test_step_value_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            diffusion.step_value(torch.arange(5.), torch.tensor([1]), torch.zeros(2, 1, 2, 2))
        with self.assertRaises(ValueError):
            diffusion.step_value(torch.arange(5.), torch.tensor([1, 2]), torch.zeros(2, 1, 2, 2, dtype=torch.long))
        with self.assertRaises(IndexError):
            diffusion.step_value(torch.arange(5.), torch.tensor([-1, 2]), torch.zeros(2, 1, 2, 2))

    def test_q_sample_known_noise(self):
        x_0 = torch.tensor([[[[0.4, -0.2]]], [[[0.1, 0.7]]]], dtype=torch.float64)
        eps = torch.tensor([[[[-0.5, 0.9]]], [[[0.8, -0.6]]]], dtype=torch.float64)
        t = torch.tensor([0, 2])
        values = diffusion.q_sample(x_0, t, eps)
        beta = [0.0001 + i * (0.02 - 0.0001) / 999 for i in range(3)]
        alpha_bar = [1 - beta[0], math.prod(1 - v for v in beta)]
        expected = torch.stack([
            math.sqrt(alpha_bar[i]) * x_0[i] + math.sqrt(1 - alpha_bar[i]) * eps[i]
            for i in range(2)
        ])
        torch.testing.assert_close(values, expected, rtol=0, atol=3e-15)
        with self.assertRaises(ValueError):
            diffusion.q_sample(x_0, t, eps[:, :, :, :1])

    def test_reverse_mean_and_x0(self):
        x_0 = torch.tensor([[[[-0.4]]], [[[0.7]]], [[[0.2]]]], dtype=torch.float64)
        eps = torch.tensor([[[[0.7]]], [[[-0.6]]], [[[1.1]]]], dtype=torch.float64)
        t = torch.tensor([0, 300, 999])
        x_t = diffusion.q_sample(x_0, t, eps)
        torch.testing.assert_close(diffusion.predict_x0(x_t, t, eps), x_0, rtol=0, atol=4e-13)
        torch.testing.assert_close(
            diffusion.posterior_mean(x_0, x_t, t),
            diffusion.reverse_mean(x_t, t, eps),
            rtol=0, atol=4e-13,
        )
        wrong_eps = torch.zeros_like(eps)
        self.assertFalse(torch.allclose(diffusion.predict_x0(x_t, t, wrong_eps), x_0))

    def test_p_sample_noise_and_final_step(self):
        class FixedNoise(torch.nn.Module):
            def __init__(self, eps):
                super().__init__()
                self.eps = eps

            def forward(self, x, t):
                return self.eps

        x_0 = torch.tensor([[[[0.25]]], [[[-0.4]]]], dtype=torch.float64)
        eps = torch.tensor([[[[-0.3]]], [[[1.2]]]], dtype=torch.float64)
        t = torch.tensor([0, 500])
        x_t = diffusion.q_sample(x_0, t, eps)
        noise = torch.tensor([[[[100.0]]], [[[0.5]]]], dtype=torch.float64)
        actual = diffusion.p_sample(FixedNoise(eps), x_t, t, noise=noise)
        expected = diffusion.posterior_mean(x_0, x_t, t)
        expected[1] += math.sqrt(diffusion.betas[500].item()) * noise[1]
        torch.testing.assert_close(actual, expected, rtol=0, atol=3e-13)
        other = noise.clone()
        other[0] = -1000.0
        actual_other = diffusion.p_sample(FixedNoise(eps), x_t, t, noise=other)
        torch.testing.assert_close(actual[0], actual_other[0], rtol=0, atol=0)

    def test_p_sample_clips_x0(self):
        class ZeroNoise(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        x_t = torch.tensor([[[[2.0]]]], dtype=torch.float64)
        t = torch.tensor([0])
        clipped = diffusion.p_sample(ZeroNoise(), x_t, t, noise=torch.zeros_like(x_t))
        unclipped = diffusion.p_sample(ZeroNoise(), x_t, t, noise=torch.zeros_like(x_t), clip_denoised=False)
        torch.testing.assert_close(clipped, torch.ones_like(clipped), rtol=0, atol=1e-12)
        self.assertGreater(unclipped.item(), 1.5)

    def test_p_sample_loop_state_and_order(self):
        recorded = []

        def fake_sample(model, x_t, t, clip_denoised=True):
            recorded.append(t.tolist())
            return x_t + (t[0].item() + 1)

        torch.manual_seed(87)
        expected = torch.randn(2, 1, 2, 2) + 6
        torch.manual_seed(87)
        with patch.object(diffusion, "num_diff_steps", 3), patch.object(diffusion, "p_sample", side_effect=fake_sample):
            actual = diffusion.p_sample_loop(None, (2, 1, 2, 2), device="cpu")
        self.assertEqual(recorded, [[2, 2], [1, 1], [0, 0]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_ddpm_loss_gradient_and_target(self):
        class ScaleNoise(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

            def forward(self, x, t):
                return self.weight * x

        x_0 = torch.tensor([[[[0.2, -0.3]]], [[[0.8, 0.4]]]], dtype=torch.float64)
        model = ScaleNoise()
        torch.manual_seed(119)
        t = torch.randint(0, diffusion.num_diff_steps, (2,))
        eps = torch.randn_like(x_0)
        expected = ((model(diffusion.q_sample(x_0, t, eps), t) - eps) ** 2).mean()
        torch.manual_seed(119)
        actual = diffusion.ddpm_loss(model, x_0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)
        actual.backward()
        self.assertTrue(torch.isfinite(model.weight.grad))
        self.assertNotEqual(model.weight.grad.item(), 0)


if __name__ == "__main__":
    unittest.main()
