import unittest

import torch
from torch import nn

from diffusionpack.config import learning_rate, warmup_steps, ema_decay
from diffusionpack.main.train import init_training, train_step


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

    def forward(self, x, t):
        return self.weight * x + self.bias


class TestTraining(unittest.TestCase):
    def test_warmup_and_ema_updates(self):
        model = TinyDenoiser()
        optimizer, scheduler, ema_model = init_training(model)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0)
        self.assertFalse(ema_model.training)
        self.assertTrue(all(not p.requires_grad for p in ema_model.parameters()))
        starting_weight = model.weight.detach().clone()
        x = torch.tensor([[[[0.4, -0.2]]], [[[0.1, 0.5]]]], dtype=torch.float64)
        torch.manual_seed(125)
        loss_first = train_step(model, x, optimizer, scheduler, ema_model)
        self.assertTrue(torch.isfinite(loss_first))
        torch.testing.assert_close(model.weight, starting_weight, rtol=0, atol=0)
        torch.testing.assert_close(ema_model.weight, starting_weight, rtol=0, atol=0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], learning_rate / warmup_steps)
        old_ema_weight = ema_model.weight.detach().clone()
        torch.manual_seed(125)
        loss_second = train_step(model, x, optimizer, scheduler, ema_model)
        self.assertTrue(torch.isfinite(loss_second))
        self.assertNotEqual(model.weight.item(), starting_weight.item())
        expected_ema = ema_decay * old_ema_weight + (1 - ema_decay) * model.weight.detach()
        torch.testing.assert_close(ema_model.weight, expected_ema, rtol=0, atol=1e-15)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2 * learning_rate / warmup_steps)
        self.assertEqual(scheduler.lr_lambdas[0](warmup_steps), 1.0)
        self.assertEqual(scheduler.lr_lambdas[0](warmup_steps + 20), 1.0)

    def test_train_step_without_scheduler(self):
        model = TinyDenoiser()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x = torch.randn(2, 1, 3, 3, dtype=torch.float64)
        original = model.weight.detach().clone()
        loss = train_step(model, x, optimizer)
        self.assertTrue(torch.isfinite(loss))
        self.assertNotEqual(model.weight.item(), original.item())


if __name__ == "__main__":
    unittest.main()
