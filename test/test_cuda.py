import unittest

import torch

from diffusionpack.backbone import DDPMUNet
from diffusionpack.diffusion import p_sample
from diffusionpack.main.train import init_training, train_step


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class TestCUDA(unittest.TestCase):
    def test_forward_backward_ema_and_sampling(self):
        device = torch.device("cuda:0")
        model = DDPMUNet(
            resolution=32, in_channels=1, base_channels=32,
            channel_mult=(1, 2, 2, 2), num_res_blocks=1,
            attention_resolutions=(16,), dropout=0,
        ).to(device)
        optimizer, scheduler, ema_model = init_training(model)
        x = torch.randn(2, 1, 32, 32, device=device)
        result = model(x, torch.tensor([0, 999]))
        self.assertTrue(result.is_cuda)
        self.assertEqual(result.shape, x.shape)
        original = [p.detach().clone() for p in model.parameters()]
        loss_first = train_step(model, x, optimizer, scheduler, ema_model)
        loss_second = train_step(model, x, optimizer, scheduler, ema_model)
        self.assertTrue(torch.isfinite(loss_first))
        self.assertTrue(torch.isfinite(loss_second))
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(model.parameters(), original)))
        output = p_sample(ema_model, torch.randn_like(x), torch.tensor([0, 1]))
        self.assertTrue(output.is_cuda)
        self.assertTrue(torch.isfinite(output).all())
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
