import math
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from diffusionpack.backbone.ddpm_unet import DDPMUNet, ResBlock, AttnBlock, Upsample, Downsample, timestep_embedding, init_module


class TestUNet(unittest.TestCase):
    def test_original_and_512_layouts(self):
        settings = [
            (32, 3, (1, 2, 2, 2), 35746307, 22, 4),
            (256, 3, (1, 1, 2, 2, 4, 4), 113673219, 32, 8),
            (512, 1, (1, 1, 1, 2, 2, 4, 4), 116312321, 37, 8),
        ]
        for size, channels, mult, params, resblocks, middle_size in settings:
            with self.subTest(resolution=size), torch.device("meta"):
                model = DDPMUNet(
                    resolution=size, in_channels=channels, base_channels=128,
                    channel_mult=mult, num_res_blocks=2,
                    attention_resolutions=(16,), dropout=0,
                )
                self.assertEqual(sum(p.numel() for p in model.parameters()), params)
                self.assertEqual(sum(isinstance(m, ResBlock) for m in model.modules()), resblocks)
                self.assertEqual(sum(isinstance(m, AttnBlock) for m in model.modules()), 6)
                self.assertEqual(sum(isinstance(s["attn"], AttnBlock) for s in model.down if "attn" in s), 2)
                self.assertEqual(sum(isinstance(s["attn"], AttnBlock) for s in model.up if "attn" in s), 3)
                positions = []
                handles = [
                    m.register_forward_hook(lambda module, inputs, output: positions.append(inputs[0].shape[-1]))
                    for m in model.modules() if isinstance(m, AttnBlock)
                ]
                x = torch.empty(1, channels, size, size, device="meta")
                result = model(x, torch.tensor([250]))
                for handle in handles:
                    handle.remove()
                self.assertEqual(result.shape, x.shape)
                self.assertEqual(sorted(positions), sorted([16] * 5 + [middle_size]))

    def test_512_resolution_only_is_not_original_layout(self):
        with torch.device("meta"):
            model = DDPMUNet(resolution=512, in_channels=1)
        self.assertEqual(sum(isinstance(m, AttnBlock) for m in model.modules()), 1)
        self.assertEqual(model.resolution // (2 ** (len(model.channel_mult) - 1)), 64)

    def test_timestep_embedding(self):
        emb = timestep_embedding(torch.tensor([0, 100]), 128)
        self.assertEqual(emb.shape, (2, 128))
        torch.testing.assert_close(emb[0, :64], torch.zeros(64), rtol=0, atol=0)
        torch.testing.assert_close(emb[0, 64:], torch.ones(64), rtol=0, atol=0)

    def test_shortcut_linear_matches_1x1_convolution(self):
        block = ResBlock(32, 64, 128, dropout=0.0).double()
        self.assertIsInstance(block.shortcut, nn.Linear)
        x = torch.randn(2, 32, 3, 3, dtype=torch.float64)
        linear = block.shortcut(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        conv = F.conv2d(x, block.shortcut.weight[:, :, None, None], block.shortcut.bias)
        torch.testing.assert_close(linear, conv, rtol=0, atol=2e-15)
        self.assertIsInstance(ResBlock(32, 32, 128, dropout=0).shortcut, nn.Identity)

    def test_attention_matches_einsum(self):
        attn = AttnBlock(32).eval()
        x = torch.randn(2, 32, 3, 3)
        actual = attn(x)
        h = attn.norm(x)
        q = attn.q(h).flatten(2).transpose(1, 2)
        k = attn.k(h).flatten(2).transpose(1, 2)
        v = attn.v(h).flatten(2).transpose(1, 2)
        weights = torch.softmax(torch.einsum("bic,bjc->bij", q, k) / math.sqrt(32), dim=-1)
        combined = torch.einsum("bij,bjc->bic", weights, v)
        expected = x + attn.proj_out(combined.transpose(1, 2).reshape_as(x))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_nearest_upsampling(self):
        layer = Upsample(32)
        layer.conv = nn.Identity()
        x = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2).repeat(1, 32, 1, 1)
        actual = layer(x)
        expected = x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_downsample_padding(self):
        layer = Downsample(32)
        with torch.no_grad():
            layer.down.weight.zero_()
            layer.down.bias.zero_()
            layer.down.weight[0, 0, 0, 0] = 1
        x = torch.zeros(1, 32, 4, 4)
        x[0, 0] = torch.arange(16).reshape(4, 4)
        y = layer(x)
        torch.testing.assert_close(y[0, 0], x[0, 0, ::2, ::2], rtol=0, atol=0)
        self.assertEqual(y.shape, (1, 32, 2, 2))

    def test_initialization_scale(self):
        module = init_module(nn.Linear(32, 64), scale=1)
        bound = math.sqrt(3 / ((32 + 64) / 2))
        self.assertLessEqual(module.weight.abs().max().item(), bound)
        self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
        module_small = init_module(nn.Linear(32, 64), scale=0)
        self.assertLessEqual(module_small.weight.abs().max().item(), bound * 1e-5)

    def test_small_cpu_forward_and_backward(self):
        torch.manual_seed(3)
        model = DDPMUNet(
            resolution=32, in_channels=1, base_channels=32,
            channel_mult=(1, 2, 2, 2), num_res_blocks=1,
            attention_resolutions=(16,), dropout=0,
        )
        x = torch.randn(1, 1, 32, 32)
        result = model(x, torch.tensor([250]))
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.isfinite(result).all())
        result.square().mean().backward()
        grads = [p.grad for p in model.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
