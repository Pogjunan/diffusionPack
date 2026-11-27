from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def init_module(module: nn.Module, scale: float = 1.0) -> nn.Module:
    out_ch, in_ch, *kernel = module.weight.shape
    kernel_size = math.prod(kernel)
    input_connections = in_ch * kernel_size
    output_connections = out_ch * kernel_size
    # Average connections for weight initialization
    avg_connections = (input_connections + output_connections) / 2
    if scale == 0:
        scale = 1e-10
    bound = math.sqrt(3 * scale / avg_connections)
    nn.init.uniform_(module.weight, -bound, bound)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


def conv(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, scale: float = 1.0):
    padding = kernel // 2 if stride == 1 else 0
    return init_module(nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding), scale)


def linear(in_ch: int, out_ch: int, scale: float = 1.0):
    return init_module(nn.Linear(in_ch, out_ch), scale)


def norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half < 2:
        raise ValueError('Embedding dim must be at least 4')
    freq = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32)
                     * (-math.log(10000.0) / (half - 1)))
    phases = timesteps.float()[:, None] * freq[None, :]
    out = torch.cat([phases.sin(), phases.cos()], dim=1)
    if dim % 2:
        out = F.pad(out, (0, 1))
    return out


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, dropout: float):
        super().__init__()
        self.norm1 = norm(in_ch)
        self.conv1 = conv(in_ch, out_ch)
        self.emb_proj = linear(temb_ch, out_ch)
        self.norm2 = norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv(out_ch, out_ch, scale=0.0)
        self.shortcut = linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        if isinstance(self.shortcut, nn.Identity):
            shortcut = self.shortcut(x)
        else:
            shortcut = self.shortcut(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return shortcut + h


class AttnBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.norm = norm(channels)
        self.q = conv(channels, channels, kernel=1)
        self.k = conv(channels, channels, kernel=1)
        self.v = conv(channels, channels, kernel=1)
        self.proj_out = conv(channels, channels, kernel=1, scale=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x)
        n = height * width
        q = self.q(h).reshape(batch, channels, n).transpose(1, 2)
        k = self.k(h).reshape(batch, channels, n)
        v = self.v(h).reshape(batch, channels, n).transpose(1, 2)
        weights = torch.softmax(torch.bmm(q, k) * (channels ** -0.5), dim=-1)
        out = torch.bmm(weights, v).transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.proj_out(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.down = conv(channels, channels, kernel=3, stride=2)

    def forward(self, x):
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("This original DDPM port expects even spatial sizes at each downsample")
        return self.down(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = conv(channels, channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class DDPMUNet(nn.Module):

    def __init__(self, resolution=32, in_channels=3, base_channels=128,
                 channel_mult=(1, 2, 2, 2), num_res_blocks=2,
                 attention_resolutions=(16,), dropout=0.1):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mult = tuple(channel_mult)
        self.attn_resolutions = tuple(attention_resolutions)
        self.num_res_blocks = num_res_blocks
        factor = 2 ** (len(channel_mult) - 1)
        if resolution % factor:
            raise ValueError(f'resolution {resolution} is not divisible by {factor}')

        self.time_dense0 = linear(base_channels, base_channels * 4)
        self.time_dense1 = linear(base_channels * 4, base_channels * 4)
        temb_dim = base_channels * 4
        self.conv_in = conv(in_channels, base_channels)
        self.down = nn.ModuleList()
        skip_channels = [base_channels]
        ch = base_channels
        current_res = resolution
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = base_channels * mult
                block = ResBlock(ch, out_ch, temb_dim, dropout)
                self.down.append(nn.ModuleDict({
                    'res': block,
                    'attn': AttnBlock(out_ch) if current_res in attention_resolutions else nn.Identity(),
                }))
                ch = out_ch
                skip_channels.append(ch)
            if level < len(channel_mult) - 1:
                self.down.append(nn.ModuleDict({'downsample': Downsample(ch)}))
                current_res //= 2
                skip_channels.append(ch)
        self.mid1 = ResBlock(ch, ch, temb_dim, dropout)
        self.mid_attn = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, temb_dim, dropout)
        self.up = nn.ModuleList()
        for level in reversed(range(len(channel_mult))):
            target_ch = base_channels * channel_mult[level]
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.up.append(nn.ModuleDict({
                    'res': ResBlock(ch + skip_ch, target_ch, temb_dim, dropout),
                    'attn': AttnBlock(target_ch) if current_res in attention_resolutions else nn.Identity(),
                }))
                ch = target_ch
            if level != 0:
                self.up.append(nn.ModuleDict({'upsample': Upsample(ch)}))
                current_res *= 2
        self.norm_out = norm(ch)
        self.conv_out = conv(ch, in_channels, scale=0.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if x.shape[1:] != (self.in_channels, self.resolution, self.resolution):
            raise ValueError('Image channels and resolution must match the U-Net configuration')
        if t.shape != (x.shape[0],) or t.dtype not in (torch.int32, torch.int64):
            raise ValueError('t must be a length-B integer tensor')
        temb = timestep_embedding(t, self.base_channels)
        temb = self.time_dense1(F.silu(self.time_dense0(temb)))
        h = self.conv_in(x)
        hs = [h]
        for stage in self.down:
            if 'downsample' in stage:
                h = stage['downsample'](h)
            else:
                h = stage['res'](h, temb)
                h = stage['attn'](h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for stage in self.up:
            if 'upsample' in stage:
                h = stage['upsample'](h)
            else:
                h = stage['res'](torch.cat((h, hs.pop()), dim=1), temb)
                h = stage['attn'](h)
        return self.conv_out(F.silu(self.norm_out(h)))
