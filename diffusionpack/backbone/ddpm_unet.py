import math
import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timesteps, dim):
    i = torch.arange(dim // 2, device=timesteps.device, dtype=torch.float32)
    freq = torch.exp(i * (-math.log(10000.0) / max(dim // 2 - 1, 1)))
    angle = timesteps.float()[:, None] * freq[None, :]
    emb = torch.cat([angle.sin(), angle.cos()], dim=1)
    return F.pad(emb, (0, dim % 2))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, dropout):
        super().__init__()
        # 32 channel groups
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # Channel projection
        self.shortcut = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(temb))[:, :, None, None]
        residual = self.conv2(self.dropout(F.silu(self.norm2(h))))
        x = self.shortcut(x.movedim(1, -1)).movedim(-1, 1)
        return x + residual  # S(x) + R(x, t)


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        channels = x.shape[1]
        q = self.q(h).flatten(2).transpose(1, 2)
        k = self.k(h).flatten(2)
        v = self.v(h).flatten(2).transpose(1, 2)
        # softmax(QK^T / sqrt(C)) V
        weights = torch.softmax(torch.bmm(q, k) * (channels ** -0.5), dim=-1)
        out = torch.bmm(weights, v).transpose(1, 2).reshape_as(x)
        return x + self.proj_out(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, x):
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("This original DDPM port expects even spatial sizes at each downsample")
        return self.down(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


def default_channel_mult(resolution):
    levels = 4
    while resolution > 16 * 2 ** (levels - 1):
        levels += 1
    return (1, 2, 2, 2) + (2,) * (levels - 4)


class DDPMUNet(nn.Module):

    def __init__(self, resolution=32, in_channels=3, base_channels=128,
                 channel_mult=None, num_res_blocks=2,
                 attention_resolutions=(16,), dropout=0.1):
        super().__init__()
        if not isinstance(resolution, int) or resolution < 8:
            raise ValueError("resolution must be an integer at least 8")
        if channel_mult is None:
            channel_mult = default_channel_mult(resolution)
        self.resolution = resolution
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mult = tuple(channel_mult)
        self.attn_resolutions = tuple(attention_resolutions)
        self.num_res_blocks = num_res_blocks
        factor = 2 ** (len(channel_mult) - 1)
        if resolution % factor:
            raise ValueError(f'resolution {resolution} is not divisible by {factor}')
        if resolution // factor > 32:
            raise ValueError("Bottleneck attention exceeds 32x32; add downsampling levels")

        self.time_dense0 = nn.Linear(base_channels, base_channels * 4)
        self.time_dense1 = nn.Linear(base_channels * 4, base_channels * 4)
        temb_dim = base_channels * 4
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
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
        self.norm_out = nn.GroupNorm(32, ch, eps=1e-6)
        self.conv_out = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        if x.shape[1:] != (self.in_channels, self.resolution, self.resolution):
            raise ValueError('Image channels and resolution must match the U-Net configuration')
        if t.shape != (x.shape[0],) or t.dtype not in (torch.int32, torch.int64):
            raise ValueError('t must be a length-B integer tensor')
        t = t.to(device=x.device)
        temb = timestep_embedding(t, self.base_channels).to(self.time_dense0.weight.dtype)
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
        h = self.mid1(h, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)
        for stage in self.up:
            if 'upsample' in stage:
                h = stage['upsample'](h)
            else:
                h = stage['res'](torch.cat((h, hs.pop()), dim=1), temb)
                h = stage['attn'](h)
        return self.conv_out(F.silu(self.norm_out(h)))
