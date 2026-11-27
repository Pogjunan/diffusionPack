import torch
from torch import nn
from torch.nn import functional as F


class SWCrossAttention(nn.Module):
    def __init__(self, channels, window=8, shift=0):
        super().__init__()
        self.window = window
        self.shift = shift
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def partition(self, features):
        batch, channels, height, width = features.shape
        window = self.window
        return (features.reshape(batch, channels, height // window, window, width // window, window)
                .permute(0, 2, 4, 3, 5, 1).reshape(-1, window * window, channels))

    def restore(self, windows, batch, height, width):
        window = self.window
        return (windows.reshape(batch, height // window, width // window, window, window, -1)
                .permute(0, 5, 1, 3, 2, 4).reshape(batch, -1, height, width))

    def attention_mask(self, height, width, padded_height, padded_width, shift, device):
        valid = torch.zeros(1, 1, padded_height, padded_width, dtype=torch.bool, device=device)
        valid[:, :, :height, :width] = True
        valid = self.partition(torch.roll(valid, (-shift, -shift), (-2, -1))).squeeze(-1)
        allowed = valid[:, :, None] & valid[:, None, :]

        if shift:
            regions = torch.zeros(1, 1, padded_height, padded_width, device=device)
            slices = (slice(0, -self.window), slice(-self.window, -shift), slice(-shift, None))
            for row, row_slice in enumerate(slices):
                for column, column_slice in enumerate(slices):
                    regions[:, :, row_slice, column_slice] = row * 3 + column
            regions = self.partition(regions).squeeze(-1)
            allowed = allowed & (regions[:, :, None] == regions[:, None, :])

        diagonal = torch.eye(self.window ** 2, dtype=torch.bool, device=device)
        return allowed | (~valid[:, :, None] & diagonal[None])

    def forward(self, noise_features, condition_features):
        batch, channels, height, width = noise_features.shape
        pad_height, pad_width = (-height) % self.window, (-width) % self.window
        noise = F.pad(noise_features, (0, pad_width, 0, pad_height))
        condition = F.pad(condition_features, (0, pad_width, 0, pad_height))
        padded_height, padded_width = noise.shape[-2:]
        shift = self.shift if min(height, width) > self.window else 0

        query = self.partition(torch.roll(self.query(noise), (-shift, -shift), (-2, -1)))
        key = self.partition(torch.roll(self.key(condition), (-shift, -shift), (-2, -1)))
        value = self.partition(torch.roll(self.value(noise), (-shift, -shift), (-2, -1)))
        window_shape = (batch, -1, self.window ** 2, channels)
        query, key, value = (tensor.reshape(window_shape) for tensor in (query, key, value))

        mask = None
        if shift or pad_height or pad_width:
            mask = self.attention_mask(height, width, padded_height, padded_width, shift, noise.device)
            mask = mask.unsqueeze(0)

        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        attended = self.restore(attended, batch, padded_height, padded_width)
        attended = torch.roll(attended, (shift, shift), (-2, -1))[:, :, :height, :width]
        return self.proj_out(attended)


class SWCIBlock(nn.Module):
    def __init__(self, channels, output_channels, mode, time_dim, window, norm_groups):
        super().__init__()
        self.mode = mode
        self.norm1 = nn.GroupNorm(norm_groups, channels, eps=1e-6)
        self.norm2 = nn.GroupNorm(norm_groups, channels, eps=1e-6)
        self.time_fc = nn.Linear(time_dim, 2 * channels)
        self.attention1 = SWCrossAttention(channels, window)
        self.attention2 = SWCrossAttention(channels, window, shift=window // 2)
        if mode == "down":
            self.conv = nn.Conv2d(channels, output_channels, 3, stride=2, padding=1)
            self.skip = nn.Conv2d(channels, output_channels, 1, stride=2)
        elif mode == "up":
            self.conv = nn.ConvTranspose2d(channels, output_channels, 4, stride=2, padding=1)
            self.skip = nn.Conv2d(channels, output_channels, 1)
        else:
            self.conv = nn.Conv2d(channels, output_channels, 3, padding=1)
            self.skip = nn.Conv2d(channels, output_channels, 1)

    def forward(self, features, condition, time_embedding):
        scale, shift = self.time_fc(time_embedding).chunk(2, dim=1)
        scale, shift = scale[:, :, None, None], shift[:, :, None, None]
        skip = F.interpolate(features, scale_factor=2, mode="nearest") if self.mode == "up" else features
        features = self.attention1(self.norm1(features) * (1.0 + scale) + shift, condition)
        features = self.attention2(self.norm2(features) * (1.0 + scale) + shift, condition)
        return self.conv(features) + self.skip(skip)
