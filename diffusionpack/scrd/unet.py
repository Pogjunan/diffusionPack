import math

import torch
from torch import nn
from torch.nn import functional as F

from .stcg import STCG
from .swci import SWCIBlock


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, timestep):
        half = self.channels // 2
        frequencies = torch.exp(
            torch.arange(half, device=timestep.device, dtype=torch.float32)
            * (-math.log(10000) / (half - 1))
        )
        angles = timestep.float()[:, None] * frequencies[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class SWCIUNet(nn.Module):
    def __init__(self, condition_channels, target_channels, widths, time_dim,
                 window, norm_groups, output_norm_groups):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_conv = nn.Conv2d(condition_channels + target_channels, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([
            SWCIBlock(channels, widths[index + 1] if index < len(widths) - 1 else channels,
                      "down" if index < len(widths) - 1 else "same", time_dim, window, norm_groups)
            for index, channels in enumerate(widths)
        ])
        self.up_blocks = nn.ModuleList([
            SWCIBlock(channels, widths[index - 1] if index > 0 else channels,
                      "up" if index > 0 else "same", time_dim, window, norm_groups)
            for index, channels in enumerate(widths)
        ])
        self.merge_convs = nn.ModuleList([
            nn.Conv2d(2 * channels, channels, 1) for channels in widths[:-1]
        ])
        self.output_norm = nn.GroupNorm(output_norm_groups, widths[0], eps=1e-6)
        self.output_conv = nn.Conv2d(widths[0], target_channels, 3, padding=1)

    def forward(self, noisy, condition, condition_features, timestep):
        time_embedding = self.time_embedding(timestep)
        features = self.input_conv(torch.cat([noisy, condition], dim=1))
        saved_features = []
        for index, block in enumerate(self.down_blocks):
            saved_features.append(features)
            features = block(features, condition_features[index], time_embedding)

        for index in reversed(range(len(self.up_blocks))):
            if index < len(self.up_blocks) - 1:
                skip = saved_features[index]
                features = F.interpolate(features, size=skip.shape[-2:], mode="nearest")
                features = self.merge_convs[index](torch.cat([features, skip], dim=1))
            features = self.up_blocks[index](features, condition_features[index], time_embedding)
        return self.output_conv(F.silu(self.output_norm(features)))


class SCRDUNet(nn.Module):
    """STCG + SWCI, with history and future frames stacked along channels."""

    def __init__(self, map_channels=1, history_steps=5, future_steps=4,
                 widths=(32, 64, 128, 256), window=8, tcu_depth=3, time_dim=128,
                 norm_groups=32, output_norm_groups=8, use_tcu=True, resolution=None):
        super().__init__()
        if resolution is not None and (type(resolution) is not int or resolution < 1):
            raise ValueError("resolution must be a positive integer")
        self.resolution = resolution
        counts = (map_channels, history_steps, future_steps, window, tcu_depth,
                  time_dim, norm_groups, output_norm_groups)
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("Channel, frame, window, depth and group counts must be positive integers")
        widths = tuple(widths)
        if not widths or any(type(channels) is not int or channels < 1 for channels in widths):
            raise ValueError("widths must contain positive integers")
        if any(channels % norm_groups for channels in widths) or widths[0] % output_norm_groups:
            raise ValueError("Each channel width must be divisible by its GroupNorm group count")
        if time_dim < 4 or time_dim % 2:
            raise ValueError("time_dim must be even and at least 4")
        self.map_channels = map_channels
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.condition_channels = history_steps * map_channels
        self.target_channels = future_steps * map_channels
        self.widths = widths
        self.use_tcu = use_tcu
        # TCU ablation: keep the spatial encoder and output conv.
        # This is not a verified reproduction of the paper's entire STCG removal.
        self.stcg = STCG(self.condition_channels, widths, tcu_depth if use_tcu else 0)
        self.unet = SWCIUNet(self.condition_channels, self.target_channels, widths,
                            time_dim, window, norm_groups, output_norm_groups)

    def prepare_condition(self, condition):
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise ValueError("condition must have shape [B, history_steps * map_channels, H, W]")
        if self.resolution is not None and condition.shape[-2:] != (self.resolution, self.resolution):
            raise ValueError("Condition resolution differs from the model configuration")
        return self.stcg(condition)

    def forward(self, noisy, timestep, condition, condition_features=None):
        if noisy.ndim != 4 or noisy.shape[1] != self.target_channels:
            raise ValueError("noisy must have shape [B, future_steps * map_channels, H, W]")
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise ValueError("condition must have shape [B, history_steps * map_channels, H, W]")
        if noisy.shape[0] != condition.shape[0] or noisy.shape[-2:] != condition.shape[-2:]:
            raise ValueError("noisy and condition must have the same batch and spatial dimensions")
        if self.resolution is not None and noisy.shape[-2:] != (self.resolution, self.resolution):
            raise ValueError("Prediction resolution differs from the model configuration")
        if timestep.shape != (noisy.shape[0],):
            raise ValueError("timestep must have shape [B]")
        if min(noisy.shape) < 1:
            raise ValueError("Batch and spatial dimensions must be positive")
        if condition_features is None:
            condition_features = self.prepare_condition(condition)
        if len(condition_features) != len(self.widths):
            raise ValueError("One condition feature is required for each U-Net level")
        height, width = noisy.shape[-2:]
        for index, (features, channels) in enumerate(zip(condition_features, self.widths)):
            if index:
                height, width = (height + 1) // 2, (width + 1) // 2
            if features.shape != (noisy.shape[0], channels, height, width):
                raise ValueError(f"Condition feature {index} has an incompatible shape")
        prediction = self.unet(noisy, condition, condition_features, timestep)
        if prediction.shape != noisy.shape:
            raise RuntimeError("The predicted epsilon shape differs from the noisy target shape")
        return prediction
