import torch
from torch import nn

from .ddpm_unet import DDPMUNet


class RestoreUNet(DDPMUNet):
    def __init__(self, target_channels=3, condition_channels=3, **settings):
        if target_channels < 1 or condition_channels < 1:
            raise ValueError("Target and condition must both have at least one channel")
        super().__init__(in_channels=target_channels + condition_channels,
                         use_adagn=True, **settings)
        self.target_channels = target_channels
        self.condition_channels = condition_channels
        self.conv_out = nn.Conv2d(self.conv_out.in_channels, target_channels, 3, padding=1)

    def forward(self, noisy_target, gamma, condition):
        if noisy_target.shape[1] != self.target_channels or condition.shape[1] != self.condition_channels:
            raise ValueError("Target or condition channel count differs from the model configuration")
        model_input = torch.cat((noisy_target, condition), dim=1)
        return super().forward(model_input, gamma)
