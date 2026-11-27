from torch import nn


class SpatialEncoder(nn.Module):
    def __init__(self, input_channels, widths):
        super().__init__()
        self.blocks = nn.ModuleList()
        for index, channels in enumerate(widths):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(input_channels, channels, 3, stride=1 if index == 0 else 2, padding=1),
                nn.SiLU(),
            ))
            input_channels = channels

    def forward(self, condition):
        features = []
        for block in self.blocks:
            condition = block(condition)
            features.append(condition)
        return features


class TCU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dwd_conv = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels)
        self.conv1x1 = nn.Conv2d(channels, channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, channels)

        # Inherited initialization; the paper does not specify gate initialization.
        nn.init.xavier_uniform_(self.conv1x1.weight, gain=0.1)
        nn.init.ones_(self.conv1x1.bias)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.ones_(self.fc.bias)

    def forward(self, features):
        spatial_cues = self.conv1x1(self.dwd_conv(self.dw_conv(features)))
        channel_cues = self.fc(self.pool(features).flatten(1))[:, :, None, None]
        return spatial_cues * channel_cues * features


class STCG(nn.Module):
    def __init__(self, input_channels, widths, tcu_depth):
        super().__init__()
        self.encoder = SpatialEncoder(input_channels, widths)
        self.tcu_blocks = nn.ModuleList([
            nn.Sequential(*(TCU(channels) for _ in range(tcu_depth)))
            for channels in widths
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1) for channels in widths
        ])

    def forward(self, condition):
        encoded = self.encoder(condition)
        return [conv(tcu(features)) for features, tcu, conv in
                zip(encoded, self.tcu_blocks, self.output_convs)]
