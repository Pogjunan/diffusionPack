from .restore_unet import RestoreUNet


class PredictionUNet(RestoreUNet):
    def __init__(self, history_steps=5, map_channels=1, **settings):
        super().__init__(
            target_channels=map_channels,
            condition_channels=history_steps * map_channels,
            **settings,
        )
        self.history_steps = history_steps
        self.map_channels = map_channels
