from .restore_unet import RestoreUNet


class PredictionUNet(RestoreUNet):
    def __init__(self, history_steps=5, map_channels=1, future_steps=1, **settings):
        if any(type(n) is not int or n < 1 for n in (history_steps, future_steps, map_channels)):
            raise ValueError("History, future and channels must be positive integers")
        super().__init__(
            target_channels=future_steps * map_channels,
            condition_channels=history_steps * map_channels,
            **settings,
        )
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.map_channels = map_channels
