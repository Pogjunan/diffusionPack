from ..backbone.prediction_unet import PredictionUNet
from ..prediction import prediction_loss
from .prediction_data import ForecastDataset, manifest_hash
from .train import load_checkpoint, model_settings, start_training, train_batches


def prediction_settings(config):
    settings = model_settings(config.model, resolution=config.resolution,
                              base_channels=config.base_channels)
    del settings["in_channels"]
    settings.update(
        history_steps=config.history_steps,
        map_channels=config.map_channels,
    )
    if config.channel_mult is not None:
        settings["channel_mult"] = tuple(config.channel_mult)
    if config.resolution >= 16:
        resolutions = [config.resolution // (2 ** level) for level in range(len(settings["channel_mult"]))]
        if 16 not in resolutions:
            raise ValueError("channel_mult must include a 16x16 resolution level")
    return settings


def train_prediction(config):
    if config.steps is None and config.epochs is None:
        raise ValueError("Set epochs or steps")
    device = start_training(config)
    dataset = ForecastDataset(
        config.manifest,
        history_steps=config.history_steps,
        map_channels=config.map_channels,
        resolution=config.resolution,
        cadence_minutes=config.cadence_minutes,
        history_lag_minutes=config.history_lag_minutes,
        lead_minutes=config.lead_minutes,
        value_min=config.value_min,
        value_max=config.value_max,
        require_target=True,
    )
    if not config.units:
        raise ValueError("Set physical units in the prediction config")
    settings = prediction_settings(config)
    current = dict(
        task="prediction",
        manifest_sha256=manifest_hash(config.manifest),
        cadence_minutes=config.cadence_minutes,
        history_lag_minutes=config.history_lag_minutes,
        lead_minutes=config.lead_minutes,
        value_min=config.value_min,
        value_max=config.value_max,
        units=config.units,
    )
    checkpoint = load_checkpoint(config.resume) if config.resume else None
    model = PredictionUNet(**settings).to(device)
    return train_batches(config, model, dataset, settings, "forecast", current, checkpoint, prediction_loss)
