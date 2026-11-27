from pathlib import Path

from ..backbone.prediction_unet import PredictionUNet
from ..scrd.unet import SCRDUNet
from ..prediction import prediction_loss
from ..scrd.prediction import scrd_prediction_loss
from .prediction_data import ForecastDataset
from .train import load_checkpoint, model_settings, start_training, train_batches


def prediction_settings(config):
    if config.backbone == "scrd":
        return dict(resolution=config.resolution, history_steps=config.history_steps,
                    future_steps=config.future_steps, map_channels=config.map_channels,
                    widths=tuple(config.scrd_widths), window=config.scrd_window,
                    tcu_depth=config.scrd_tcu_depth, time_dim=config.scrd_time_dim,
                    norm_groups=config.scrd_norm_groups, output_norm_groups=config.scrd_output_norm_groups,
                    use_tcu=config.scrd_use_tcu)
    if config.backbone != "ddpm":
        raise ValueError("Prediction backbone must be ddpm or scrd")
    settings = model_settings(config.model, resolution=config.resolution,
                              base_channels=config.base_channels)
    del settings["in_channels"]
    settings.update(
        history_steps=config.history_steps,
        future_steps=config.future_steps,
        map_channels=config.map_channels,
    )
    if config.channel_mult is not None:
        settings["channel_mult"] = tuple(config.channel_mult)
    if config.resolution >= 16:
        resolutions = [config.resolution // (2 ** level) for level in range(len(settings["channel_mult"]))]
        if 16 not in resolutions:
            raise ValueError("channel_mult must include a 16x16 resolution level")
    return settings


def prediction_dataset(config):
    return ForecastDataset(
        config.csv_path,
        history_steps=config.history_steps,
        future_steps=config.future_steps,
        map_channels=config.map_channels,
        resolution=config.resolution,
        cadence_minutes=config.cadence_minutes,
        history_lag_minutes=config.history_lag_minutes,
        lead_minutes=config.lead_minutes,
        value_min=config.value_min,
        value_max=config.value_max,
        require_target=True,
        target_csv=config.target_csv,
    )


def train_prediction(config):
    if config.steps is None and config.epochs is None:
        raise ValueError("Set epochs or steps")
    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        if not config.resume or Path(config.resume).resolve() != (output / "latest.pt").resolve():
            raise FileExistsError("Use an empty output folder or resume its latest.pt")
    device = start_training(config)
    dataset = prediction_dataset(config)
    if not config.units:
        raise ValueError("Set physical units in the prediction config")
    settings = prediction_settings(config)
    current = dict(
        task="prediction",
        noise_conditioning="gamma_adagn",
        cadence_minutes=config.cadence_minutes,
        history_lag_minutes=config.history_lag_minutes,
        lead_minutes=config.lead_minutes,
        value_min=config.value_min,
        value_max=config.value_max,
        units=config.units,
    )
    if config.target_csv is not None:
        current["target_csv"] = str(config.target_csv)
    if config.backbone == "scrd":
        current.update(backbone="scrd", noise_conditioning="scrd_timestep",
                       prediction_target="residual", reverse_variance="beta")
    checkpoint = load_checkpoint(config.resume) if config.resume else None
    if checkpoint:
        previous = checkpoint.get("training", {})
        if previous.get("backbone", "ddpm") != config.backbone:
            raise ValueError("Resume with the same prediction backbone")
        if previous.get("noise_conditioning") != current["noise_conditioning"]:
            raise ValueError("Retrain old conditional checkpoints for the selected noise conditioning")
        if previous.get("target_csv") != current.get("target_csv"):
            raise ValueError("Resume with the same target CSV selection")
    if config.backbone == "scrd":
        model = SCRDUNet(**settings).to(device)
        loss_function = scrd_prediction_loss
    else:
        model = PredictionUNet(**settings).to(device)
        loss_function = prediction_loss
    return train_batches(config, model, dataset, settings, "forecast", current, checkpoint, loss_function)
