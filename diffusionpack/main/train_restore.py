from ..backbone.restore_unet import RestoreUNet
from ..restore import restore_loss
from .restore_data import CifarJpeg, PairedImages
from .train import load_checkpoint, model_settings, start_training, train_batches


def train_restore(config):
    if config.steps is None and config.epochs is None:
        raise ValueError("Set epochs or steps")
    device = start_training(config)
    settings = model_settings(config.model, resolution=config.resolution,
                              base_channels=config.base_channels)
    del settings["in_channels"]
    settings.update(condition_channels=config.condition_channels,
                    target_channels=config.target_channels)
    if config.dataset == "paired":
        dataset = PairedImages(config.condition_dir, config.target_dir, config.resolution,
                               config.condition_channels, config.target_channels)
    elif config.dataset == "cifar10_jpeg":
        if config.target_channels != 3 or config.condition_channels != 3:
            raise ValueError("CIFAR JPEG needs three condition and three target channels")
        dataset = CifarJpeg(config.data_root, config.resolution, quality=config.jpeg_quality)
    else:
        raise ValueError("Choose cifar10_jpeg or paired")

    saved = load_checkpoint(config.resume) if config.resume else None
    training = {"task": "restore"}
    if config.dataset == "cifar10_jpeg":
        training["jpeg_quality"] = config.jpeg_quality
    model = RestoreUNet(**settings).to(device)
    return train_batches(config, model, dataset, settings, config.dataset, training, saved, restore_loss)
