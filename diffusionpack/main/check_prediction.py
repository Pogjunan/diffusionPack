import torch

from ..backbone.prediction_unet import PredictionUNet
from ..prediction import prediction_loss
from ..prediction_config import PredictionTrainConfig
from ..scrd.unet import SCRDUNet
from ..scrd.prediction import scrd_prediction_loss
from .train import choose_device
from .train_prediction import prediction_dataset, prediction_settings


def check_prediction(config, valid_csv=None, valid_targets=None, test_csv=None,
                     test_targets=None, batch_size=1, check_all=False):
    if not config.units:
        raise ValueError("Set the physical units in the prediction config")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Check batch_size must be a positive integer")

    train = prediction_dataset(config)
    splits = {"train": train}
    for name, csv_path, target_csv in (("valid", valid_csv, valid_targets),
                                       ("test", test_csv, test_targets)):
        if csv_path is None and target_csv is not None:
            raise ValueError(f"Set {name}_csv when using {name}_targets")
        if csv_path is not None:
            options = dict(vars(config), csv_path=csv_path, target_csv=target_csv)
            splits[name] = prediction_dataset(PredictionTrainConfig(**options))
    targets = {}
    histories = {}
    for name, source in splits.items():
        print(f"{name}: {len(source)} windows; exclusions={source.rejections}")
        targets[name] = {time for _, _, future in source.windows for time in future}
        histories[name] = {time for _, history, _ in source.windows for time in history}
    for i, first in enumerate(splits):
        for second in list(splits)[i + 1:]:
            overlap = targets[first] & targets[second]
            if overlap:
                raise ValueError(f"{first}/{second} targets overlap: {min(overlap)}")
            overlap = histories[first] & targets[second]
            if overlap:
                raise ValueError(f"{second} targets appear in {first} history: {min(overlap)}")
            if max(targets[first]) > min(origin for origin, _, _ in splits[second].windows):
                raise ValueError(f"{first}/{second} are not chronologically separated; review the split")
    if len(splits) < 3:
        print("Set validation and test CSVs to check all three splits.")

    if check_all:
        for name in ("train", "valid"):
            if name in splits:
                for moment in sorted(histories[name] | targets[name]):
                    splits[name].load_grid(moment)
                print(f"{name}: all used grids checked")
    device = choose_device(config.device)
    settings = prediction_settings(config)
    model = (SCRDUNet(**settings) if config.backbone == "scrd" else PredictionUNet(**settings)).to(device)
    loss_function = scrd_prediction_loss if config.backbone == "scrd" else prediction_loss
    batch = [train[index] for index in range(min(batch_size, len(train)))]
    history = torch.stack([item[0] for item in batch]).to(device)
    future = torch.stack([item[1] for item in batch]).to(device)
    with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
        loss = loss_function(model, future, history)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite prediction loss")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("Missing or non-finite model gradients")
    print(f"Device: {device}; history: {tuple(history.shape)}; future: {tuple(future.shape)}")
    print(f"Backbone: {config.backbone}; loss: {loss.item():.6f}; gradients finite")
    print("Batch check passed.")
    return loss.item()
