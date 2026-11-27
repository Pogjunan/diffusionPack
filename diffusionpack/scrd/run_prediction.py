from diffusionpack.prediction_config import PredictionInferenceConfig, PredictionTrainConfig
from diffusionpack.main.train_prediction import train_prediction
from diffusionpack.main.inference_prediction import infer_prediction
from diffusionpack.main.check_prediction import check_prediction


mode = "check"

train_config = PredictionTrainConfig(
    backbone="scrd",
    history_steps=5,
    future_steps=4,
    resolution=512,
    csv_path="data/forecast/train.csv",
    target_csv=None,
    value_min=None,
    value_max=None,
    units=None,
    steps=250000,
    epochs=None,
    lr=0.0002,
    warmup_steps=0,
    amp=False,
    output="runs/prediction_scrd",
)
inference_config = PredictionInferenceConfig(
    checkpoint="runs/prediction_scrd/latest.pt",
    csv_path="data/forecast/test.csv",
    target_csv=None,
    output="runs/prediction_scrd/forecasts",
    weights="model",
    samples=8,
)


valid_csv = None
valid_targets = None
test_csv = None
test_targets = None
check_batch_size = 1
check_all = False


if __name__ == "__main__":
    if mode == "check":
        check_prediction(train_config, valid_csv, valid_targets, test_csv, test_targets,
                         batch_size=check_batch_size, check_all=check_all)
    elif mode == "train":
        train_prediction(train_config)
    elif mode == "inference":
        infer_prediction(inference_config)
    else:
        raise ValueError("Choose check, train or inference")
