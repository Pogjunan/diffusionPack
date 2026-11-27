from diffusionpack.prediction_config import PredictionInferenceConfig, PredictionTrainConfig
from diffusionpack.main.train_prediction import train_prediction
from diffusionpack.main.inference_prediction import infer_prediction


mode = "train"

train_config = PredictionTrainConfig(
    manifest="data/forecast/train.csv",
    model="light",
    resolution=32,
    channel_mult=(1, 2, 2, 2),
    history_steps=5,
    cadence_minutes=30,
    lead_minutes=30,
    history_lag_minutes=0,
    value_min=None,
    value_max=None,
    units="W/m2",
    batch_size=8,
    epochs=30,
    output="runs/prediction",
)

inference_config = PredictionInferenceConfig(
    checkpoint="runs/prediction/latest.pt",
    manifest="data/forecast/test.csv",
    output="runs/prediction/forecasts",
    max_origins=1,
    samples=1,
)


if __name__ == "__main__":
    if mode == "train":
        train_prediction(train_config)
    elif mode == "inference":
        infer_prediction(inference_config)
    else:
        raise ValueError("Choose train or inference")
