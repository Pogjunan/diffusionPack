from diffusionpack.prediction_config import PredictionInferenceConfig, PredictionTrainConfig
from diffusionpack.main.train_prediction import train_prediction
from diffusionpack.main.inference_prediction import infer_prediction


mode = "train"

train_config = PredictionTrainConfig()
inference_config = PredictionInferenceConfig()


if __name__ == "__main__":
    if mode == "train":
        train_prediction(train_config)
    elif mode == "inference":
        infer_prediction(inference_config)
    else:
        raise ValueError("Choose train or inference")
