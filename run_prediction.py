from diffusionpack.prediction_config import PredictionInferenceConfig, PredictionTrainConfig
from diffusionpack.main.train_prediction import train_prediction
from diffusionpack.main.inference_prediction import infer_prediction
from diffusionpack.main.check_prediction import check_prediction


mode = "check"

train_config = PredictionTrainConfig()
inference_config = PredictionInferenceConfig()


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
