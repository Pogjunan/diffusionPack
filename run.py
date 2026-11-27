from diffusionpack.config import InferenceConfig, TrainConfig
from diffusionpack.prediction_config import PredictionInferenceConfig, PredictionTrainConfig
from diffusionpack.restore_config import RestoreInferenceConfig, RestoreTrainConfig


task = "restore"
action = "train"

restore_train = RestoreTrainConfig()
restore_inference = RestoreInferenceConfig()
prediction_train = PredictionTrainConfig()
prediction_inference = PredictionInferenceConfig()
unconditional_train = TrainConfig()
unconditional_inference = InferenceConfig()


def main():
    if task == "prediction" and action == "train":
        from diffusionpack.main.train_prediction import train_prediction
        return train_prediction(prediction_train)
    if task == "prediction" and action == "inference":
        from diffusionpack.main.inference_prediction import infer_prediction
        return infer_prediction(prediction_inference)
    if task == "restore" and action == "train":
        from diffusionpack.main.train_restore import train_restore
        return train_restore(restore_train)
    if task == "restore" and action == "inference":
        from diffusionpack.main.inference_restore import infer_restore
        return infer_restore(restore_inference)
    if task == "unconditional" and action == "train":
        from diffusionpack.main.train import run_training
        return run_training(unconditional_train)
    if task == "unconditional" and action == "inference":
        from diffusionpack.main.inference import run_inference
        return run_inference(unconditional_inference)
    raise ValueError("Choose unconditional, restore or prediction, and train or inference")


if __name__ == "__main__":
    main()
