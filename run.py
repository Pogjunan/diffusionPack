from diffusionpack.config import InferenceConfig, TrainConfig
from diffusionpack.prediction_config import PredictionInferenceConfig, PredictionTrainConfig
from diffusionpack.restore_config import RestoreInferenceConfig, RestoreTrainConfig


task = "restore"
action = "train"

restore_train = RestoreTrainConfig()
restore_inference = RestoreInferenceConfig()


def main():
    if task == "prediction" and action == "train":
        from diffusionpack.main.train_prediction import train_prediction
        return train_prediction(PredictionTrainConfig())
    if task == "prediction" and action == "inference":
        from diffusionpack.main.inference_prediction import infer_prediction
        return infer_prediction(PredictionInferenceConfig())
    if task == "restore" and action == "train":
        from diffusionpack.main.train_restore import train_restore
        return train_restore(restore_train)
    if task == "restore" and action == "inference":
        from diffusionpack.main.inference_restore import infer_restore
        return infer_restore(restore_inference)
    if task == "unconditional" and action == "train":
        from diffusionpack.main.train import run_training
        return run_training(TrainConfig())
    if task == "unconditional" and action == "inference":
        from diffusionpack.main.inference import run_inference
        return run_inference(InferenceConfig())
    raise ValueError("Choose unconditional, restore or prediction, and train or inference")


if __name__ == "__main__":
    main()
