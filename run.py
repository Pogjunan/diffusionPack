from diffusionpack.config import InferenceConfig, TrainConfig
from diffusionpack.restore_config import RestoreInferenceConfig, RestoreTrainConfig


task = "restore"
action = "train"

restore_train = RestoreTrainConfig(
    dataset="cifar10_jpeg",
    condition_dir="data/train/condition",
    target_dir="data/train/target",
    model="light",
    resolution=32,
    batch_size=8,
    epochs=30,
    output="runs/restore",
)

restore_inference = RestoreInferenceConfig(
    checkpoint="runs/restore/latest.pt",
    condition_path=None,
    output="runs/restore/images",
)


def main():
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
    raise ValueError("Choose restore or unconditional, and train or inference")


if __name__ == "__main__":
    main()
