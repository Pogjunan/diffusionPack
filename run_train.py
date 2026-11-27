from diffusionpack.config import TrainConfig
from diffusionpack.main.train import run_training


config = TrainConfig()


if __name__ == "__main__":
    run_training(config)
