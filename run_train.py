from diffusionpack.config import TrainConfig
from diffusionpack.main.train import run_training


config = TrainConfig(
    model="light",
    dataset="cifar10",
    data_root="data",
    batch_size=8,
    epochs=30,
    steps=None,
    lr=0.0002,
    warmup_steps=5000,
    output="runs/cifar_light",
    resume=None,
    save_every=1000,
    log_every=100,
    device="auto",
    amp=True,
    progress=True,
)


if __name__ == "__main__":
    run_training(config)
