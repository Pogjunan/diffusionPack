from diffusionpack.config import InferenceConfig
from diffusionpack.main.inference import run_inference


config = InferenceConfig(
    checkpoint="runs/cifar_light/latest.pt",
    samples=8,
    batch_size=2,
    output="runs/cifar_light/samples.png",
    device="auto",
    weights="ema",
    progress=True,
)


if __name__ == "__main__":
    run_inference(config)
