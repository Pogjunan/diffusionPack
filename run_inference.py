from diffusionpack.config import InferenceConfig
from diffusionpack.main.inference import run_inference


config = InferenceConfig()


if __name__ == "__main__":
    run_inference(config)
