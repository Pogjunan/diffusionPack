import argparse
from pathlib import Path

from diffusionpack.scrd.runner import diagnose, infer, train
from diffusionpack.scrd.config import SCRDConfig


def main():
    parser = argparse.ArgumentParser(description="SCRD and matched DDPM comparisons; edit scrd/config.py")
    parser.add_argument("action", choices=("check-data", "train", "infer", "evaluate", "persistence", "diagnose"))
    parser.add_argument("--backbone", choices=("ddpm", "scrd"))
    parser.add_argument("--target", choices=("future", "residual"))
    parser.add_argument("--noise-conditioning", choices=("timestep", "gamma"))
    parser.add_argument("--reverse-variance", choices=("beta", "posterior"))
    for name in ("output", "resume", "checkpoint", "device", "train-csv", "valid-csv", "test-csv",
                 "train-targets", "valid-targets", "test-targets"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--max-origins", type=int)
    parser.add_argument("--all-origins", action="store_true")
    options = vars(parser.parse_args())
    action = options.pop("action")
    all_origins = options.pop("all_origins")
    config = SCRDConfig(**{key: value for key, value in options.items() if value is not None})
    if options["output"] is None and action not in ("train", "check-data"):
        config.output = str(Path(config.checkpoint).parent / action)
    if all_origins:
        config.max_origins = None
    if action in ("train", "check-data"):
        train(config, check_data=action == "check-data")
    elif action == "diagnose":
        diagnose(config)
    else:
        infer(config, evaluate=action != "infer", persistence=action == "persistence")


if __name__ == "__main__":
    main()
