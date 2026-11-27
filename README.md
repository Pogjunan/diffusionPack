# diffusionPack

## Abstract

Diffusion models are widely used in generative modeling and have applications in many fields. This package aims to provide reusable PyTorch baselines for unconditional and conditional diffusion, starting with Denoising Diffusion Probabilistic Models (DDPM). I want to reduce the time spent implementing diffusion models from scratch. The package will support research on new conditioning methods and the reimplementation of prior work, including papers without publicly available code. I will extend it step by step as I study other methods, such as DDIM, score-based models, classifier-free guidance, flow matching, and drifting models.

## Implementation Principles

I implement the models myself when possible, using research papers and public reference code as guides. I use generative AI mainly to check whether the code matches the mathematics and to find errors. It may also help me write functions beyond my current experience. I review AI-assisted code against the papers and available implementations whenever possible. My goal is to understand the code and build a reliable foundation for future research.

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/Pogjunan/diffusionPack.git
```

Update the package:

```bash
pip install --upgrade --force-reinstall git+https://github.com/Pogjunan/diffusionPack.git
```

## Current Status

The DDPM training step supports the original CIFAR-10 learning rate warmup and exponential moving average (EMA). Full training and generation quality have not been verified.

## Training

Start with a DDPM U-Net on the same device as your image batches. Initialize the optimizer, scheduler, and EMA model once per run.

```python
from diffusionpack.main.train import init_training, train_step

optimizer, scheduler, ema_model = init_training(model)
for images in image_batches:
    loss = train_step(model, images, optimizer, scheduler, ema_model)
```

Use `ema_model` with `p_sample_loop` for sampling.
