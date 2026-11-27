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

DDPM equations, the U-Net, reverse sampling, and a single-batch training step are included as preliminary PyTorch implementations. Full training and generation quality have not been verified.
