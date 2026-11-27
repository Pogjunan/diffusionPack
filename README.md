# diffusionPack

## Abstract

diffusionPack is a PyTorch package for developing and studying diffusion-based generative models. It starts with Denoising Diffusion Probabilistic Models (DDPM) and aims to provide reusable implementations for unconditional and conditional generation, reducing the need to rebuild baselines for each research project. Planned extensions include DDIM, score-based generative models, classifier-free guidance, flow matching, and related methods.

## Implementation Principles

Implementations are developed with reference to original research papers and publicly available author code. The focus is on understanding the underlying mathematics, keeping the implementation readable, and checking numerical behavior before using models in research. The package is intended to support reproduction, comparison, and extension of generative modeling methods.

## Installation

```bash
pip install git+https://github.com/Pogjunan/diffusionPack.git
```

## Update

```bash
pip install --upgrade --force-reinstall git+https://github.com/Pogjunan/diffusionPack.git
```
