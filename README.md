# diffusionPack

## Abstract

Diffusion models are widely used in generative modeling and many applied problems. This project aims to build a reusable PyTorch implementation baseline, starting with Denoising Diffusion Probabilistic Models (DDPM). I will study the mathematical relation between unconditional and conditional diffusion and implement the methods based on their original equations. The package is intended to provide baseline models for new conditional tasks, support comparisons with prior work, and help implement methods from papers without public code. It will grow as I study more papers, including work on DDIM, score-based models, classifier-free guidance, flow matching, and drifting models.

## Implementation Principles

I develop this package by reading papers and writing code. I use the original equations and established public implementations as references. Generative AI may help me check equations, find bugs, or write difficult functions. I review AI-assisted code myself and compare it with the papers and reference implementations whenever possible. My goal is to build a codebase that I understand and can use with confidence in future research.

## Installation

Install from GitHub:

```bash
python -m pip install git+https://github.com/Pogjunan/diffusionPack.git
```

For local development:

```bash
git clone https://github.com/Pogjunan/diffusionPack.git
cd diffusionPack
python -m pip install -e .
```

Check the installation:

```bash
python -c "import diffusionpack; print(diffusionpack.__version__)"
```

## Current Status

The package structure is available. Model training and sampling are not yet implemented.
