# diffusionPack

A Python package for studying diffusion models and building reusable baselines in PyTorch.

## Purpose

diffusionPack is a personal research project. I study diffusion models by reading the original papers and implementing their equations. My goal is to build a common codebase for future research. I want to compare existing methods with new models and make errors in training and sampling easier to trace.

## Approach

I write the core algorithms myself. I use the original papers and the authors' code as references. AI tools help me check equations, review code, and find errors, but they do not replace my own implementation.

The project starts with DDPM. I plan to add conditional diffusion, DDIM, score-based models, and related methods as I study them.

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

## Current status

The package structure is available. Model training and sampling are not yet implemented.
