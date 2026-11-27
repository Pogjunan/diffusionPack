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

## Forecasting with SSI maps

Download [run_prediction.py](run_prediction.py) and set the grid size, channel multipliers and physical value range. Each row in `train.csv` or `test.csv` contains `timestamp,path,available_at` for one NumPy grid. Both times include a UTC offset. Split the series by time before training.

`history_lag_minutes` is the interval from the last input map to forecast issue. `lead_minutes` runs from issue to target. With both set to 30, an issue at 12:00 uses maps through 11:30 and predicts 12:30. Each input must be available by 12:00.

Run `python run_prediction.py` with `mode = "train"` or `mode = "inference"`. Set `samples` to draw several forecasts per issue. Each inference run saves its grids and `predictions.csv` in a separate folder.
