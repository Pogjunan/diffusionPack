# diffusionPack

## Abstract

diffusionPack is a PyTorch package for developing and studying diffusion-based generative models. It starts with Denoising Diffusion Probabilistic Models (DDPM) and aims to provide reusable implementations for unconditional and conditional generation, reducing the need to rebuild baselines for each research project. The next implementation is DDIM. Later work will cover [Drifting Models](https://arxiv.org/abs/2602.04770) and SCRD.

## Implementation Principles

Implementations are developed with reference to original research papers and publicly available author code. The focus is on understanding the underlying mathematics, keeping the implementation readable, and checking numerical behavior before using models in research. The package is intended to support reproduction, comparison, and extension of generative modeling methods.

## Models

| Task | Input | Output |
| --- | --- | --- |
| Generation | Gaussian noise | Image |
| Restoration | Degraded image | Clean image |
| Prediction | Past spatial maps | One future map |

Choose `model = "light"` or `model = "original"` for any task.

[![Unconditional DDPM Light samples](examples/light/grid.png)](examples/light)

## References

- DDPM. [Paper](https://arxiv.org/abs/2006.11239) · [Diffusion equations in TensorFlow](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py) · [U-Net](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py)
- Restoration uses image conditioning from [Palette, Section 3](https://arxiv.org/html/2111.05826v2#S3). [Independent PyTorch implementation](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/136b29f58d0af6e5db9f3655d2891f5a855fcdaa/models/network.py)
- Prediction extends the same denoiser to a fixed history of spatial maps. [Forecast dataset](diffusionpack/main/prediction_data.py) · [Related work on past-frame conditioning](https://arxiv.org/abs/2205.09853)

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

Run `python run_prediction.py` with `mode = "train"` or `mode = "inference"`. Set `samples` to draw several forecasts per issue. Each call saves its forecast maps in a separate folder.
