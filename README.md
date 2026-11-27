# diffusionPack

## Abstract

diffusionPack is a PyTorch package for developing and studying diffusion-based generative models. It starts with Denoising Diffusion Probabilistic Models (DDPM) and aims to provide reusable implementations for unconditional and conditional generation, reducing the need to rebuild baselines for each research project. The next implementation is DDIM. Later work will cover [Drifting Models](https://arxiv.org/abs/2602.04770).

## Implementation Principles

Implementations are developed with reference to original research papers and publicly available author code. The focus is on understanding the underlying mathematics, keeping the implementation readable, and checking numerical behavior before using models in research. The package is intended to support reproduction, comparison, and extension of generative modeling methods.

## Models

| Task | Input | Output |
| --- | --- | --- |
| Generation | Gaussian noise | Image |
| Restoration | Degraded image | Clean image |
| Prediction | Past spatial maps | One or more future maps |

Choose `model = "original"` for the CIFAR-10 U-Net defaults or `model = "light"` for a smaller U-Net. All tasks use the same 1,000-step DDPM reverse schedule. Unconditional generation keeps integer timestep conditioning. Restoration and Prediction use continuous gamma conditioning during training and AdaGN in the residual blocks.

In [run.py](run.py), set `task` to `"unconditional"`, `"restore"` or `"prediction"` and `action` to `"train"` or `"inference"`. Run `python run.py`.

[![Unconditional DDPM Light samples](examples/light/grid.png)](examples/light)

## References

- DDPM. [Paper](https://arxiv.org/abs/2006.11239) · [Diffusion equations in TensorFlow](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py) · [U-Net](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py)
- Restoration uses Palette-style channel concatenation and conditional noise prediction, with [SR3-style gamma sampling](https://arxiv.org/abs/2104.07636) (uniform step interval, then uniform interpolation between adjacent cumulative alphas) and [AdaGN](https://arxiv.org/abs/2105.05233). [Palette, Section 3](https://arxiv.org/html/2111.05826v2#S3) · [Independent PyTorch implementation](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/136b29f58d0af6e5db9f3655d2891f5a855fcdaa/models/network.py)
- Prediction extends the same conditional denoiser to a fixed history and joint future maps. [Forecast dataset](diffusionpack/main/prediction_data.py) · [Related work on past-frame conditioning](https://arxiv.org/abs/2205.09853)

## Image Resolution

Set one square `resolution` per training run. The U-Net automatically adds
downsampling levels for large inputs when `channel_mult` is not supplied.
The unconditional 32x32 architecture is unchanged.

| Resolution | Channel multipliers | Smallest feature map |
| --- | --- | --- |
| 32 | (1, 2, 2, 2) | 4x4 |
| 512 | (1, 2, 2, 2, 2, 2) | 16x16 |
| 1024 | (1, 2, 2, 2, 2, 2, 2) | 16x16 |

Default full attention remains at 16x16 and in the middle block.
Explicitly choosing too few downsampling levels is rejected if the middle
attention would exceed 32x32.

```python
TrainConfig(dataset="folder", resolution=512, model="light",
            base_channels=32, batch_size=1)
RestoreTrainConfig(dataset="paired", resolution=512,
                   base_channels=32, batch_size=1)
PredictionTrainConfig(resolution=1024, base_channels=32, batch_size=1)
```

Unconditional folder images are resized and center-cropped; restoration
pairs are resized together. Forecast `.npy` maps must already match the
configured resolution and are never resized. The trained resolution and
architecture are saved in each checkpoint; a 32x32 checkpoint cannot be
reused with a larger architecture. Large-image training still needs
substantial GPU memory despite bounded attention resolution. Consider
smaller batches, mixed precision, or training on spatial crops.

## Installation

```bash
pip install git+https://github.com/Pogjunan/diffusionPack.git
```

## Update

```bash
pip install --upgrade --force-reinstall git+https://github.com/Pogjunan/diffusionPack.git
```

## Forecasting with SSI maps

Set the grid size, channel multipliers and scaling bounds in [prediction_config.py](diffusionpack/prediction_config.py) or pass them to `PredictionTrainConfig` in [run_prediction.py](run_prediction.py). Choose the bounds from the training data or known physical limits. Forecasts can exceed these bounds. Each row in `train.csv` or `test.csv` contains `timestamp,path,available_at` for one NumPy grid. Use ISO 8601 timestamps with a UTC offset (e.g., `2026-01-01T12:00:00+09:00`). Times without an offset are treated as UTC. All times are converted to UTC. Split the series by time before training.

`history_lag_minutes` is the interval from the last input map to forecast issue. `lead_minutes` runs from issue to the first target; later targets are spaced by `cadence_minutes`. With both set to 30, an issue at 12:00 uses maps through 11:30 and predicts 12:30. Each input must be available by 12:00.

Set `history_steps` and `future_steps` in `PredictionTrainConfig`. For example, five past maps and four future maps with one channel each:

```python
PredictionTrainConfig(history_steps=5, future_steps=4, map_channels=1,
                      cadence_minutes=30, lead_minutes=30)
```

Set the scaling bounds and `csv_path` before training. The model receives 9 channels and predicts 4 noise channels. It generates the four future maps jointly, then saves one `.npy` file per lead time. Maps from the same ensemble draw share a sample index. Inference does not require future targets. The default `future_steps=1` retains single-map prediction.

Run `python run_prediction.py` first with `mode = "check"`, then with `mode = "train"` or `mode = "inference"`. Set `samples` to draw several forecasts per issue. Each call saves its forecast maps in a separate folder. When resuming training, use the same training CSV and data; their contents are not checked against the checkpoint.

## Conditional training and checkpoints

Gamma is sampled between `alpha_bars[t]` and `alpha_bars_prev[t]` during training. The same gamma is used to create the noisy target and as the input to the existing sinusoidal embedding. During sampling, the model receives the fixed `alpha_bars[t]` at each discrete DDPM step. The schedule and reverse equations are unchanged. This is not a full SR3, Palette or MCVD reproduction.

The new conditional models need new checkpoints because the noise-level input and AdaGN projection sizes differ from the previous implementation. Unconditional checkpoint shapes are unchanged. The training/inference schedules are not independently configurable yet. Run `python -m pytest -q` for structural and CPU integration tests; actual forecast accuracy and GPU training require separate experiments.
