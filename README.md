# diffusionPack

## Abstract

diffusionPack is a PyTorch package for developing and studying diffusion-based generative models. It implements Denoising Diffusion Probabilistic Models (DDPM) and DDIM sampling, and aims to provide reusable implementations for unconditional and conditional generation, reducing the need to rebuild baselines for each research project. Later work will cover [Drifting Models](https://arxiv.org/abs/2602.04770).

## Implementation Principles

The implementations follow original research papers and the released code where available. The project covers DDPM and DDIM, conditional restoration and prediction, and Drifting Models (plan).

I have verified the current diffusion implementations against these mathematical formulations within the scope of my understanding. I intended as research baselines for conditional generation, time-series generation(etc. forecasting), and the development of new diffusion methods. The priority is implementing the methods described in the original papers.

## Models

| Method | Description |
| --- | --- |
| DDPM | Markovian diffusion and stochastic generation |
| DDIM | Non-Markovian diffusion and accelerated sampling |
| Restoration | Conditional diffusion for image-to-image restoration |
| Prediction | Conditional diffusion for spatiotemporal forecasting |
| Drifting Models | Planned implementation |

The following samples were generated using the light DDPM model on an NVIDIA RTX 3060.

[![Unconditional DDPM Light samples](examples/light/grid.png)](examples/light)

## Installation

Install the package from GitHub to reuse its components in other research projects.

```bash
python -m pip install git+https://github.com/Pogjunan/diffusionPack.git
```

```python
from diffusionpack.backbone.restore_unet import RestoreUNet
from diffusionpack.restore import restore_loss, restore_sample
from diffusionpack.ddim import ddim_sample
```

## Working with the Source

```bash
git clone https://github.com/Pogjunan/diffusionPack.git
cd diffusionPack
```

Install the dependencies listed in [setup.py](setup.py) in your Python environment before running the examples. Edit the shared diffusion schedule in [config.py](diffusionpack/config.py):

```python
num_diff_steps = 1000
beta_start = 0.0001
beta_end = 0.02
```

Keep this schedule unchanged between training and inference. DDIM reuses the trained noise predictor; `sampling_steps` controls inference only.

## Examples

### 1. Unconditional image generation (DDPM)

Put your images in `data/images/train/` and keep `valid/` and `test/` separate for evaluation. To customize image loading or the U-Net, edit [train.py](diffusionpack/main/train.py) (`get_dataset`, `model_settings`) or [ddpm_unet.py](diffusionpack/backbone/ddpm_unet.py).

Set these values in [run.py](run.py):

```python
task = "unconditional"
action = "train"

unconditional_train = TrainConfig(
    dataset="folder", data_root="data/images/train",
    model="light", resolution=32, batch_size=8, epochs=30,
    output="runs/images",
)
unconditional_inference = InferenceConfig(
    checkpoint="runs/images/latest.pt", sampler="ddpm",
    samples=8, output="runs/images/samples.png",
)
```

Run `python run.py` to train, then set `action = "inference"` and run it again to generate images. Use `valid/` and `test/` for separate evaluation; the current training loop does not evaluate them automatically.

### 2. Paired image restoration (DDIM-40)

Prepare matching degraded and clean images in `data/restore/train/condition/` and `data/restore/train/target/`. Use the same folder structure for `valid/` and `test/`. Customize [restore_data.py](diffusionpack/main/restore_data.py) and [restore_unet.py](diffusionpack/backbone/restore_unet.py) if needed.

In [run.py](run.py):

```python
task = "restore"
action = "train"

restore_train = RestoreTrainConfig(
    dataset="paired", condition_dir="data/restore/train/condition",
    target_dir="data/restore/train/target", resolution=128,
    model="light", base_channels=32, batch_size=2,
    epochs=30, output="runs/restore",
)
restore_inference = RestoreInferenceConfig(
    checkpoint="runs/restore/latest.pt",
    condition_path="data/restore/test/condition",
    sampler="ddim", sampling_steps=40, eta=0.0,
    output="runs/restore/images",
)
```

Run `python run.py` with `action = "train"`, then `action = "inference"`. Compare the restored images with `data/restore/test/target/`; use `valid/condition/` and `valid/target/` for validation separately.

### 3. Solar irradiance forecasting (DDIM-50)

Predict one 512 × 512 SSI map 30 minutes ahead from five past maps. Save the original numerical grids as `.npy` files and prepare chronological `train.csv`, `valid.csv`, and `test.csv` splits under `data/ssi/`. Each CSV lists observation time, grid path (relative to the CSV), and availability time. For example, six consecutive rows of `test.csv` could be:

```csv
timestamp,path,available_at
2021-02-19T01:30:00Z,maps/0130.npy,2021-02-19T01:30:00Z
2021-02-19T02:00:00Z,maps/0200.npy,2021-02-19T02:00:00Z
2021-02-19T02:30:00Z,maps/0230.npy,2021-02-19T02:30:00Z
2021-02-19T03:00:00Z,maps/0300.npy,2021-02-19T03:00:00Z
2021-02-19T03:30:00Z,maps/0330.npy,2021-02-19T03:30:00Z
2021-02-19T04:00:00Z,maps/0400.npy,2021-02-19T04:00:00Z
```

Use actual timestamps, availability times and physical scaling bounds for your dataset. Customize [prediction_data.py](diffusionpack/main/prediction_data.py) and [prediction_unet.py](diffusionpack/backbone/prediction_unet.py) for a different forecasting task.

In [run_prediction.py](run_prediction.py):

```python
mode = "check"

train_config = PredictionTrainConfig(
    csv_path="data/ssi/train.csv", resolution=512,
    history_steps=5, future_steps=1, map_channels=1,
    cadence_minutes=30, lead_minutes=30,
    value_min=0.0, value_max=1200.0,  # Example scaling bounds
    backbone="ddpm", model="light", base_channels=32,
    batch_size=1, epochs=30, output="runs/ssi",
)
valid_csv = "data/ssi/valid.csv"
test_csv = "data/ssi/test.csv"

inference_config = PredictionInferenceConfig(
    checkpoint="runs/ssi/latest.pt", csv_path=test_csv,
    origin_time="2021-02-19T03:30:00Z",
    sampler="ddim", sampling_steps=50, eta=0.0,
    output="runs/ssi/forecasts",
)
```

Run `python run_prediction.py` with `mode = "check"`, then `"train"`, then `"inference"`. The check verifies the temporal splits and a training batch; validation metrics are not calculated automatically. The example generates the 04:00 forecast from the five observations through 03:30. The output is saved as a `.npy` map. Change `sampling_steps` to 40 to use DDIM-40 without retraining.

The SCRD backbone uses its existing DDPM sampler; the DDIM settings above apply to the basic Prediction U-Net.

## Update

If the dependencies are already installed, update only diffusionPack:

```bash
python -m pip install --upgrade --force-reinstall --no-deps git+https://github.com/Pogjunan/diffusionPack.git
```

For a cloned repository, update the source:

```bash
git pull
```

## References

- **DDPM** — [Paper](https://arxiv.org/abs/2006.11239) · [Diffusion equations in TensorFlow](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py) · [U-Net](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py)
- **DDIM** — [Paper](https://arxiv.org/abs/2010.02502) · [Author code](https://github.com/ermongroup/ddim) · [Sampler](diffusionpack/ddim.py)
- **Restoration** — [Palette](https://arxiv.org/abs/2111.05826) · [SR3](https://arxiv.org/abs/2104.07636) · [AdaGN](https://arxiv.org/abs/2105.05233) · [Independent PyTorch implementation](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
- **Prediction** — [MCVD](https://arxiv.org/abs/2205.09853)
- **Drifting Models** — [Paper](https://arxiv.org/abs/2602.04770)

### Application Example: SCRD

[SCRD](https://doi.org/10.1109/LGRS.2024.3486112) — Li et al., *IEEE Geoscience and Remote Sensing Letters*, 2024.

I independently implemented SCRD for precipitation nowcasting and adapted it to forecasting 512 × 512 surface solar irradiance maps.

*Figures will be added.*

<!--
![Original SCRD results](examples/scrd/paper_results.png)

*Results from the original paper.*

![SSI forecasting example](examples/scrd/ssi_example.png)

*Five observed SSI maps and the 30-min target (GT). The SCRD panel is an illustrative mock-up, not a model output; it will be replaced with experimental results.*
-->

## Discussion: Image Resolution

Appendix B of DDPM uses four U-Net resolution levels for 32 × 32 images and six for 256 × 256 images, but does not specify a general rule for arbitrary resolutions. Inspired by this design, I defined the following default depth settings for different input sizes. These are my own settings, rather than exact reproductions of every architecture in the paper.

| Input size | U-Net levels | Feature map sizes along the downsampling path |
| --- | ---: | --- |
| 32 × 32 | 4 | 32 → 16 → 8 → 4 |
| 256 × 256 | 5 | 256 → 128 → 64 → 32 → 16 |
| 512 × 512 | 6 | 512 → 256 → 128 → 64 → 32 → 16 |
| 1024 × 1024 | 7 | 1024 → 512 → 256 → 128 → 64 → 32 → 16 |

The depth and channel widths can be changed using `channel_mult` when constructing the U-Net.
