# diffusionPack

## Abstract

diffusionPack is a PyTorch package for studying generative models and developing baselines for future research.

## Implementation Principles

The implementations follow original research papers and the authors' released code where available. The project covers DDPM and DDIM, conditional restoration and prediction, and concludes with the planned implementation of Drifting Models.

I have verified the current diffusion implementations against their mathematical formulations within the scope of my understanding. They are intended as research baselines for conditional generation, time-series forecasting, and the development of new diffusion methods. The priority is implementing the methods described in the original papers rather than optimizing GPU performance.

## Models

| Model | Description |
| --- | --- |
| DDPM | Unconditional generation |
| DDIM | Accelerated sampling |
| Restoration | Conditional image restoration |
| Prediction | Spatiotemporal forecasting |
| Drifting Models | Planned |

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

Clone the repository for experiments and method development.

```bash
git clone https://github.com/Pogjunan/diffusionPack.git
cd diffusionPack
python -m pip install -e .
```

In [run.py](run.py), select `task` (`unconditional`, `restore`, or `prediction`) and `action` (`train` or `inference`), configure the dataset and model, and run:

```bash
python run.py
```

For inference with DDIM, set `sampler="ddim"`, `sampling_steps=50`, and `eta=0.0` in the corresponding inference configuration.

## Update

If the dependencies are already installed, update only diffusionPack:

```bash
python -m pip install --upgrade --force-reinstall --no-deps git+https://github.com/Pogjunan/diffusionPack.git
```

For an editable installation, update the local repository instead:

```bash
git pull --ff-only
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
