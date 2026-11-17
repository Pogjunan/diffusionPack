# diffusionPack

**An equation-first, research-oriented Python package for diffusion and related generative models.**

diffusionPack is a personal research and learning project built to understand generative modeling from its mathematical foundations and turn that understanding into reusable PyTorch implementations.

The project will grow **one paper and one verified component at a time**. It starts with Denoising Diffusion Probabilistic Models (DDPM) and is intended to expand toward conditional generation, alternative sampling methods, score-based modeling, and other generative frameworks as they are studied.

## Why this project exists

Reimplementing a diffusion model for every new research problem makes it difficult to tell whether an unexpected result comes from the mathematics, the model architecture, the implementation, or the experimental setup.

diffusionPack aims to provide a consistent baseline for future research:

- **Understand the equations.** Connect every core operation to its mathematical definition and primary reference.
- **Build reusable baselines.** Keep validated components reusable across datasets, conditioning strategies, and research projects.
- **Make failures easier to investigate.** Separate data processing, noise schedules, prediction targets, training objectives, and sampling procedures.
- **Compare methods fairly.** Make the differences between prior work and new model designs explicit.
- **Learn by implementing.** Extend the package while studying research papers, rather than adding methods only for the sake of a large feature list.

Correct equations and executable code do not, by themselves, guarantee successful training or high-quality generation. The goal is a framework in which assumptions and implementation choices can be inspected and tested.

## Development principles

1. **Paper-first and equation-first.** Use original papers and, where available, established author-released repositories as primary references.
2. **Implement the core algorithms personally.** AI tools may assist with mathematical review, debugging, documentation, and code comparison; generated code is not accepted as correct without independent review.
3. **Verify before extending.** Establish a working unconditional baseline before introducing additional conditioning or more complex architectures.
4. **Keep components distinguishable.** Treat the diffusion process, prediction parameterization, backbone, objective, conditioning, and sampler as separate design choices.
5. **Document deviations.** When an implementation differs from the original paper, record the change and its reason.

## Project status

**Early development — packaging scaffold only.** Installation and importing the package work. DDPM training, sampling, and the proposed model APIs are **not implemented in this repository yet**. This README describes the intended direction, not a claim that these features are ready.

## Installation

Install directly from GitHub:

```bash
python -m pip install "git+https://github.com/Pogjunan/diffusionPack.git"
```

Or clone the repository and install it in editable mode for development:

```bash
git clone https://github.com/Pogjunan/diffusionPack.git
cd diffusionPack
python -m pip install -e .
```

Check the current installation:

```bash
python -c "import diffusionpack; print(diffusionpack.__version__)"
```

The package name used by Python is `diffusionpack`; the GitHub repository is named `diffusionPack`.

## How to use (planned)

The intended interface will select unconditional or conditional execution through an explicit configuration and offer straightforward training and inference routines. **The following is a design sketch, not executable code yet:**

```python
# Planned API — not implemented yet
from diffusionpack import DiffusionPack

model = DiffusionPack(mode="unconditional", backbone="DDPMUNet")
model.train(dataset)
samples = model.sample(num_samples=16)
```

The first runnable example will cover unconditional DDPM training and sampling. Conditional experiments will reuse the same diffusion core while providing a condition to the denoiser at training and sampling time.

## Planned organization

```text
diffusionpack/
├── config.py
├── diffusion.py
├── backbone/
│   ├── unet.py
│   ├── ddpm_unet.py
│   └── self_attention_unet.py
├── conditioning/
│   ├── unconditional.py
│   └── conditional.py
├── parameterizations/
│   └── epsilon.py
├── samplers/
│   └── ddpm.py
├── objectives/
│   └── epsilon_mse.py
└── main/
    ├── train.py
    ├── inference.py
    └── trainNinference.py
```

This is the **planned architecture**, not a list of completed modules. The initial release will remain intentionally small.

## Roadmap

- **v0.1 — Unconditional DDPM:** implement and validate the original forward process, noise-prediction objective, U-Net, and reverse sampling.
- **v0.2 — Conditional DDPM:** add explicit conditioning and compare against the unconditional baseline.
- **v0.3 — DDIM:** introduce a different sampler while reusing a compatible trained denoiser.
- **Later — Score-based generative modeling:** score matching, Langevin dynamics, and reverse-time SDEs.
- **Later — Conditional guidance and related approaches:** classifier-free guidance (CFG), flow matching, and Drifting-model research directions when studied.

Roadmap items are intentions, not implemented features or promised release dates.

## Primary references

- Ho, Jain, and Abbeel (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — [authors' reference code](https://github.com/hojonathanho/diffusion)
- Song et al. (2021), [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — [authors' PyTorch code](https://github.com/yang-song/score_sde_pytorch)
- Song, Meng, and Ermon (2020), [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- Saharia et al. (2022), [Palette: Image-to-Image Diffusion Models](https://arxiv.org/abs/2111.05826)

**Maintained as an independent learning and research project.** It is not an official implementation of the papers listed above.
