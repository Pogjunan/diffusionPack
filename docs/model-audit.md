# DDPM task implementation and Research PStack audit

Audit date: 2026-11-23 UTC. Baseline: [`28d9a93`](https://github.com/Pogjunan/diffusionPack/tree/28d9a933690825232c9a33f67b747cca8f65fb2d).
This review follows the supplied Research PStack research-mode, diffusion, math-to-code, time-series, and reproducibility procedures.
The [research contract](../.agents/research/PROJECT_PROFILE.md) separates verified properties from unknown data and evaluation assumptions.

## Three tasks share two size presets

| User-facing entry | Mathematical target | Condition | Implementation |
| --- | --- | --- | --- |
| Unconditional | Image distribution p(x) | None | `DDPMUNet`, `ddpm_loss`, `p_sample_loop` |
| Restoration | Clean image distribution p(y given c) | Degraded image c | `RestoreUNet`, `restore_loss`, `restore_sample` |
| Prediction | Future map distribution p(y_future given history) | Past maps concatenated in time order | `PredictionUNet`, `prediction_loss`, `prediction_sample` |
| Light | Same target as the selected task | Same condition as the selected task | `model_settings("light")` |

Light is a model-size preset, not a fourth conditioning mechanism. The six task-size combinations run through the same DDPM core.
Original uses base channels 128 and two encoder residual blocks per resolution; Light uses 64 and one.
The following counts are measured at 32×32, with RGB generation, RGB restoration, and five single-channel history maps for prediction.

| Task | Light parameters | Original parameters | Input channels | Epsilon output channels |
| --- | ---: | ---: | ---: | ---: |
| Unconditional | 6,079,811 | 35,746,307 | 3 | 3 |
| Restoration | 6,081,539 | 35,749,763 | 6 | 3 |
| Prediction | 6,080,385 | 35,747,457 | 6 | 1 |

Weights are task-specific. The existing unconditional RGB checkpoint is not a trained restoration or SSI checkpoint.
Each conditional task needs its own training data and trained weights.

## Forward noise, loss, and sampling

For target y and diffusion index t, the shared core computes:

$$\beta_t=\operatorname{linspace}(0.0001,0.02,1000)_t,\quad \alpha_t=1-\beta_t,\quad \bar\alpha_t=\prod_{s=0}^{t}\alpha_s$$

$$y_t=\sqrt{\bar\alpha_t}\,y+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad L=\operatorname{mean}_{B,C,H,W}(\epsilon_\theta(y_t,t,c)-\epsilon)^2.$$

The unconditional denoiser omits c. Conditional wrappers bind c while calling the same loss and sampler.
Only the target receives diffusion noise. `RestoreUNet` concatenates `[noisy_target, condition]` before the first convolution and emits target-channel epsilon.
`PredictionUNet` inherits this operation and sets condition channels to `history_steps * map_channels`.

The sampler visits t=999 through t=0, reconstructs and clips the estimated clean target to [-1, 1], and uses the posterior mean with fixed-large reverse variance based on beta.
It adds Gaussian noise at t>0 and no added reverse noise at t=0. The condition remains fixed throughout all steps.
The tests check known-noise forward values, reconstruction, final-step behavior, conditional gradients, and deliberately incorrect inputs.
This is an epsilon-prediction DDPM implementation, following the framework in [Ho et al.](https://arxiv.org/abs/2006.11239).

Restoration follows the conditional image-to-image approach used by [Palette](https://arxiv.org/abs/2111.05826).
Prediction uses past-frame conditioning, a concept also used in [MCVD](https://arxiv.org/abs/2205.09853).
This repository does not reproduce the full Palette or MCVD architecture or experiments. It has no MCVD masking scheme, classifier-free guidance, or multi-frame autoregressive video generator.

## Prediction is reusable for spatial maps, with SSI defaults

The backbone contains no solar-radiation equation, solar-position feature, or SSI-specific output layer.
The dataloader accepts NumPy grids with shape `[C,H,W]` or single-channel `[H,W]` and stacks a fixed number of observations into channels.
`history_steps`, `map_channels`, cadence, issue lag, lead, resolution, units, and physical range are configurable.
The default five 30-minute maps, one output map, and `W/m2` unit are SSI-oriented choices.

This makes prediction reusable for fixed-cadence, registered 2D spatial fields after domain-appropriate preprocessing and training.
It is not a general predictor for arbitrary tabular data, text, irregular sequences, variable-length history, or several future horizons in one call.
Input and target share the configured map channels and square spatial resolution. One global min/max range is shared across channels.
Missing history is rejected rather than imputed. Geospatial reprojection, per-variable scaling, exogenous covariates, and cross-split validation are not implemented here.
The forecast lead is checkpoint metadata, not a learned lead embedding; a checkpoint is trained for one configured lead.

For issue time I, lag L, cadence D, history length K, and lead h, the window is:

$$[I-L-(K-1)D,\ldots,I-L],\qquad \text{target time}=I+h.$$

Every input must have `available_at <= I`. With L=30 and h=30 minutes, an issue at 12:00 uses history through 11:30 and targets 12:30.
The latest-observation-to-target gap is 60 minutes in this example. Lag defaults to zero and must match actual data latency.
The reviewed commit includes the lag support missing from the earlier patch.
Forecast inference now creates a separate run directory, records checkpoint SHA and per-member seed, and supports several samples per issue.

## Executed evidence

Hardware: NVIDIA L40S, GPU 0. Python 3.10.19, PyTorch 2.10.0+cu128. CUDA checks are not RTX 3060 Ti validation.

| Property | Command or evidence | Result | Scope |
| --- | --- | --- | --- |
| Baseline tests | `python -m unittest discover -s test -v`; [log](validation/baseline-tests.log) | PASS, 55 tests | Includes independent numerical and negative-control tests |
| Regression suite after input-error fix | Same command; [log](validation/tests.log) | PASS, 56 tests | Includes invalid config rejection |
| Six task-size combinations | `python -m scripts.audit_modes --device cuda:0`; [JSON](validation/modes-smoke.json) | PASS | Finite forward loss, gradients, and conditional-channel gradients |
| Light restoration pipeline | Same audit command | PASS | Synthetic paired images, 2 steps then resume to 3, full 1,000-step sampling, PNG output |
| Light prediction pipeline | Same audit command | PASS | Synthetic maps, 2 steps then resume to 3, 10-minute availability delay with 30-minute lag |
| Prediction ensemble behavior | Same audit command | PASS | Two members per run, two runs, distinct members, repeated seed matches, no overwritten output |
| Actual Light gallery | `python -m scripts.verify_gallery`; [JSON](validation/gallery-integrity.json) | PASS | 30 unique RGB 32×32 files, original grid correspondence, 31 SHA checks, wrong-hash negative control |
| Generic PStack `verify-run` schema | [Raw result](validation/pstack-verify-run.json) | FAIL: schema not applicable | It requires a train/val/test evaluation split and split-based checkpoint selection; this gallery has neither. All 31 artifact hashes matched. No artificial evaluation split was assigned to force a pass. |
| Real SSI batch and cross-split leakage | No SSI manifest/data supplied | NOT RUN | Synthetic time-axis tests do not establish real availability or split integrity |
| FID, restoration PSNR/SSIM, forecasting RMSE/CRPS/calibration | No evaluation protocol or real conditional predictions supplied | NOT RUN | No quantitative performance claim |
| Full training of six configurations | Not part of this review | NOT RUN | Only the prior unconditional Light run has the published 20 epoch output |
| Independent training reproduction | Original run has no recorded Git commit and checkpoint is local | NOT RUN | Gallery hashes establish identity, not full reproducibility |

To reproduce the bounded code checks from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s test -v
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m scripts.audit_modes --device cuda:0
python -m scripts.verify_gallery
```

The audit uses temporary synthetic data and checkpoints and does not modify real training outputs.
It deliberately checks original/light gradients but runs full conditional pipeline smoke tests only for Light.

## Remaining work before performance claims

An invalid training configuration previously called an out-of-scope `parser.error`, causing `NameError`.
The update raises `ValueError` and adds a regression test; valid training mathematics and settings are unchanged.

Restoration inference still reuses output filenames when the same output folder is used. Use distinct folders for checkpoints and seeds.
Prediction resume fingerprints the CSV manifest, not each underlying `.npy` file. Preserve and version those files separately.
Checkpoint state restores weights, optimizer, scheduler, AMP, and EMA, but not the exact RNG and dataloader position. Resumption is not bitwise continuation.
Changing batch size on resume can change the cumulative step target computed from an epoch budget. Keep batch size fixed for the audited epoch interpretation.

For SSI, establish the actual unit/range, missing-value policy, timestamp semantics, spatial alignment, and chronological train/validation/test manifests.
Then compare against a persistence baseline on a held-out period with an explicit map/pixel aggregation and forecast ensemble metric.
The current evidence supports implementation checks and reusable spatial forecasting structure. It does not establish that all four user-facing entries have completed scientific validation.
