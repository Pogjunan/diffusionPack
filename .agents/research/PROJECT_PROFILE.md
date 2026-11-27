# diffusionPack research contract

Reviewed baseline: `28d9a933690825232c9a33f67b747cca8f65fb2d` on 2026-11-23 UTC.
The requested change publishes the existing 30-image DDPM Light gallery and audits task implementations.
No dataset split, diffusion parameterization, noise schedule, training objective, or existing checkpoint is changed.

The common parameterization is epsilon prediction, linear beta from 0.0001 to 0.02, and 1,000 discrete DDPM steps.
Unconditional training normalizes image channels to [-1, 1]. Restoration uses paired degraded and clean images.
Prediction uses fixed-cadence historical maps, explicit observation availability, a configured issue-time lag, and a single positive lead.
Prediction normalization uses a caller-supplied finite physical range shared by the map channels.

The existing Light gallery used the CIFAR-10 training split, batch 4, 20 cumulative epochs, 250,000 steps, EMA, and seed 42 on L40S.
Its original training Git revision was not recorded. Retained local source hashes and the checkpoint hash are recorded in gallery provenance.
The checkpoint remains local. Published artifacts are generated images, documentation, tests, and audit evidence.

Real SSI dataset identity, actual availability rules, spatial registration, chronological split membership, physical range, and evaluation protocol are UNKNOWN.
Real restoration training performance, SSI forecasting accuracy, forecast calibration, and CIFAR-10 FID are NOT RUN in this review.
Synthetic checks establish bounded execution and numerical properties only. They do not establish scientific performance.

Validation scope: existing unit tests; original/light forward and backward passes for all three tasks; short conditional training and resume; full 1,000-step conditional sampling; forecast output isolation; gallery file integrity.
No full training rerun or model selection is authorized or performed as part of this audit.
