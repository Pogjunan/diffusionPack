---
name: research-diffusion-audit
description: Review DDPM and later diffusion formulas and implementation choices.
---

# Diffusion code review

Adapted from the user's Research PStack archive.

Follow paper -> original author code -> local function -> independent test.
For DDPM, verify beta/alpha schedules, timestep indices, q_sample, epsilon
targets, loss reduction, posterior, reverse mean/variance, clipping and the
last no-noise step when implemented. Confirm whether model output means
epsilon, x0, v or score, without silently converting.
For conditional models verify conditioning availability and target formation.
Check sampling and learning separately; low loss and correct equations do
not establish generated sample quality. Report PASS, FAIL and NOT RUN.
