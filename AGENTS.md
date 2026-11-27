# diffusionPack — code review rules

This is a personally developed, equation-first research package.
The author writes and reviews the core algorithms. AI helps with paper/code
comparison, limited drafts when requested, numerical checks and debugging.

Reference: Ho, Jain, Abbeel (2020), DDPM, original TensorFlow source:
https://github.com/hojonathanho/diffusion
Reference commit: 1e0dceb3b3495bbe19116a5e1b3596cd0706c543

Before changing research code:
- Identify the source equation, exact current function, shapes, dtype, and timestep rule.
- Do not change dataset membership, target, normalization, schedule, objective,
  checkpoint choice or scientific claims without an explicit user request.
- Keep the CIFAR DDPM settings as reference: T=1000, linear beta 0.0001 to
  0.02, epsilon MSE, fixedlarge reverse variance, 16x16 self-attention.
- Do not silently add DDIM, v-prediction, score SDE, conditioning or new losses.
- Prefer simple file names and short variable names. No new leading-underscore
  custom function or variable names. Do not add unrelated modules.
- Compare the original TensorFlow operation with the PyTorch implementation;
  a matching name or shape is not proof of equivalence.
- Independently verify at least one tiny numerical example; include an
  intentionally incorrect case that must fail.
- Run: python -m unittest discover -s test -v
- Report PASS, FAIL, or NOT RUN, with the actual executed command and evidence.
  A test pass does not prove generated image quality or exact reproduction.
- Do not execute long GPU jobs or commit additional changes without permission.

Focused Codex review guides (adapted from the provided Research PStack):
.agents/skills/research-math-code-audit/SKILL.md
.agents/skills/research-diffusion-audit/SKILL.md
