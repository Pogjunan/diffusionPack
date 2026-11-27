---
name: research-math-code-audit
description: Check equations, tensor dimensions, losses and gradients against code.
---

# Math and code review

Adapted from the user's Research PStack archive.

Read the paper equation and original repository function first.
Define every relevant symbol, shape, axis, dtype and time index.
Check exact equality versus approximation and state assumptions.
Compare with a hand-computed independent small example and a deliberately
wrong negative control. Check gradients and reduction axes where relevant.
Give each check a PASS, FAIL or NOT RUN with evidence. A numerical match on
one example is not a guarantee of correctness. Do not silently modify
experimental definitions or datasets to make tests pass.
