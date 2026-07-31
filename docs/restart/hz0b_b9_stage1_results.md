# HZ-0B B9 Stage 1: Unfreeze Last HZ-0A Block -- Results

Date: 2026-07-30. First B9 work: per the plan's own staging ("1. unfreeze
only memory-adjacent projections"), unfroze HZ-0A's LAST block (index 30
of 31, 9,449,472 params -- the block immediately upstream of where B6/B7's
memory injection happens) and fine-tuned it jointly with the B7 write
controller, against the real, previously-frozen checkpoint. Same task as
B7 (`scripts/hz0b_b9_stage1_finetune_probe.py`, reusing
`scripts/hz0b_b7_real_integration_probe.py`'s exact prompts/labels),
`confidence_scaled=True` (the validated B6/B7 fix), controller_lr=0.4
(B7's own best-working rate), a deliberately much smaller block_lr=1e-5
(a pretrained ~301M-scale weight, not a fresh random init).

## Result: general quality preserved, memory task got WORSE, not better

| B9's required measurement | Result |
| --- | --- |
| General val loss (no memory) | 2.474712 -> 2.473419 (**-0.05%**, unchanged) |
| Memory-task rank (write-then-read, held-out) | **1184.5 / 24576** (worse than B7's frozen-backbone baseline of 179.4-325.0) |
| Write controller gate weight norm | 1.3868 (comparable magnitude to B7's own trained values) |
| Memory interference (should_write=0 drift) | **0.000000** (exact -- the confidence-scaling fix holds even with a fine-tuned block) |

General language quality is genuinely preserved -- a real, clean result,
and the interference guarantee holds exactly even now that part of the
backbone is also training. But the memory-task result is a real
regression, not an improvement: rank 1184.5 is meaningfully worse than
what B7 achieved with the backbone fully frozen (179.4 at 1000 steps
without confidence-scaling, 325.0 at higher step count with it).

## Why, most likely: joint optimization is harder here, not easier

Train loss at step 2999 was **22.04** -- compare to B7's own confidence-
scaled run at similar hyperparameters (controller_lr=0.4), which reached
**~9.4 by step 2499**. Same task, same controller learning rate, more
than double the residual loss with the block unfrozen. The extra
trainable capacity did not make the task easier to optimize -- it made it
slower to converge, plausibly because:

- **A moving target**: the controller is learning to read hidden states
  produced by the last block, but that block's own output is also
  shifting under gradient descent at the same time -- the controller has
  to chase a representation that keeps changing, instead of fitting a
  fixed one.
- **Two learning rates on one loss surface**: `block_lr` (1e-5) and
  `controller_lr` (0.4) are four orders of magnitude apart, an untested
  combination -- the relative pacing between them was not tuned, only
  chosen to be "conservative for the pretrained weight, same-as-before
  for the controller."

Neither is confirmed here -- disclosed as the plausible, untested
explanation, not a settled finding.

## Honest read against B9's exit gate

B9's exit gate: *"Memory improvements survive limited fine-tuning without
destroying HZ-0A quality."*

- **"Without destroying HZ-0A quality"**: satisfied, cleanly (-0.05%,
  exact-zero interference).
- **"Memory improvements survive"**: not demonstrated -- there was no
  memory improvement to survive. Unfreezing the last block made the
  memory task harder to learn at this step budget and LR combination, not
  easier. This does not mean fine-tuning categorically can't help (the
  hyperparameter combination tried here is one point, not a sweep) -- it
  means this specific attempt didn't show the benefit B9 is looking for,
  and that's reported directly rather than reframed as a partial win.

## Untried, real next steps (not run in this pass)

- **Curriculum**: train the controller alone first (backbone fully
  frozen, matching B7 exactly) until it converges, THEN unfreeze the
  block and continue -- avoids the moving-target problem from the start
  rather than fighting it from step 0.
- **Much smaller or zero effective block movement early on**: a
  warmup schedule for `block_lr` (start at 0, ramp up) rather than a
  constant small rate from step 0.
- **More steps**: the joint run's own loss curve (22.04 at step 2999,
  still descending) had not remotely plateaued the way B7's own runs did
  -- unlike the GDN-3 tiny-scale investigation, this was NOT checked with
  a longer run before writing this up; a real, named gap, not silently
  skipped.
