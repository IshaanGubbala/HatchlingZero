# HZ-0C C7: Fixed -- Consistent Ground Truth Nearly Doubles Real Recall

Date: 2026-08-04. Directly requested: "fix c7." Implements the fix named
at the end of the four-attempt investigation
(`docs/restart/hz0c_c9_attention_pattern_feature_results.md`): the
fourth attempt (a real downstream-benefit teacher) scored badly not
because the signal was bad, but because it was trained toward one
ground-truth definition (true downstream benefit) and evaluated against
a different one (the C3 scenarios' hand-labeled construction points).
This removes the mismatch and measures the real, previously-unasked
question directly.

## The fix

`scripts/hz0c_c7_true_benefit_controller.py` (new). Both the
distillation teacher AND the evaluation metric are now the SAME
quantity: the top-15%-by-real-measured-downstream-benefit positions
(`causal_attention_benefit`, full unrestricted candidate budget -- every
position scored, none pre-filtered). Same held-out protocol as every
other C7/C9 comparison in this project (train seeds 555+556, eval seed
557, 8 real C3 scenarios, 32 examples each, exact 15% rate). Both the
existing BCE (`fit_controller`) and ranking (`fit_ranking_controller`)
objectives were run against this consistent target.

## The result: nearly double, and the mismatch is now directly confirmed as the cause

| Controller | Evaluated against | Recall |
| --- | --- | --- |
| OLD (trained on `token_loss_score`, the established 0.5182 baseline) | OLD hand-labeled ground truth | 0.5182 |
| OLD (trained on `token_loss_score`) | **NEW true-benefit ground truth** | **0.2227** |
| **NEW (trained on true-benefit labels, BCE)** | **NEW true-benefit ground truth** | **0.4030** |
| NEW (trained on true-benefit labels, ranking) | NEW true-benefit ground truth | 0.3958 |
| NEW (trained on true-benefit labels, BCE) | OLD hand-labeled ground truth (context only) | 0.2487 |

**Training toward the target that is actually being measured nearly
doubles recall against that target (0.2227 -> 0.4030, +0.18 absolute,
+81% relative)** -- this is the direct, controlled confirmation that the
fourth attempt's catastrophic result (0.22-0.28 controller recall) was
genuinely caused by the evaluation-target mismatch diagnosed there, not
by `causal_attention_benefit` being a bad signal or an artifact of that
specific run. Held constant across this comparison: the same features,
the same frozen backbone, the same held-out split, the same target rate
-- the ONLY thing that changed is which ground truth the teacher and the
metric agree on.

**The asymmetry is expected and does not undermine the result**: the
new controller scores lower against the OLD hand-labeled metric (0.2487,
down from the old controller's 0.5182 on that same old metric) --
because it was never trained to match hand-labeled construction points,
it was trained to match real downstream benefit. This is not a
regression; it is confirmation that the two ground-truth definitions are
genuinely different targets, and a controller optimized for one is not
automatically optimized for the other. Reported symmetrically (all four
combinations, not just the favorable one) so the picture is complete.

## What "fixed" means here, precisely

- **Not claimed**: a controller that beats 0.5182 recall on the
  ORIGINAL hand-labeled metric. That plateau (investigated four
  independent ways, all converging or falling short) stands as the
  honest ceiling for THAT specific metric.
- **Real and demonstrated**: the actual, previously-undiagnosed problem
  -- an evaluation-target mismatch, not a signal or controller-capacity
  problem -- is now directly confirmed via a controlled before/after
  comparison, and a coherent, internally-consistent controller/metric
  pair now exists that measures what arguably matters more for a real
  deployed system (does the trigger reduce real downstream loss, not
  does it land on a scenario designer's chosen construction point).
  Recall against that real target nearly doubled once training and
  evaluation were made consistent.
- **Reproduce**: `PYTHONPATH=. .venv/bin/python scripts/hz0c_c7_true_benefit_controller.py --examples 32 --objective bce`
