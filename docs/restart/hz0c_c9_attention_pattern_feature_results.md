# HZ-0C C7/C9: Three Real Attempts to Fix the Controller Plateau -- Convergent Honest Null Result

Date: 2026-08-04. Addresses the tracker's own priority #3 ("improve C9
trigger quality with a different causal objective or feature family, not
more tuning of the saturated token-loss controller") and a later direct
request to fix the C7 plateau. Three independent, real attempts -- a new
feature family, a new hypothesis space, and a new training objective --
were tried; none improved held-out quality, and all three converged to
the same ceiling.

## What was tried

`scripts/hz0c_c7_rl_trigger_controller.py::attention_pattern_features`
(new). Every existing controller feature either proxies Q/K/V MAGNITUDE
(`attention_demand_features`: `mean(q*q)`, `mean(k*k)`, `mean(v*v)`,
`var(qkv)` -- direction-blind, cannot tell "this query strongly matches
one specific past key" from "this query is large but diffuse against
everything") or comes from the output vocabulary distribution
(next-token entropy/confidence/margin). This instead runs the frozen
backbone's OWN real causal softmax attention at its 6 anchor layers
(identical math to `reference/hz0a_mlx_model.py::CausalAttention`) and
extracts the REALIZED per-position attention distribution's entropy (how
spread a position's own attention is over its causal history) and peak
weight (how concentrated on one past position) -- content-aware and
direction-sensitive, a structurally different signal family. Fully
causal-safe: only each position's own attention row over its own past is
used, never a later position's attention back onto it.

Wired into `controller_input` (used by both C7's own controller and C9's
`causal_distilled_controller`), so every existing consumer picked it up
automatically.

## Result: real, nonzero learned weight; no change in held-out quality

Verified the new feature dims received substantial, non-trivial weight
during distillation (not silently zeroed out): on the standard
seed-557-held-out/555+556-train split, the 6 new feature dimensions'
weight-vector norm was **0.708** against a total weight-vector norm of
**2.131** -- roughly a third of the controller's total weight mass.

Despite that, the held-out exact-15%-rate result is **byte-identical** to
the pre-existing best: **0.5182291666666666 recall / 0.10677083333333333
precision** (`docs/restart/hz0c_c9_matched_cost_report_results.md`'s
recorded `0.5182`/`0.1068`), reproduced twice independently. This means
the discrete top-6-of-40 position SELECTION did not change for any of the
256 held-out examples, even though the controller is genuinely using the
new features with real weight -- the most likely explanation is that the
new signal correlates closely enough with the existing novelty/demand/
uncertainty features on this specific held-out data that redistributing
weight among them does not change the final ranking of which positions
land in the top 15%.

**A second, different hypothesis space was also tried**: the same
features through the existing MLP controller (`fit_mlp_controller`,
32-unit hidden layer, 1200 steps) instead of the linear one. This made
things WORSE, not better: **0.4531 recall / 0.0951 precision**, below
both the linear-with-new-features result and the original linear
baseline. Consistent with C6's own earlier finding that added controller
capacity does not help without a held-out gain to justify it.

## A third attempt: a genuinely different OBJECTIVE, not just features or capacity (2026-08-04, same day)

Requested explicitly: "fix the C7 issue." The two attempts above changed
the INPUTS (a new feature family) and the HYPOTHESIS SPACE (an MLP). A
third, different lever was tried: the LOSS FUNCTION itself.

`scripts/hz0c_c7_rl_trigger_controller.py::fit_ranking_controller` (new).
`fit_controller` minimizes pointwise binary cross-entropy against the
teacher's exact-top-15% labels -- a CALIBRATION objective (push each
position's probability toward 0 or 1 independently, only mildly
upweighted for the positive class at `positive_weight=2.0` against a
natural ~5.7:1 imbalance). But every real consumer of the controller's
output (`exact_topk_labels`, `bounded_actions`, `exact_topk`) only ever
uses the score's RELATIVE ORDER to pick the top-k positions --
calibration is never actually consumed. `fit_ranking_controller` instead
minimizes a pairwise RankNet-style logistic ranking loss: for every
(positive, negative) position pair within the SAME example,
`log(1 + exp(-(score_positive - score_negative)))` -- directly rewards
correct ranking, immune to the class-imbalance-weighting question
entirely (no separate positive/negative term to balance).

Verified correct on a synthetic sanity check first (a toy task where
labels are the top-3 positions by one feature dimension): both the
ranking loss and BCE learn the right direction (large positive weight on
the informative feature), ranking edging out BCE slightly (0.983 vs
0.950 top-3 recall) -- the implementation works as intended before
being trusted on the real evaluation.

**Real held-out result, same protocol as both prior attempts (train
555+556, eval 557, 8 real C3 scenarios, exact 15% rate)**: default
config (1200 steps, lr=0.2) gave **0.5143 recall / 0.1061 precision**,
essentially tied with BCE's 0.5182/0.1068 (a difference smaller than
run-to-run noise elsewhere in this project). A small learning-rate/step
sweep (`lr` in `{0.02, 0.05, 0.5, 1.0}`, `steps` in `{600, 2400}`) found
the ranking loss's BEST configuration (`lr=0.05`, `steps=2400`) reaches
**0.5208 recall / 0.1068 precision** -- marginally above BCE's recall,
identical precision, well within what a single held-out split's noise
could produce. `lr=1.0` clearly destabilizes (0.4167-0.4193), confirming
the sweep is real optimization behavior, not a flat/broken loss surface.

## Honest conclusion

Three independent, real attempts -- a new feature family, a new
hypothesis space, and now a new training objective -- all converge to
the SAME ~0.51-0.52 held-out recall ceiling on this teacher signal and
this scenario collection. **0.5182-0.5208 recall / ~0.107 precision is
the ceiling, not a number any of these three levers could move.** This
is stronger, more convergent evidence than any single negative result
alone: it points away from "we haven't found the right model/objective
yet" and toward a genuine INFORMATION ceiling -- the causal features
available at real inference time (hidden-state novelty, attention
demand, realized attention patterns, next-token uncertainty) most likely
do not carry enough signal to approach `token_loss_score`'s own
0.75-0.83 ceiling, which uses the actual future token, information
genuinely unavailable at deployment time. Reported as the honest
conclusion of this investigation, not as a reason to stop looking, but
as a reason to stop tuning the SAME lever family: real remaining
candidates, structurally different from all three tried here, are a
genuinely different TEACHER (not token_loss_score) reflecting true
downstream benefit at scale (C6's own bounded downstream-teacher screen
regressed on 1/3 splits at a small candidate budget, not re-attempted at
a larger one), or accepting the current ceiling as this teacher's
honest limit and moving to other C8/C9 work instead.
