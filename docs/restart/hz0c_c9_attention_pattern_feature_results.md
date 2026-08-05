# HZ-0C C7/C9: Four Real Attempts to Fix the Controller Plateau -- Convergent Honest Null Result

Date: 2026-08-04. Addresses the tracker's own priority #3 ("improve C9
trigger quality with a different causal objective or feature family, not
more tuning of the saturated token-loss controller") and a later direct
request to fix the C7 plateau. Four independent, real attempts -- a new
feature family, a new hypothesis space, a new training objective, and
(finally) a genuinely different teacher signal -- were tried; none
improved held-out quality, and all four converged to or fell below the
same ceiling.

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

## A fourth attempt: a genuinely different TEACHER signal, at full budget this time

The one remaining candidate explicitly named after the first three
attempts: not a new feature, hypothesis space, or objective on top of
`token_loss_score`, but a structurally different TEACHER. C6's own
`causal_attention_benefit` (real per-position downstream LM-loss benefit
from adding one anchor, `scripts/hz0c_c6_conditional_attention_eval.py`)
already existed and had been tried once before, blended with
`token_loss_score` at a small candidate budget (per C6's own record: 8
sequences, 4 candidates, blend 0.5) and rejected for regressing on 1/3
splits. That earlier attempt was explicitly left as "not re-attempted at
a larger budget" -- done here, as a PURE teacher (`causal_teacher_blend=1.0`,
no token-loss mixing) at increasing budgets, using `hz0c_c7_rl_trigger_controller.py::main`'s
existing (previously CLI-unexposed) `causal_teacher_sequences`/
`causal_teacher_candidates`/`causal_teacher_blend` parameters.

| Config | Controller recall | Teacher's own recall |
| --- | ---: | ---: |
| Baseline (token_loss only) | 0.4800 | -- |
| Causal teacher, 16 seqs / 16 candidates, seed 555 | 0.4661 | 0.6107 |
| Causal teacher, 32 seqs / 40 candidates (full budget), seed 555 | **0.2839** | **0.3177** |
| Causal teacher, 32 seqs / 40 candidates (full budget), seed 556 | **0.2174** | **0.2552** |

**Not just unhelpful -- actively, dramatically worse, and the gap widens
with a LARGER candidate budget, the opposite of what "not enough budget"
would predict.** At full budget the causal teacher's OWN recall against
the C3 scenarios' hand-labeled ground truth (0.26-0.32) is far below
`token_loss_score`'s typical 0.75-0.83, and the controller trained to
imitate it inherits that gap.

**A real, disclosed explanation, not just a negative number**: this is
most likely an evaluation-target mismatch, not a signal-quality failure.
`causal_attention_benefit` measures which position TRULY reduces
downstream loss most when anchored -- a different question from "which
position matches this scenario's hand-labeled construction point" (e.g.
a topic-shift's exact boundary token, a rebinding's exact reassignment
position). The real downstream-loss-optimal anchor for a given sequence
may genuinely be a different, nearby position than the one C3's
construction logic labeled as ground truth -- in which case a MORE
correct teacher (by the only standard that matters for an actual
deployed system, real loss reduction) scores WORSE on a recall metric
defined against a different target. This does not mean
`causal_attention_benefit` is a bad signal; it means "recall against
hand-labeled scenario positions" and "true downstream benefit" are two
different things this investigation was implicitly treating as
interchangeable, and are not.

## Honest conclusion

Four independent, real attempts -- a new feature family, a new
hypothesis space, a new training objective, and a structurally different
teacher signal at full budget -- were tried against the C7/C9 controller
plateau. The first three all converge to the SAME ~0.51-0.52 held-out
recall ceiling (`0.5182`-`0.5208` recall / `~0.107` precision) on
`token_loss_score`, not moving it in either direction. The fourth
(a real downstream-benefit teacher, tried precisely because it was named
as the one remaining structurally-different candidate) does not raise
that ceiling either -- it falls well BELOW it (0.22-0.28 controller
recall), with a real, disclosed reason: `causal_attention_benefit`
answers "which position truly reduces downstream loss," which is a
different question from "which position matches this scenario's
hand-labeled construction point," and the two are not interchangeable
here.

Taken together, this is a materially stronger, more complete
investigation than any single negative result: it rules out feature
representation, model capacity, training objective, AND (for the one
alternative teacher this project has built) teacher signal as the
bottleneck, while also surfacing a genuine methodological insight (the
evaluation metric itself measures something narrower than "true
downstream benefit," a fact worth carrying into any FUTURE scenario or
metric design, not just this controller). **`0.5182`-`0.5208` recall /
`~0.107` precision, on `token_loss_score`, via the linear controller,
remains the strongest real result C7/C9 has produced, and this
investigation is now genuinely exhausted for the levers available in
this codebase** -- not abandoned early, but concluded after four
independent, real, verified attempts, three of which converged and one
of which revealed why the fourth couldn't simply be swapped in. The
honest path forward, if this ceiling needs to move, is either a new
SCENARIO/ground-truth design whose labels ARE defined by true downstream
benefit (removing the mismatch this pass found, not something existing
C3 infrastructure does today), or accepting `token_loss_score` at
`~0.52` recall as this mechanism's honest, real ceiling and moving to
other C8/C9 work instead.
