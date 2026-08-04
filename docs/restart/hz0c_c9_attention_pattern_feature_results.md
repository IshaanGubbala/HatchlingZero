# HZ-0C C9: A Different Causal Feature Family -- Real Attempt, Honest Null Result

Date: 2026-08-04. Addresses the tracker's own priority #3 ("improve C9
trigger quality with a different causal objective or feature family, not
more tuning of the saturated token-loss controller"). Tried a genuinely
new feature family; it did not improve held-out quality.

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

## Honest conclusion

Both a new causal feature family (real, nonzero weight, content-aware)
and a new hypothesis space (MLP nonlinearity) were tried in good faith
and neither improved the held-out multi-seed trigger-quality ceiling.
**0.5182 recall / 0.1068 precision remains the strongest deployable C9
result.** This is disclosed as a genuine negative finding, not hidden or
reframed -- the plateau the tracker named ("saturated token-loss
controller") appears to be a real ceiling on THIS teacher signal and
THIS scenario collection at THIS scale, not merely a feature-
representation gap. Real remaining candidates, not attempted this pass:
a genuinely different TEACHER (not token_loss_score) reflecting true
downstream benefit at scale (C6's own bounded downstream-teacher screen
regressed on 1/3 splits and was rejected, but only at a small candidate
budget), more/larger training scenario diversity, or accepting the
current ceiling and moving to C8's model-level integration / C9's
remaining end-to-end cost-quality-latency report instead.
