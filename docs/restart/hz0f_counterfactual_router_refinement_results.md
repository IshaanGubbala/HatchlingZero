# HZ-0F: Counterfactual-Utility Router Refinement

Date: 2026-08-06. Follow-up to F3's finding (real, regime-balanced
routing-selection headroom exists, growing with training scale) and the
motivating architecture survey's candidate #3 (train the router on real
per-token counterfactual utility instead of a coarse domain-label proxy).
Turns F1's oracle (`per_token_losses_forced_expert`) into an actual
supervision signal and tests it directly.

## Implementation

`reference/hz0f_counterfactual_warmstart.py`. `compute_counterfactual_labels`
computes a real per-token best-expert label: `argmin` over the
`num_experts` forced-expert per-token losses (reusing F1's own forced-
expert forward, not a reimplementation). `counterfactual_warm_start`
trains ONLY the router (matching `supervised_warm_start`'s contract --
experts/fallback untouched) via per-TOKEN cross-entropy against these
labels, instead of E3's per-BATCH single domain-label target.

## Real discovery before the real experiment: labels are degenerate on fresh experts

Testing `compute_counterfactual_labels` on `init_e6_layers`' own output
(the standard starting point for warm-start) found every label
collapsing to a single expert -- checked directly, not assumed a bug:
`init_e6_layers` broadcasts the SAME dense-FFN slice identically across
all 4 experts (verified: `expert_gate_w[0]` bit-equal to
`expert_gate_w[1]`, `[2]`, etc.), so "which expert is best" is genuinely
undefined/arbitrary at that point -- there is no real per-token utility
signal to learn from until experts have differentiated through real
training. **This rules out "replace the initial domain-label warm-start
with counterfactual warm-start" as the design** -- redesigned as a
POST-curriculum refinement step instead: run the real, existing
curriculum first (differentiates the experts), then apply a short
counterfactual router-only refinement on top, using labels computed from
the now-differentiated experts.

## Real experiment: refinement after real training, 3 seeds

Full real curriculum (`run_curriculum`, fast-but-real
`15/15/15`-step/`warm_start_steps=20` protocol, matching this
investigation's own established test scale), then 20 steps of
counterfactual router refinement on top, compared against the
unrefined baseline, same fair dense comparison
(`run_warm_dense_baseline`) both times:

| Seed | Variant | Domain win | Per-domain mean | OOD gap (MoE - dense) |
| --- | --- | :---: | ---: | ---: |
| 0 | baseline | 3/5 | 2.1005 | -0.0016 |
| 0 | refined | 4/5 | 2.1004 | **0.0033** |
| 1 | baseline | 3/5 | 2.1001 | 0.0043 |
| 1 | refined | 3/5 | 2.1025 | **0.0090** |
| 2 | baseline | 3/5 | 2.1000 | -0.0039 |
| 2 | refined | 4/5 | 2.1002 | **0.0059** |

**Two real, consistent findings**:

1. **Domain win count improves in 2 of 3 seeds** (`3/5 -> 4/5`), per-
   domain mean loss essentially unchanged -- real headroom in routing
   selection (F3's finding) was genuinely capturable, if modestly.
2. **The OOD/general gap gets WORSE in every single seed**
   (`-0.0016 -> 0.0033`, `0.0043 -> 0.0090`, `-0.0039 -> 0.0059` -- all
   three deltas positive, i.e. MoE's disadvantage vs. dense on general
   quality grows after refinement, not just fails to improve).

## This refines, not just confirms, F3's prediction

F3 predicted counterfactual routing "may improve overall quality but is
unlikely by itself to eliminate the relative OOD tradeoff." The real
result found here is more specific and slightly less favorable than
"no effect either way": refinement doesn't just fail to close the OOD
gap, it ACTIVELY WIDENS it, consistently, while modestly helping
in-domain specialization. Mechanistically coherent with every other
finding in this investigation: sharpening the router's decisiveness
toward each token's single best-performing expert is itself a form of
specialization pressure, and specialization pressure costs OOD
generality here regardless of WHICH mechanism applies it (architecture,
as F4/F5's fallback investigation found; or routing sharpness, as found
here) -- the same structural signature, now confirmed at a third,
different mechanism.

## Verdict

**Not adopted.** Counterfactual router refinement provides a real but
small in-domain benefit at a real, consistent, measurable OOD cost --
not a free win, and not the "improves everything, hurts nothing" result
that would justify adoption. Locked in as a regression test
(`tests/reference/test_hz0f_counterfactual_warmstart.py`, 2 tests) so
the direction of both findings (degenerate labels on fresh experts;
consistent OOD-gap widening after refinement) is caught if future
changes to routing/warm-start/curriculum code flip it, not silently
re-asserted.
