# HZ-0F: Investigation Summary and Close-Out (F1-F5)

Date: 2026-08-06. HZ-0F was a follow-up investigation into E8/E10's
disclosed structural OOD quality tradeoff, triggered by a 13-candidate
architecture proposal. Rather than committing to any of the 13 candidates
directly, the investigation ran a sequence of cheap, real, falsifiable
diagnostics first, each gating the next step on real evidence.

## Sequence and real findings

- **F1** (`docs/restart/hz0e_f1_oracle_routing_audit_results.md`): oracle
  routing audit. The router's per-token selection is not
  disproportionately worse OOD than in-distribution -- ruling out
  abstention-to-dense and a mandatory shared-dense-trunk as the
  evidence-supported fix. Also corrected a wrong claim (MoE's internal
  fallback is not frozen -- it receives real gradients from overflow).
- **F2** (`docs/restart/hz0e_f2_gate_overflow_fallback_results.md`): gate
  confidence and overflow/fallback audits. Neither gate miscalibration
  nor fallback quality showed a large OOD-specific effect at the fast
  diagnostic training scale.
- **F3** (`docs/restart/hz0e_f3_full_scale_reaudit_results.md`): re-ran
  F1/F2 at the full 50-step training scale. Routing-selection headroom
  grows with training but stays regime-balanced (confirming F1 at
  scale). A NEW finding emerged only at this scale: a real, growing,
  OOD-unfavorable fallback-vs-dense differential, absent at the faster
  diagnostic scale.
- **F4** (`docs/restart/hz0e_f4_fallback_isolation_results.md`): three-arm
  causal test at single-layer (27) scope. `broad_only` fallback training
  (fallback trains only on general-prose replay, not curriculum-domain
  overflow gradients) reproducibly REVERSED the OOD deficit into a net
  MoE advantage, in all 3 seeds, while preserving most in-domain wins
  and improving absolute fallback quality with no compensating cost.
  `frozen` was a clean negative result, ruling out "isolation from
  curriculum domains" as the mechanism in favor of "needs dedicated
  general training."
- **F5** (`docs/restart/hz0e_f5_joint_scope_fallback_validation_results.md`):
  validated `broad_only` at the full 3-layer joint scope (27/28/30)
  before locking it as a default or combining it with anything else.
  **The fix did not survive**: worse than `current` in 2 of 3 seeds,
  better in only 1, never net-positive. `frozen`-is-worse and domain-win
  preservation both DID generalize.

## Final verdict

**Broad-only fallback training is an effective local intervention, not a
validated system-level solution.** It reliably fixes the layer-27 OOD
deficit in isolation (F4, real, reproducible, 3/3 seeds), but multi-layer
MoE introduces an unresolved interaction that prevents that single-layer
recovery from composing across the full 3-layer contract (F5, real,
reproducible, 2/3 seeds worse). The mechanism behind that non-composition
is not identified and is not assumed away.

Two things ARE established as general, not layer-scope-dependent:

- The fallback needs active training toward general robustness -- mere
  protection from curriculum-domain specialization (freezing) is
  consistently worse, not neutral, at both scopes.
- Domain-win (in-distribution specialization) is not put at risk by any
  fallback-training policy tested, at either scope -- the open failure
  is specifically about recovering general/OOD quality once multiple
  MoE layers interact, not about specialization quality itself.

Counterfactual-utility router training (the one other candidate F3 found
real, regime-balanced headroom for) was deliberately kept separate and
not combined with the fallback fix, since a compound result at this
point would have obscured whether the fallback fix itself scales -- it
does not, so combining now would risk misattributing credit.

## Stopping here

This is the correct scientific stopping point for this round: a real,
reproducible, single-layer fix was found, tested rigorously at the scope
that matters (joint, not isolated), found not to generalize, and
reported as such rather than forced into a "fix" narrative the evidence
does not support.

## If reopened: the next cheapest diagnostic

A layer-composition matrix -- `broad_only` applied to each individual
layer alone (`27`, `28`, `30`), each pair (`27+28`, `28+30`, `27+30`),
and all three -- would reveal whether one specific layer's interaction
breaks the effect, or whether the failure only emerges from cumulative
specialization drift across 2+ simultaneously-specializing layers. Not
run in this investigation; the natural next step if HZ-0F resumes.
