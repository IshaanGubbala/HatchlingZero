# HZ-0F: Investigation Summary and Close-Out

Date: 2026-08-06 (updated to cover the full phase, not just F1-F5). HZ-0F
was a follow-up investigation into E8/E10's disclosed structural OOD
quality tradeoff, triggered by a 13-candidate architecture proposal, and
later extended to a second survey of 2026 architecture/systems
developments. Rather than committing to any candidate directly, the
investigation ran a sequence of cheap, real, falsifiable diagnostics
first, each gating the next step on real evidence -- across two threads:
(1) the fallback/routing investigation (F1-F5), and (2) three independent
technique tests motivated by the later survey (Attention Residuals,
`mx.gather_mm`, counterfactual router refinement). **This is HZ-0F's
final close-out -- the phase is done; HZ-0G is next.**

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

## Verdict on the fallback/routing thread (F1-F5)

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

**If reopened**: a layer-composition matrix -- `broad_only` applied to
each individual layer alone (`27`, `28`, `30`), each pair (`27+28`,
`28+30`, `27+30`), and all three -- would reveal whether one specific
layer's interaction breaks the effect, or whether the failure only
emerges from cumulative specialization drift across 2+ simultaneously-
specializing layers. Not run in this investigation.

## Three further technique tests, motivated by a 2026 systems/architecture survey

- **Attention Residuals** (`docs/restart/hz0f_attnres_ablation_results.md`):
  depth-wise attention over previous layer representations, implemented
  and tested (full-rank, low-rank, multi-head) on the tiny exact-GDN-2
  model, real text, 3 seeds. **Rejected at this scale** -- standard
  residual won on every metric, every variant, every seed, by a
  consistent `~0.05-0.07` nat margin, not implementation error (verified
  via a real correctness property plus 32/32 passing tests). Not
  falsified as an idea -- the source technique was validated at
  100M-1B parameters, `20-200x` larger than this test. Unresolved at
  that scale; not attempted here.
- **`mx.gather_mm`** (`docs/restart/hz0f_gather_mm_benchmark_results.md`):
  MLX's native grouped-matmul primitive, benchmarked against E9's
  hand-written `mx.fast.metal_kernel` custom kernel. **Real win** --
  `gather_mm` consistently beat the custom kernel (`~19.7-19.9ms` vs
  `~20.5-20.6ms`, verified correct via bit-exact synthetic parity and
  real-checkpoint tolerance checks), closing the gap to MLX's own
  reference implementation from E9's `~5-6%` down to `~0.5-1%`. Fused
  gate+up projection showed no consistent benefit once a real
  weight-recaching confound was found and fixed. `gather_mm` is now the
  best PMetal-class MoE kernel result in this project, superseding five
  rounds of hand-written Metal engineering with less code.
- **Counterfactual router refinement**
  (`docs/restart/hz0f_counterfactual_router_refinement_results.md`):
  turned F1's oracle into a real router-supervision signal, applied as a
  post-curriculum refinement (a real precondition discovery ruled out
  applying it as an initial warm-start: experts are bit-identical at
  `init_e6_layers`' own starting point, making "best expert" arbitrary
  until real training differentiates them). **Real, mixed result** --
  improved domain-win count in 2 of 3 seeds, but WIDENED the general/OOD
  gap in every single seed. Not adopted -- refines rather than merely
  confirms F3's prediction: not "no effect either way" but "actively
  worse for OOD while modestly helping specialization."

## HZ-0F's single, unifying finding

**Three structurally independent mechanisms -- fallback training target
(F4/F5), router decisiveness (counterfactual refinement), and multi-layer
composition itself (F5) -- all produced the SAME signature**: anything
that sharpens in-domain specialization measurably costs general/OOD
quality, and multi-layer interaction is where single-layer fixes go to
die without a known mechanism. This is now a load-bearing, three-times-
confirmed property of this MoE design, not a one-off finding from any
single experiment. Two of three motivated architecture/systems ideas from
the later survey were real wins or real informative negatives
(`gather_mm`: adopt; AttnRes: scale-gated rejection); the third
(counterfactual routing) joined the same specialization-tradeoff pattern
as the fallback work.

## Stopping here

**HZ-0F is closed.** This is the correct scientific stopping point: every
thread pursued either produced a real, adopted improvement
(`gather_mm`), a real and precisely bounded negative result (AttnRes at
this scale, counterfactual refinement, joint-scope fallback
composition), or converged on the same honest, unresolved mechanistic
question (why doesn't specialization-vs-generality compose across
layers) rather than being forced into a false resolution. HZ-0G
(architecture freeze + integration) is next; see
`plans/HZ-0G_Integration_Plan.md`.
