# HZ-0F F3: Full-Scale Re-Audit (50-Step Protocol)

Date: 2026-08-06. F1/F2 were run at this project's fast-but-real
diagnostic scale (`balanced_steps=15, mixed_steps=15, imbalanced_steps=15,
warm_start_steps=20`), matching E8's own regression-test convention, not
the `50/50/50`-step, `warm_start_steps=40` scale the headline E8/E10
aggregate numbers were measured at. Per the direct follow-up request:
re-run F1 and F2 at the FULL default protocol (`run_curriculum`/
`run_warm_dense_baseline`'s own defaults -- `balanced_steps=50,
mixed_steps=50, imbalanced_steps=50, warm_start_steps=40`), 3 seeds, to
test whether the OOD gap or the oracle headroom changes with longer
specialization training.

Real training time at full scale: `~23-26s` per seed (both MoE
curriculum and dense baseline together) -- fast enough that "diagnostic
scale vs. headline scale" was never a real computational constraint, only
an unexamined assumption carried over from the regression-test
convention.

## Result 1: routing-selection oracle gap grows with training, but stays regime-balanced (confirms the prediction)

| Seed | Unscaled gap (in-dist) | Unscaled gap (OOD) | Gated gap (in-dist) | Gated gap (OOD) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.0921 | 0.0968 | 0.1410 | 0.1393 |
| 1 | 0.0882 | 0.0939 | 0.1367 | 0.1369 |
| 2 | 0.0882 | 0.0950 | 0.1358 | 0.1362 |

Compared to the 15-step scale (unscaled `~0.034-0.039`, gated
`~0.105-0.109`), the oracle gap roughly DOUBLED with the fuller
curriculum -- real, growing headroom in routing selection as
specialization training proceeds. But the gated gap (the more realistic
framing) is now essentially IDENTICAL between regimes in every seed
(`0.1410` vs `0.1393`, `0.1367` vs `0.1369`, `0.1358` vs `0.1362` --
seed 1 is OOD-slightly-HIGHER, the only case in either scale where OOD
edges ahead, and even then by `0.0002`). **This exactly matches the
predicted outcome**: real headroom exists for training the router on
counterfactual utility (candidate #3), and it would likely improve
BOTH regimes roughly equally -- it is not, by itself, a lever for
closing the RELATIVE OOD tradeoff, confirmed now at the scale that
matters, not just the fast diagnostic scale.

## Result 2: a real, growing, OOD-unfavorable fallback effect (new at this scale)

| Seed | In-dist fallback gap | OOD fallback gap | OOD - in-dist differential | Overflow rate (in-dist / OOD) |
| --- | ---: | ---: | ---: | ---: |
| 0 | -0.0058 | 0.0002 | 0.0060 | 42.9% / 45.6% |
| 1 | -0.0055 | 0.0085 | 0.0140 | 42.9% / 45.6% |
| 2 | -0.0084 | 0.0089 | 0.0173 | 42.3% / 44.4% |

At 15 steps this differential was small and inconsistent in sign
(`0.0053, 0.0005, 0.0001` -- essentially noise). At 50 steps it is
**consistently positive and meaningfully larger in all 3 seeds**
(`0.0060, 0.0140, 0.0173`) -- a real, reproducible, training-scale-
dependent effect, not noise. The pattern: the shared fallback
increasingly BEATS the fair dense baseline in-distribution as training
proceeds (`-0.0055` to `-0.0084`, consistently negative -- getting
BETTER than dense on the domains the curriculum trains on), while
staying flat-to-slightly-worse than dense OOD (`0.0002` to `0.0089`).

**Mechanistic explanation, consistent with F2's correction**: the
fallback is not frozen -- it receives real gradient updates whenever a
token overflows to it. Overflow happens on curriculum-domain tokens
(prose/code/math/json/tools) throughout training, so the fallback
incidentally absorbs curriculum-domain-flavored specialization through
overflow-triggered gradients alone, DESPITE being nominally the
"general" shared path. It is not immune to the same
specialization-costs-generality tradeoff the experts show explicitly --
it is quietly acquiring a smaller version of the same tradeoff, just
through an indirect (overflow-gated) training signal instead of direct
task assignment.

**Quantified contribution to the aggregate gap**: weighting the OOD
fallback gap by the real OOD overflow rate (`~45%`) estimates this
effect's contribution to the aggregate documented OOD quality gap
(dense `2.5408` vs MoE `2.5559`, `0.0151` nats): `0.456 * 0.0002 =
0.0001` (seed 0), `0.456 * 0.0085 = 0.0039` (seed 1), `0.444 * 0.0089 =
0.0040` (seed 2) -- roughly `0-27%` of the full aggregate gap, seed-
dependent, growing with training. **Real and directionally consistent,
but not the whole explanation on its own** -- the majority of the
aggregate gap, especially in seed 0, remains unaccounted for by this
mechanism alone.

## Result 3: gate calibration remains small and non-OOD-amplified even at full scale

| Seed | In-dist gate delta | OOD gate delta |
| --- | ---: | ---: |
| 0 | -0.0052 | -0.0018 |
| 1 | -0.0020 | -0.0040 |
| 2 | -0.0016 | -0.0041 |

Comparable in magnitude to the 15-step results (`~0.002-0.006` nats),
no consistent growth or OOD-specific amplification with more training.
This mechanism is confirmed, now at two training scales, as NOT a
meaningful contributor.

## Synthesis and honest next step

The leading hypothesis proposed alongside this re-run -- "the gap is a
distributed representation/generalization effect from expert
specialization, not one broken routing component" -- is well-supported
by these results, with one partial, real, actionable exception found:
the shared fallback's incidental overflow-driven training IS measurably
absorbing some of the same specialization-costs-generality tradeoff the
experts show explicitly, and this effect is real, reproducible, and
growing with training scale (unlike gate calibration, which stayed flat).
It explains a real but partial (`0-27%`, seed-dependent) share of the
aggregate gap.

This suggests a concrete, narrower, lower-risk intervention worth testing
before any of the 13 heavier architecture candidates: **hold the
fallback's weights frozen after warm-start** (do not let it receive
gradient updates from overflow tokens during curriculum training) or
**train it on a broad/general corpus separately from the curriculum's 5
domains**, isolating it from the same specialization pressure the
experts are intentionally given. This is a small, targeted change
compared to a shared-dense-trunk architecture rewrite, and this
document's own evidence is what motivates it -- not a re-ranking of the
original 13 candidates without cause.

Not yet tested: whether freezing (or separately training) the fallback
actually closes any of the aggregate gap when tried directly. That is
the natural next real experiment, not assumed here.
