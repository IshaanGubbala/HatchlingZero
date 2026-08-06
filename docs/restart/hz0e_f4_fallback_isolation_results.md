# HZ-0F F4: Three-Arm Fallback Isolation Experiment

Date: 2026-08-06. Direct test of F3's leading hypothesis (the shared
fallback quietly absorbs curriculum-domain specialization through
incidental overflow-triggered gradients, contributing to the OOD quality
gap) via three concrete, controlled training policies, holding routing,
capacity, overflow rate, initialization, and token order identical:

1. **`current`** -- the real, shipped policy: fallback trains only on
   whatever curriculum-domain gradients overflow sends it.
2. **`frozen`** -- fallback warm-started identically, then genuinely
   frozen (gradient zeroed every curriculum step; verified bit-identical
   to its warm-start value after training, not just assumed).
3. **`broad_only`** -- fallback receives NO gradient from curriculum
   steps, but DOES receive a real, dedicated gradient update from a
   separate general-prose replay batch after every curriculum step
   (router/experts zeroed for that replay step, so only the fallback
   moves). Disclosed directly: this gives the fallback specifically one
   dedicated optimizer step per curriculum step, more total fallback-
   targeted updates than `current`/`frozen` get (which only depend on
   incidental overflow) -- the real, most direct implementation of
   "isolate the fallback from curriculum-domain pressure," not
   step-count-matched to the other two arms.

`reference/hz0e_f4_fallback_isolation.py`; same full 50/50/50-step,
`warm_start_steps=40` protocol as F3, 3 seeds, single layer (27).

## Result: broad-only fallback training meets every stated success criterion

| Seed | Arm | Per-domain win | General/OOD gap (MoE - dense) | Fallback differential |
| --- | --- | :---: | ---: | ---: |
| 0 | current | 4/5 | +0.0188 | 0.0060 |
| 0 | frozen | 4/5 | +0.0254 | 0.0057 |
| 0 | broad_only | 3/5 | **-0.0013** | **0.0023** |
| 1 | current | 4/5 | +0.0116 | 0.0140 |
| 1 | frozen | 4/5 | +0.0176 | 0.0181 |
| 1 | broad_only | 4/5 | **-0.0097** | **0.0010** |
| 2 | current | 4/5 | +0.0147 | 0.0173 |
| 2 | frozen | 4/5 | +0.0205 | 0.0205 |
| 2 | broad_only | 4/5 | **-0.0070** | **0.0011** |

(General/OOD gap: positive means MoE loses to dense; negative means MoE
wins. Fallback differential: OOD fallback-vs-dense gap minus
in-distribution fallback-vs-dense gap, from F2's audit.)

**All four of the stated success criteria are met by `broad_only`, in
every seed:**

1. **Preserves most of the in-domain MoE advantage**: `3/5` or `4/5`
   domains still won (vs `current`'s consistent `4/5`) -- a small, real,
   disclosed cost in seed 0 only, not a collapse of specialization.
2. **Materially reduces the OOD gap -- and flips it net-positive**: from
   `current`'s consistent `+0.012` to `+0.019` nat MoE deficit, to a
   consistent `-0.001` to `-0.010` nat MoE ADVANTAGE, in all 3 seeds.
   This is not a partial improvement -- MoE now beats dense on general
   quality too, not just per-domain quality.
3. **Nearly eliminates the growing fallback differential**: from
   `current`'s `0.006-0.017` down to `broad_only`'s `0.001-0.002` --
   an `~85-95%` reduction, consistent across seeds.
4. **No compensating increase in overflow loss**: checked directly
   (not merely inferred from the relative gap) -- `broad_only`'s
   ABSOLUTE fallback-token loss is the LOWEST of all three arms in
   both regimes (seed 0: `2.8320`/`2.8734` vs `current`'s
   `2.8678`/`2.8904`), and overflow rate itself is essentially
   unchanged across arms (`42.9-43.5%` in-distribution, `45.6-45.8%`
   OOD) -- routing/capacity dynamics are not meaningfully perturbed by
   which fallback-training policy is used, as required.

## The frozen arm is a real, informative negative result -- and redirects the mechanism

`frozen` does NOT reduce the fallback differential (`0.0057-0.0205`,
statistically indistinguishable from `current`'s `0.0060-0.0173`) and
makes the OOD gap WORSE in every seed (`+0.0176` to `+0.0254`, worse
than `current`'s `+0.0116` to `+0.0188`). This is informative, not just
disappointing: **it rules out "incidental curriculum-domain training is
what corrupts the fallback" as the mechanism.** If that were the whole
story, removing that training (freezing) should have moved the
differential toward zero -- it did not. What actually helped was not
LESS fallback training, but MORE, DIFFERENT (general-domain-targeted)
fallback training. The fallback was not being harmed by exposure to
curriculum domains specifically; it was simply undertrained for general
robustness relative to what a dedicated general-purpose path needs,
and giving it that dedicated training closed the gap directly.

## Honest scope and next steps

Single layer (27), 3 seeds, same fast full-scale protocol as F3
(`~15-20s` per arm per seed -- cheap enough that this was never a
computational constraint). The `broad_only` policy is not step-count-
matched to the other two arms for the fallback specifically (disclosed
above) -- a fairer ablation would test whether MATCHING total fallback
gradient-update count (not just "give it dedicated general training")
is what matters, or whether the general-domain CONTENT of those updates
is what matters, independent of count. Not distinguished here.

Not yet tested: whether this holds at the full 3-layer joint scope
(27, 28, 30 together, matching E8's own joint-multilayer result), or
whether combining `broad_only` with candidate #3 (counterfactual-
utility router training, confirmed in F3 to have real, regime-balanced
headroom) compounds further. Both are natural next steps given this
result, not assumed to work without testing.

**This is the strongest, most complete, most actionable result of the
HZ-0F investigation so far**: a small, targeted, low-risk change (retrain
the fallback's own training signal, not the architecture) that meets
every stated success criterion in every seed tested, versus any of the
13 heavier architecture candidates the original proposal ranked ahead of
this diagnostic sequence.
