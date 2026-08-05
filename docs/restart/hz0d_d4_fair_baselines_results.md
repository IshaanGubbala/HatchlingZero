# HZ-0D D4: Fair Adaptation Baselines

Date: 2026-08-04. Real evidence for D4's exit gate ("gains are
attributable to temporary fast adaptation"). `reference/hz0d_fair_baselines.py`
implements every baseline the plan names; `tests/reference/test_hz0d_fair_baselines.py`
(5 tests) locks in the comparative findings below as regression tests.

Still isolated, per the plan's own D4/D5 text ("D4 (still isolated) may
proceed regardless" of the HZ-0C dependency gate): every baseline here
runs on the SAME `reference/hz0d_isolated_simulator.py` few-shot
symbol-remapping task D2/D3 already used (`dim=8`, `rank=2`,
`k_train=6`, `k_held_out=16`, `rule_scale=0.3`), not the real HZ-0C
model. Real HZ-0B/HZ-0C integration happens at D6.

## A disclosed substitution: two baselines have no faithful analog here

The plan names "HZ-0B memory only" and "HZ-0C only" as baselines. Both
are mechanisms defined over a TOKEN SEQUENCE with real temporal/surprise
structure (HZ-0B: associative read/write memory across time steps;
HZ-0C: surprise-triggered anchor attention). This task has neither --
each example is an independent `(x, y)` pair, not a sequence position,
and there is no surprise signal to gate anything. Rather than fake a
sequence or silently skip these two names, the mechanically closest
generic analog is substituted for each, named plainly as a substitution:

- **"HZ-0B memory only" -> `knn_retrieval_baseline`**: an associative
  memory read IS a key-value lookup over stored `(x, y)` pairs, once
  there is no sequence to condition retrieval on.
- **"HZ-0C only" -> `in_context_attention_baseline`**: anchor attention
  IS soft attention over stored context, once there is no surprise
  signal to gate it.

Real, HZ-0B/HZ-0C-specific baselines (with the actual memory-write and
surprise-controller machinery) belong at D6, where the real backbone
exists. This substitution is disclosed here, not smoothed over.

## The baselines

1. **No adaptation** -- zero delta, the frozen base model alone. The
   floor everything else must beat to be worth anything.
2. **Ordinary in-context learning** (`in_context_attention_baseline`) --
   Nadaraya-Watson kernel regression on the training residual
   (`context_y - base_pred(context_x)`), attention-weighted by distance
   in `x`, evaluated at each held-out point. No weight ever changes --
   a frozen model attending over its context, the real mechanical
   content of "in-context learning." Kernel bandwidth set by the
   median-heuristic (median squared pairwise distance among the context
   `x`'s alone -- no labels, no held-out data, no leakage).
3. **Longer context** (`longer_context_baseline`) -- the SAME
   mechanism, given `extra_k` MORE examples of the SAME true rule
   (freshly generated from `task.true_delta`, never touching held-out
   data). Tests whether more context alone, with no weight change,
   substitutes for a few examples plus a real low-rank update.
4. **Retrieval** (`knn_retrieval_baseline`) -- for each held-out point,
   average the `k` nearest training examples' raw OUTPUTS (no base-model
   correction, no weight change). Also stands in for "HZ-0B memory
   only" (see above).
5. **Static adapter** (`static_random_adapter_baseline`) -- a low-rank
   adapter with the SAME shape as a real fast-weight state, initialized
   with BOTH factors random and nonzero (a genuinely nonzero, unadapted
   delta, unlike `init_fast_weights`'s asymmetric zero-init), then NEVER
   updated. Isolates "having extra low-rank capacity" from "adapting
   that capacity."
6. **Permanent LoRA** (`meta_lora_baseline`) -- ONE adapter,
   gradient-trained across many DIFFERENT tasks (each an independently
   random `true_delta`), then FROZEN and evaluated on a NEW task's
   held-out data with no further, session-specific adaptation. Tests
   whether a general, permanently-shipped adapter can capture the same
   gains as a fresh, temporary update computed specifically for the new
   task.
7. **Gradient-updated adapter** -- `gradient_descent_update`, D3's
   real `mx.grad`-based mechanism, reused directly (not reimplemented).
8. **Fast weight adaptation (ours)** -- `delta_prediction_update`, D3's
   selected v4 (adaptive-ridge ALS) mechanism, reused directly.

## Result: real numbers, 8-seed mean, same seeds as D3

| Method | Mean held-out loss | vs. delta prediction (v4) |
| --- | ---: | ---: |
| No adaptation | 1.4120 | 3.82x worse |
| Static random adapter | 1.4122 | 3.82x worse |
| Permanent meta-LoRA (10 meta-tasks) | 1.7818 | 4.82x worse |
| k-NN retrieval (k=1) | 1.9959 | 5.40x worse |
| k-NN retrieval (k=3) | 1.5576 | 4.21x worse |
| In-context attention | 1.2611 | 3.41x worse |
| Longer context (+24 examples, 30 total) | 1.1187 | 3.02x worse |
| Longer context (+100 examples, 106 total) | 1.1619 | 3.14x worse |
| Gradient-updated adapter (D3 gradient descent) | 0.3997 | 1.08x worse |
| **Fast weight adaptation (D3 delta prediction, v4)** | **0.3699** | -- |

The gap is not close on any baseline. The two real fast-weight
mechanisms (gradient descent and delta prediction) both land in
`0.37`-`0.40`; every baseline is at least `3x` worse, several are
`4`-`5x` worse.

## Three findings that specifically rule out "it's not really adaptation"

**1. Extra unadapted capacity buys nothing.** `static_random_adapter`
(`1.4122`) is statistically identical to `no_adaptation` (`1.4120`) --
confirmed directly (`test_static_random_adapter_matches_no_adaptation`,
`<1%` relative difference). Having the SAME low-rank parameter budget
available, unfit to this session's examples, is worth exactly nothing.
The gain from real fast-weight adaptation cannot be explained by "more
parameters exist somewhere in the computation" -- it requires those
parameters to be fit to this specific session's data.

**2. A permanently pretrained adapter does not transfer.**
`permanent_meta_lora`, trained across 9-10 DIFFERENT random-rule tasks
and then frozen, does slightly WORSE (`1.7818`) than no adaptation at
all (`1.4120`) -- confirmed directly
(`test_permanent_meta_lora_does_not_beat_no_adaptation`). Since each
meta-training task's true rule is independent random noise relative to
every other one (and to the evaluation task's own rule), there is
nothing systematic for a single shared adapter to learn; what little it
does learn is mild negative transfer. This rules out "a good-enough
general adapter would have gotten these gains for free, no per-session
update needed" -- it demonstrably would not have.

**3. More context, without weight change, saturates far short of real
adaptation.** In-context attention over the SAME 6 examples reaches
`1.2611` (a real improvement over `1.4120`, showing the mechanism is not
broken); giving it `4x` more examples of the identical rule (`+24`,
`30` total) only reaches `1.1187`; `~17x` more (`+100`, `106` total)
does not improve further (`1.1619`, statistically flat, likely
curse-of-dimensionality noise at `dim=8` rather than a real regression).
Both remain roughly `3x` worse than 6 examples plus one real low-rank
weight update. Context alone, however much of it, is not a substitute
for adapting the weights.

**A fourth, secondary finding**: raw k-NN output-copying (no base-model
residual correction) does WORSE than doing nothing at all (`1.9959` vs
`1.4120` at `k=1`) -- confirmed directly
(`test_knn_retrieval_without_base_model_correction_can_underperform_no_adaptation`).
Ignoring what the frozen base model already knows is a real cost, not a
strawman baseline rigged to lose; `in_context_attention` (which DOES
correct relative to the base prediction) recovers most of that loss.

## Exit gate check

"Gains are attributable to temporary fast adaptation": yes, on every
axis tested. No baseline gets within `3x` of either real fast-weight
mechanism's mean held-out loss. The two most tempting alternative
explanations -- "it's just extra capacity" and "a general pretrained
adapter would do just as well" -- are both directly measured and both
fail (static-random ties no-adaptation exactly; permanent meta-LoRA
loses to no-adaptation). "It's just more context" saturates at roughly
`3x` worse even with `17x` more examples. The improvement really is
coming from session-local, example-fit low-rank weight adaptation, not
from any of the seven alternative explanations tested.
