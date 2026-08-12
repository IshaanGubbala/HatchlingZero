# HZ Integrated Candidate Plan

Inserted 2026-08-11, after `plans/HZ Phase 2R State Redesign Plan.md`
and the Phase 4/5 results. Direct response to the pattern that showed up
four separate times this session (2R-C zero-shot grouping, Phase 4
zero-shot BlockBDH, Phase 5 zero-shot depth extrapolation, and 2R-B's
own initial zero-shot-equivalent undertraining trap): **zero-shot
architectural compression is not assumed to preserve behavior.**

## HZ Principle #1

> **Structural efficiency mechanisms must be trained in-path. If a
> mechanism changes the computation path, it must be part of training
> from the start — not applied post-hoc to a model trained without it.**

Every result in this project that violated this (post-hoc compression,
post-hoc sparsity, post-hoc depth extrapolation) failed, sometimes
badly. Every result that respected it (2R-B's value bottleneck trained
from scratch, Phase 4's BlockBDH trained end-to-end on passkey) worked,
at least partially. This is now a standing design rule for the rest of
HZ, not a one-off observation.

## What we actually have, separately, right now

- **Memory win** ✅: value bottleneck + INT8, ~16-21x safe state
  reduction (task-checked on two tasks, real cliff found and bisected).
- **Compute win** ✅: BlockBDH, real measured 1.95x-6.20x wall-clock
  speedup, quality real when trained end-to-end but currently unstable
  on the harder task at aggressive fractions.
- **Stateful-task win** ✅ (from H5, predating this redesign): BDH's
  real state genuinely beats a zeroed/no-state control on passkey and
  reassignment.

**Not yet shown**: memory + speed + quality together, in the SAME
model, trained as one system. That is now the central missing piece and
the central milestone.

## Step 1 — Lock HZ-State-v1 (the safe, non-extreme config)

Per the bisection result
(`docs/restart/hz0h_phase2r_reassignment_bisection_results.md`), the
extreme 32x setting (d_state=D/8) has a real capacity cliff on the
harder task. Locking the SAFE configuration, not the most aggressive
one:

```text
Value bottleneck: d_state = D/4 (4x reduction, 0% degradation on both
                   tasks tested)
State quantization: INT8 (4x reduction, 0% degradation on top)
Combined: 16x state reduction vs. exact BDH's fp32 state
```

d_state=D/4 chosen over the more aggressive D/5.33 (d_state=6, ~21x,
2% real degradation) — start from the fully-clean point, not the edge
of the cliff. Not spending further time squeezing another 2x here; this
is a stopping point, not a target to keep pushing past.

## Step 2 — Redesign (not tune) grouped state, once, then stop if it fails again

2R-C's four-training-methods-same-floor result is treated as decisive:
information bottleneck in the FORMULATION, not an optimizer problem.
One new design, tried once:

**Learned shared state + depth slots**: instead of forcing several
layers to write into literally the same state tensor (2R-C's design),
give each layer a small learned addressing vector `a_l` over `k` shared
memory banks `S_1...S_k`, with soft read/write routing
`S_l^read = sum_j p_lj * S_j` (`p_l = softmax(a_l)` or similar). Depths
share the expensive memory but keep a learned identity within it,
instead of being forced into one undifferentiated pool.

**If this also plateaus at the same loss floor pattern 2R-C showed:
kill grouped-state compression entirely.** Value bottleneck already
solved most of the real memory problem (16x on its own); grouped state
was always the harder, more speculative lever. Do not keep iterating
past one real redesign attempt.

## Step 3 — Retarget BlockBDH to 50% active (not 12.5%), fix training stability first

50% active alone is already a real ~2x speedup, and Phase 4's own
finding was that its failure mode looks like training instability
(some seeds reach 1.00, others 0.60), not a hard capability wall.

Build **HZ-Block-v1** = HZ-State-v1 (VB D/4 + INT8) + 50% block
activation, trained end-to-end from initialization (not layered onto an
already-trained dense-state model). Run **>=5 seeds** on the
reassignment task. Real, concrete target:

> Turn 0.60-1.00 (single-seed range found so far) into consistently
> ~0.95-1.00 across seeds.

Only drop to 25% active if 50% is first made seed-stable. Do not chase
more aggressive sparsity while the current level is still unreliable.

## Step 4 — Learned router, once BlockBDH's fixed heuristic router is the bottleneck

Small learned router (`p = softmax(W_r x)`, top-k block selection) with
an explicit load-balancing term (`L = L_LM + lambda * L_balance`) to
prevent block collapse. Track block utilization, router entropy,
collapse, reassignment accuracy, CE, and wall-clock together — not just
task accuracy alone. No RL-based routing yet; softmax + top-k + balance
loss only.

## Step 5 — Train variable depth from scratch (not zero-shot extrapolation)

Phase 5's real finding: a model trained at depth `d` does not know how
to reason at `2d` — this does NOT mean adaptive depth is a dead end, it
means it was never trained for. Real fix: sample a random depth each
batch (e.g. `depth in {2,4,6,8,12}`) or curriculum (early: 2-4, middle:
2-8, late: 2-12/16), same shared weights throughout. Evaluate the
resulting model across the SAME depth range it trained on.

**The real test, not just stability**: does `accuracy(d=12) >
accuracy(d=4)` on genuinely HARD examples? If additional trained
recurrent depth measurably helps on hard cases (not just "doesn't
break"), that is a real test-time-compute mechanism and a significant
result. If it doesn't, that's real too — report it honestly either way.

## Step 6 — The integrated 25M HZ candidate

Once Steps 1/3/5 each independently work (Step 2/4 optional, additive):

```text
Faithful BDH
  + value bottleneck (D/4) + INT8 state      [Step 1]
  + 50% BlockBDH, trained end-to-end          [Step 3]
  + variable-depth training                   [Step 5]
```

Deliberately excluded for now: MoE, separate HZ-B-style memory, fast
weights, ternary weights, synthetic-gradient training. Keep the
integrated candidate clean — one real test of whether the three proven
wins compose, not a kitchen sink.

Compare, at 25M params, matched tokenizer/data/optimizer/budget:
upstream BDH, matched Transformer, HZ candidate. Measure:

- **Quality**: language CE, passkey, reassignment, harder memory tasks,
  code/reasoning if available.
- **Efficiency**: total RAM, decode throughput, prefill throughput,
  joules/token (CUDA-only for now, per the standing Mac-`powermetrics`
  gap), active FLOPs.
- **Dynamic reasoning**: accuracy vs. internal depth curve.

## The real decision gate

HZ advances to 100M only if the integrated model shows MULTIPLE
simultaneous advantages, not one isolated win:

```text
State RAM:        >=10x below exact BDH's own state
Total inference RAM: >=30% below matched Transformer at relevant context
Decode:            >=1.5x faster
Quality:           <=3-5% degradation, or better
Stateful tasks:    competitive or better
Trained depth:     measurably helps hard tasks
```

Not all six are required, but more than one real, simultaneous
advantage is — the missing thing right now is exactly this: memory,
speed, and quality wins have all been shown SEPARATELY, never together
in one trained model.

## Status

Not started. Real next bit-by-bit steps, in order: lock HZ-State-v1
(cheap, mostly done already) -> Step 3's 5-seed BlockBDH stability
check (bounded, well-defined) -> Step 5's variable-depth training ->
Step 6's integrated candidate, only once 1/3/5 each work independently.
