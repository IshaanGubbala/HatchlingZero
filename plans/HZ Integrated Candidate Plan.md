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

**Status: CLOSED (2026-08-12).** Both formulations tried failed --
direct grouped state (hard depth-block assignment, reproducible 44%
ceiling even with zero sharing) and this learned-addressing redesign
(bimodal, seed-dominated failure at every bank count including zero
sharing, `docs/restart/hz0h_phase2r_step2_soft_grouped_state_results.md`).
The second failure is arguably worse than the first (no stable outcome
at all, vs. a reproducible-but-low ceiling). Grouped-state compression
is archived, not pursued further -- no formulation #3 planned.

## Step 3 — BlockBDH: moved to its own research lane (2026-08-12), not canonical

Real status per `docs/restart/hz0h_phase4_blocksparse_results.md`
Updates 6-8: 50% active gives a genuine, measured 1.95-6.20x decode
speedup and 1.00 accuracy on passkey, but the reassignment task shows
real seed instability (0.60-1.00 across 5 seeds) that survived THREE
mechanistically distinct fix attempts (noise injection, constant-lambda
balance loss, annealed balance loss) -- all three show the identical
failure signature (help whichever seeds needed it, hurt whichever
didn't, never a clean win). **6.2x measured wall-clock speedup is too
valuable a signal to discard, but three unsuccessful stability fixes
means it doesn't belong in the canonical candidate.**

Status: **`HZ-Sparse` experimental lane** -- real speed mechanism
proven, trainability unresolved. Kept alive as separate ongoing
research, NOT bundled into HZ-Core. Real next ideas (not yet tried,
mechanistically different from all 3 failed attempts, for whenever this
lane is revisited):

- **Dense-warmup anneal**: train dense first, then gradually anneal
  active fraction 100% -> 75% -> 50%, instead of training at 50% from
  step 0 -- may avoid the abrupt regime shift that seems to trigger
  router lock-in.
- **Distillation**: dense BDH teacher -> 50%-active BlockBDH student,
  rather than training the sparse model against ground truth alone.

Re-entry condition: multi-seed stability (matching Step 3's original
target, `>=0.95-1.00` consistently across `>=5` seeds) on the
reassignment task, via a mechanism genuinely different in kind from the
three already-failed attempts -- not a fourth optimizer/loss-term tweak
on the same family.

## Step 4 — Learned router, once BlockBDH's fixed heuristic router is the bottleneck

Small learned router (`p = softmax(W_r x)`, top-k block selection) with
an explicit load-balancing term (`L = L_LM + lambda * L_balance`) to
prevent block collapse. Track block utilization, router entropy,
collapse, reassignment accuracy, CE, and wall-clock together — not just
task accuracy alone. No RL-based routing yet; softmax + top-k + balance
loss only.

## Step 5 — Variable depth: REJECTED AS TESTED (2026-08-12)

Phase 5's real finding: a model trained at depth `d` does not know how
to reason at `2d` zero-shot — training depth-variation in-path (curriculum,
not i.i.d. random sampling, which failed outright) fixes that cleanly,
1.00 accuracy at every trained depth on every trained hop count
(`docs/restart/hz0h_phase5_variable_depth_results.md` Update 3).

**But the actual hypothesis this step exists to test failed**: on the
held-out hard task, accuracy is HIGHEST at low depth (0.935 @ d=4) and
falls monotonically as depth increases (0.87 @ d=16) -- the opposite of
"more compute helps harder problems." The model learned a
difficulty-independent solution (6-hop chains solved at only 2
iterations), not "one hop per iteration," so depth stopped being the
thing doing any differentiating work. Trainability isn't the open
question here anymore -- the mechanism trains fine. The premise is
falsified.

Status: **REJECTED AS TESTED.** Not integrated into HZ-Core. Could
reopen later under a fundamentally different objective than plain
next-token loss -- intermediate latent targets, process supervision, or
RL explicitly rewarding successful extra computation -- but there is no
current evidence this belongs in HZ-Core, and no further hyperparameter
variants of the tested mechanism are planned.

## Step 6 — HZ-Core-1: the minimal integrated 25M candidate (renamed/narrowed 2026-08-12)

Real reassessment, per this step's own stated precondition ("once Steps
1/3/5 each independently work"): Step 3 does NOT independently work
(`docs/restart/hz0h_phase4_blocksparse_results.md` Updates 6-8 — real
seed instability on the reassignment task, 3 mechanistically distinct
fix attempts all failed the same way, left open and unresolved). Step 5
does NOT independently work either, in the sense that matters for this
plan's own gate: "trained depth measurably helps hard tasks" is one of
the decision-gate criteria below, and it is now FALSIFIED, not just
unproven (`docs/restart/hz0h_phase5_variable_depth_results.md` Update
3 — accuracy on the held-out hard task DECREASES monotonically as
trained depth increases). Bundling either into the 25M candidate would
make any negative result uninterpretable (real advantage from Step 1
could be masked by Step 3's seed roulette or Step 5's dead weight), and
would burn the 25M compute budget on components already known to be
broken. If it fails, three components would already be suspect and the
failure would be unattributable; if it succeeds, the extra complexity's
necessity would be unproven either way.

**Deliberately minimal: `HZ-Core-1` = Faithful BDH + Value Bottleneck +
INT8 synaptic state.** The point is to establish that the ONE mechanism
that actually survived (state compression) continues to work on a real
25M-scale language model, not just on controlled synthetic tasks.

```text
Faithful BDH
  + value bottleneck (D/4)                   [Step 1a]
  + INT8 synaptic state                      [Step 1b]
= HZ-Core-1
```

Deliberately excluded: MoE, separate HZ-B-style memory, fast weights,
ternary weights, synthetic-gradient training, BlockBDH (own `HZ-Sparse`
lane, see Step 3), variable-depth (rejected as tested, see Step 5).

**Four-way ablation**, at 25M params, matched tokenizer/data/optimizer/
budget (per `reference/hz0h_bdh_h5_memory_tasks.py`'s H5 methodology for
the task evals, and `docs/restart/hz0h_phase1_crossover_scale_sweep_results.md`'s
25M config for the matched-Transformer baseline):

```text
upstream BDH
vs BDH + VB               (isolate value bottleneck's own effect)
vs BDH + VB + INT8        (= HZ-Core-1)
vs matched Transformer
```

Measure simultaneously, not sequentially in separate one-off scripts:

- Validation cross-entropy (real held-out text, not synthetic tasks)
- Downstream reasoning/code quality, if available at this scale
- Passkey / reassignment / interference memory tasks
- Total inference RAM, state RAM specifically
- Prefill throughput, decode throughput
- Joules/token (CUDA-only for now, per the standing Mac-`powermetrics` gap)
- Long-context KV-cache crossover point

**Promotion target for HZ-Core-1**: state memory down 16-21x while
quality degradation stays <=~3%, and total inference memory meaningfully
below the matched Transformer at useful context lengths. That is a real,
publishable HatchlingZero result on its own, even without a speed win.

## The real decision gate (narrowed 2026-08-12, matching Step 6's narrowed scope)

HZ advances to 100M only if the integrated model shows MULTIPLE
simultaneous advantages, not one isolated win. "Trained depth measurably
helps hard tasks" is dropped from this list -- not because it stopped
mattering, but because it is now FALSIFIED for the mechanism tested
(`docs/restart/hz0h_phase5_variable_depth_results.md` Update 3), and
variable-depth training isn't part of the narrowed Step 6 candidate
being gated here at all:

```text
State RAM:        >=10x below exact BDH's own state
Total inference RAM: >=30% below matched Transformer at relevant context
Decode:            >=1.5x faster
Quality:           <=3-5% degradation, or better
Stateful tasks:    competitive or better
```

Not all five are required, but more than one real, simultaneous
advantage is — the missing thing right now is exactly this: memory,
speed, and quality wins have all been shown SEPARATELY, never together
in one trained model.

## Status (updated 2026-08-12)

```text
Step 1 (VB + INT8):    DONE, real reproducible win at small scale
Step 2 (grouped state): CLOSED, both formulations failed
Step 3 (BlockBDH):     moved to HZ-Sparse lane, unresolved, not canonical
Step 4 (learned router): blocked on Step 3's re-entry, not started
Step 5 (variable depth): REJECTED AS TESTED
Step 6 (HZ-Core-1):    not started -- real next step
```

Real next bit-by-bit steps, in order: confirm existing 25M-scale
infrastructure (`scripts/hz0h_stage2_runner_bdh.py`, byte-level packed
corpus at `data/packed/`, `reference/hz0h_bdh_vb_torch.py`'s `BDHVB`
fully vectorized forward, `reference/hz0a_matched_transformer.py`) with
a short smoke test before committing to a full run (the stage2 BDH
runner has never actually been executed) -> real matched-budget training
runs for all four HZ-Core-1 ablation arms -> quality + efficiency
measurement per Step 6's list -> promotion decision against the revised
gate.

Separately, independently, not blocking HZ-Core-1: `HZ-Sparse` lane
(Step 3) may resume later if a mechanistically new stability fix (dense-
warmup anneal or distillation, see Step 3) reaches multi-seed
reliability.
