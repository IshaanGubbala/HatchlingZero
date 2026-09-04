# HatchlingZero — Mainline Research Plan

## 0. North Star

HatchlingZero has one primary objective:

\[
\boxed{\textbf{Beat BDH-CQ at its own game}}
\]

That means producing a system that is simultaneously:

1. **Smarter** — higher ARC/reasoning accuracy.
2. **Faster** — lower training and inference latency.
3. **Cheaper** — fewer parameters, less VRAM, less energy, fewer GPU-seconds.
4. **More scalable with compute** — harder tasks should benefit from additional recurrent reasoning.
5. **Deployable** — improvements should survive real generation, not only teacher-forced validation loss.

The long-term target is therefore a better **quality–compute Pareto frontier**, not simply a lower LM validation loss.

---

# 1. The Scoreboard

Every serious architecture should eventually be measured on the same scoreboard.

## Intelligence
- ARC pass@1
- ARC pass@2
- exact-match task accuracy
- per-byte/token accuracy
- accuracy vs task difficulty
- accuracy vs recurrent depth \(R\)

## Compute
- inference GPU-seconds/task
- training tokens/sec
- autoregressive tokens/sec
- peak VRAM
- parameter count
- trainable parameter count
- approximate energy/cost

## Critical reasoning test

The main signature we are trying to create is:

\[
\boxed{
\text{harder task}
\Rightarrow
\text{larger useful }R
}
\]

Ideally:

\[
A(R=2)<A(R=4)<A(R=8)
\]

for sufficiently difficult tasks.

If increasing \(R\) does not improve correctness, we do **not** call the recurrence “reasoning” regardless of what validation loss does.

---

# 2. What We Have Already Learned

These findings are considered locked unless a direct replication overturns them.

## KEEP: adaptive gated residual writes

Our strongest recurrent-dynamics result.

The state-dependent adaptive gate:
- improved LM quality substantially,
- beat fixed scalar and state-independent controls,
- eliminated the catastrophic collapse at \(R=12/16\),
- demonstrated real, although tiny, state dependence.

Current lesson:

\[
\boxed{
\text{controlled state-dependent writes improve and stabilize recurrence}
}
\]

This belongs in the main architecture.

---

## KEEP: exact/high-fidelity addressing

Repeated experiments showed addressing is extremely sensitive.

Failures included:
- Q/K subspace compression,
- sparse masks,
- routing,
- candidate neuron pruning,
- key-state compression,
- static/shared support masks.

Even high-energy SVD approximations caused incorrect neuron competition.

Rule:

\[
\boxed{
\text{do not approximate the addressing side without overwhelming evidence}
}
\]

---

## KEEP: value/output-side compression

The value/output side has been much more tolerant.

The rank-64 decoder/subspace approach remains useful.

General rule:

\[
\boxed{
\text{high fidelity before selection, compression after selection}
}
\]

---

## KEEP: reduced width where validated

The old \(m=32\) setting was substantially oversized.

\(m=16\) provided roughly half the nominal compute with only a small quality penalty.

This should remain part of the efficiency-oriented baseline unless a CQ-specific test disproves it.

---

## KEEP: weight tying

Untied versions were significantly worse even when parameter matched.

The shared recurrent operator is load-bearing.

---

## KEEP: compiler-friendly dense computation

Real profiling showed the bottleneck was not theoretical FLOPs or HBM bandwidth.

It was:
- many small kernels,
- sequential dispatch,
- poor GPU utilization,
- elementwise overhead.

`torch.compile` produced roughly a 2× real speed improvement and collapsed hundreds/thousands of kernels.

Rule:

\[
\boxed{
\text{optimize graph geometry before inventing sparsity}
}
\]

---

# 3. What We Have Killed

Do not casually revive these.

## DEAD: compressed BDH-Δ belief/workspace architecture

The:
- 384-d belief bottleneck,
- 8×96 workspace,
- separate Think Cell,
- compressed persistent carry,
- predictor/corrector machinery

produced approximately:

\[
1.7862
\]

vs the much stronger BDH baseline family.

The model actively suppressed parts of the mechanism.

Lesson:

\[
\boxed{
\text{do not force BDH through a new low-dimensional latent coordinate system}
}
\]

---

## DEAD: naive “more recurrence = reasoning”

Multiple experiments showed recurrence can refine features without performing sequential reasoning.

Old models:
- peaked around \(R=2–4\),
- degraded at high \(R\),
- failed difficulty→required-depth scaling.

The adaptive gate fixed **stability**, but not reasoning.

Important distinction:

\[
\boxed{
\text{stable recurrence} \neq \text{iterative reasoning}
}
\]

---

## DEAD: same-final-answer supervision at every round

Supervising every recurrent state against the final answer caused the network to decode the answer immediately and hold it.

That explicitly rewards shortcutting.

Do not repeat.

---

## DEAD: round embeddings as previously implemented

They did not improve reasoning and slightly hurt LM quality.

---

## DEAD: addressing-side routing/sparsification

Hardware and quality both worked against it.

Dense GEMMs won.

---

## DEAD: generic MoE/value experts as tested

Small improvement at best, gate went nearly to zero, not worth architectural complexity.

---

# 4. Status of HZ-CQ-v0

HZ-CQ-v0 served as a diagnostic, not the final architecture.

It has two major problems.

### Problem 1: reasoning is actually sequence growth

Each additional “reasoning step” creates more sequence positions.

Therefore:

\[
R\uparrow
\]

also changes:
- sequence length,
- answer position,
- positional geometry,
- compute structure.

That makes it impossible to cleanly interpret \(R\) as latent compute.

### Problem 2: no true CQ persistent memory/workspace

It does not implement:

\[
S_t=U(S_{t-1},D_t)
\]

with a fixed-sized task memory.

And it does not implement:

\[
H_{r+1}=F(H_r,S)
\]

where the same workspace evolves repeatedly.

Therefore:

\[
\boxed{\text{HZ-CQ-v0 is now a completed diagnostic branch}}
\]

We should stop investing architecture work into v0.

---

# 5. What v0 Told Us

The corrected ARC training experiment is useful.

After fixing the answer-target bug and increasing ARC exposure:

- LOW \(R\) had worse held-out loss.
- MEDIUM \(R\) had best held-out loss.
- HIGH \(R\) worsened again.

However, raw per-byte accuracy was:

\[
R_{\rm LOW}: 0.6673
\]

\[
R_{\rm MED}: 0.6688
\]

\[
R_{\rm HIGH}: 0.6696
\]

Essentially identical.

Therefore:

\[
\boxed{
\text{extra }R\text{ currently changes calibration/generalization, not correctness}
}
\]

This is the important negative result.

We should not spend more time tuning v0's \(R\).

---

# 6. MAINLINE PHASE 1 — Build HZ-CQ-v1

## Single question

> Can we build a faithful BDH-CQ-style architecture where recurrent depth is genuine latent compute?

Nothing else matters until this is answered.

---

## 6.1 Persistent task memory \(S\)

Demonstrations should update a fixed-size persistent state:

\[
S_t=U_\theta(S_{t-1},E(D_t)).
\]

Use something like:

\[
S\in\mathbb{R}^{M_S\times D}
\]

with:

\[
M_S\approx4-16.
\]

Important:

- keep \(D\) high,
- no 384-d bottleneck,
- no raw demonstration sequence retained during query reasoning,
- adaptive gated writes,
- exact addressing where needed.

Conceptually:

\[
S=\text{“What transformation/rule defines this task?”}
\]

---

## 6.2 Fixed recurrent workspace \(H\)

Query reasoning should operate on:

\[
H_r\in\mathbb{R}^{M_H\times D}.
\]

For example:

\[
M_H=4\text{ or }8.
\]

Then:

\[
H_{r+1}=F_\theta(H_r,S,x).
\]

The SAME slots evolve every round.

No new sequence positions.

Therefore:

\[
R=16
\]

really means sixteen applications of the recurrent reasoning operator.

---

## 6.3 Use the validated adaptive gate

Workspace update:

\[
\Delta H_r=F_\theta(H_r,S,x)
\]

\[
g_r=C_\phi(H_r,\Delta H_r,S)
\]

\[
H_{r+1}
=
\operatorname{LN}
(H_r+g_r\Delta H_r).
\]

Start close to the known-good residual regime rather than random behavior.

---

## 6.4 Do NOT add yet

While building v1, do **not** add:

- reasoning LoRA,
- adaptive halting,
- evidence caching,
- candidate verification,
- CoT distillation,
- RL,
- new MoE,
- sparse routing,
- extra belief modules.

The first architecture must be deliberately boring.

We need to know whether:

\[
\boxed{
S+H+\text{gated recurrence}
}
\]

works by itself.

---

# 7. PHASE 1A — Structural Tests Before Training

Before spending meaningful GPU money, v1 must pass cheap correctness tests.

## Task memory tests

Verify:

1. Different demonstrations produce different \(S\).
2. Demo ordering affects \(S\) when appropriate.
3. Multiple demos accumulate information.
4. Query-time memory cost does not grow with raw demo length.
5. Removing one demonstration changes behavior.
6. Memory is actually used by the query pathway.

---

## Workspace tests

Verify:

1. \(H\) has fixed shape for every \(R\).
2. \(R=2\) and \(R=16\) differ only by recurrent computation count.
3. Sequence length is identical across \(R\).
4. Increasing \(R\) does not allocate additional token positions.
5. Gradients flow through the recurrent computation.
6. Adaptive gates remain numerically stable through \(R=16+\).

### Promotion gate

Do not train seriously until all of these pass.

---

# 8. MAINLINE PHASE 2 — Does v1 Actually Use Reasoning Depth?

This is the most important experiment in the project.

Train the simplest possible v1 on episodic reasoning data.

Use:
- ARC training episodes,
- procedural ARC-like tasks,
- composition,
- ordering,
- nested transformations,
- graph/path problems,
- symbolic rewriting,
- state-machine execution.

Training should expose multiple \(R\):

\[
R\in\{2,4,6,8,12,16\}.
\]

Use the same architecture and tied weights.

---

## Evaluation

Hold task fixed and sweep:

\[
R=\{1,2,4,6,8,12,16\}.
\]

Record:

\[
\text{accuracy}(R)
\]

and task difficulty.

### SUCCESS

We see:

\[
\boxed{
\text{difficulty}\uparrow
\Rightarrow
R_{\rm optimal}\uparrow
}
\]

Even a modest reproducible effect is enough to continue.

### FAILURE

Accuracy stays essentially flat across \(R\), like v0.

If that happens:

\[
\boxed{
\text{do NOT compensate by simply training longer}
}
\]

We then investigate why the recurrent operator cannot implement staged computation.

That becomes the next architecture problem.

---

## 8.5 Real result, 2026-09-02/03

Section 8's own FAILURE branch fired -- and its own instruction was
followed: investigated why, rather than compensating by training
longer. Real, condensed summary of a real, extensive investigation
(full detail in `plans/newnewplan.md`, dozens of real local
experiments, all committed):

**Two real synthetic task families tested, both at real 150K-300K-step
training scale**, not just quick checks:

- **Composed-permutation lookup** (K symbols, demos show full
  input/output pairs of a composed rule): real ICL emerges at scale
  (99.9% accuracy, S+H vs. a same-budget standard Transformer stuck at
  chance -- section 1's "beat BDH-CQ" scoreboard's first real
  quality-per-parameter win). But depth x R sweep shows only an
  \(R{=}1\) vs. \(R{\ge}2\) threshold effect, nothing beyond -- FAILURE
  per section 8's own criterion.
- **Genuinely-sequential FSM traversal** (answer requires real
  incremental state-tracking, not a precomputable lookup): real
  learning (well above chance) but the same FAILURE signature -- no
  \(R\)-dependence at any depth, confirmed at n=2000/cell (kill
  criterion computed exactly as section 17 Rule 3 specifies: deltas
  \(<1\)pp, well below the 1-2pp threshold).

**Real "why" investigation (the FAILURE branch's own instruction)**,
three real hypotheses tested in order, two ruled out cleanly, one
confirmed:

1. **Training budget** -- ruled out. Doubling real exposure (150K ->
   300K steps, continued from a real saved checkpoint) produced no
   change in accuracy.
2. **The gate is simply broken** -- ruled out, and productively so.
   Direct instrumentation shows the gate behaves in OPPOSITE ways on
   the two task families: collapses to near-zero by round 5 on the
   lookup task (correctly recognizing it needs no more rounds), stays
   at \(\approx1.0\) every round on the FSM task (correctly recognizing
   it needs every round). The controller is task-sensitive, not
   broken -- real evidence \(\boxed{g_r}\) works as section 2's own
   "controlled state-dependent writes improve and stabilize
   recurrence" finding predicts.
3. **Workspace capacity (\(M_H\))** -- **confirmed real.** \(M_H{=}32\)
   vs. the locked \(M_H{=}8\): +3.04 percentage points mean accuracy,
   clean at n=2000/cell (noise floors 0.83pp and 0.44pp respectively --
   the effect is ~4x the noise). \(R\) still shows no effect even at
   \(M_H{=}32\).

**Real, updated verdict**: v1's real lever for this task class is
**state capacity, not recurrent depth**. Section 6.2's original
\(M_H\in\{4,8\}\) lock (deliberately small, for the "boring first
pass") measurably left real accuracy on the table. This does not
close the "does depth ever matter" question -- it was tested at a
fixed \(M_H\), and now needs retesting at the confirmed-better
capacity, on a task where depth SHOULD matter more than a lookup-
reducible one (composed permutation already saturates near-ceiling
even at \(M_H{=}8\), so it's a poor instrument for this).

**Real, concrete next steps, not yet run**: ~~(a) find where the
\(M_H\) benefit saturates (64? higher?)~~ -- DONE, 2026-09-03, see
below; (b) retest depth x R at the confirmed-better \(M_H\) on the FSM
task specifically, now that capacity is no longer confounding the
measurement; (c) real checkpoints and a parameterized script family
(`scripts/hz0h_bdh_hzcq_v1_*` in the working tree, referenced from
`plans/newnewplan.md`) exist for whoever continues this without
needing to rebuild the harness.

**Real result, 2026-09-03: M_H=64 saturation point found.** Same
harness (`scripts/hz0h_bdh_hzcq_v1_fsm_depth_r_experiment.py
--workspace-slots 64 --allow-ablation-slots`), same 150K steps,
n=2000/cell eval. Mean accuracy 0.3761 vs \(M_H{=}32\)'s 0.3774 --
**-0.13pp, flat, well inside the ~0.8pp noise floor at depth=16**. The
capacity benefit that was real and clean from \(M_H{=}8\to32\)
(+3.04pp) does NOT continue from \(32\to64\): \(M_H{=}64\) still beats
\(M_H{=}8\) by roughly the same margin as \(M_H{=}32\) did (+2.91pp),
consistent with the earlier finding, but adds nothing further.
**Real, confirmed saturation point: the capacity lever tops out
somewhere between \(M_H{=}32\) and \(M_H{=}64\)** -- \(M_H{=}32\) is
the Pareto-efficient choice found so far (same accuracy as 64, half
the workspace parameters). \(R\) is still not a reliable lever at
\(M_H{=}64\) either: depth=16 R4->R8 is -0.20pp (flat), R4->R12 is
+1.65pp (above the 0.8pp noise floor this single run, but
inconsistent with R8's flat result -- a non-monotonic single-run
number, not treated as confirmed without replication, same discipline
as every other borderline signal this session). Gate magnitude is
still ~1.0 at every round at \(M_H{=}64\) too -- a fourth confirmation
that the gate-stays-open behavior on this task is capacity-independent.
Checkpoint saved (`results/local/hz0h_bdh_hzcq_v1_fsm_mh64_checkpoint.pt`).

---

# 9. MAINLINE PHASE 3 — Reasoning LoRA

Only start this after v1 demonstrates that recurrent depth matters.

The LoRA infrastructure already works.

Current evidence:

- rank 16:
  - 1.65M trainable params,
  - ~0.79% of model,
  - useful adaptation.
- rank 64:
  - 6.59M trainable params,
  - ~3.09%,
  - substantially closes the gap to full fine-tuning.

This proves LoRA is a useful **training-capacity mechanism**.

It does NOT yet prove reasoning transfer.

---

## The actual reasoning-LoRA experiment

Use LoRA primarily on tolerant pathways:

- encoder/value,
- decoder/output,
- recurrent write/update,
- possibly adaptive controller.

Avoid Q/K/addressing initially.

Run four matched arms:

### A — base v1

Normal full architecture training.

### B — v1 + reasoning LoRA

LoRA remains active at inference.

This asks:

> Does extra low-rank training capacity improve reasoning?

### C — v1 + LoRA → anneal LoRA scale to zero

\[
\alpha:
1\rightarrow0.75\rightarrow0.5\rightarrow0.25\rightarrow0.
\]

This asks:

> Can the base recurrent system absorb the learned reasoning behavior?

### D — LoRA teacher → base student distillation

Teacher:

\[
\text{base}+\text{LoRA}
\]

Student:

\[
\text{base only}.
\]

Distill outputs and possibly useful recurrent trajectories.

---

## Jackpot result

If:

\[
B>A
\]

and:

\[
C\approx B
\]

or:

\[
D\approx B,
\]

then we have:

\[
\boxed{
\text{extra capacity during training}
\rightarrow
\text{better reasoning}
\rightarrow
\text{zero inference overhead}
}
\]

That would be a major HatchlingZero advantage.

---

# 10. MAINLINE PHASE 4 — Curriculum / Data

Only after the architecture uses recurrence correctly.

Then attack BDH-CQ's weaknesses deliberately.

Build procedural generators for:

- transformation composition,
- ordering,
- nesting,
- object/property binding,
- conditional rules,
- extrapolation,
- arithmetic carry chains,
- graph paths,
- program execution.

The key is controlled complexity.

Example:

\[
A
\]

then:

\[
A\circ B
\]

then:

\[
A\circ B\circ C.
\]

Measure whether increasing composition depth causes optimal \(R\) to rise.

This is much more useful than simply adding millions of random examples.

---

# 11. MAINLINE PHASE 5 — Make It Faster

Only after the quality architecture is locked -- unchanged from the
original rule, now sharpened with real HZ-CQ-v1 evidence (Rule-1's
depth x R investigation, 2026-09-02) instead of being purely
hypothetical.

## 11.0 Strict research order (unchanged, now explicit as a gate)

\[
\boxed{
\text{prove genuine sequential depth reasoning first}
\rightarrow
\text{optimize execution}
\rightarrow
\text{test evidence-waterfall / reduced-refresh architecture changes}
}
\]

Nothing below this line runs ahead of the still-open genuinely-
sequential-task test (section 8 addendum, in progress as of this
writing). Speed work on a recurrence whose real depth-dependence is
still unresolved risks locking in kernels/caching around the wrong
mechanism.

## 11.1 What today's real evidence says about the bottleneck

**Explicit, evidence-based statement**: current evidence does **not**
show HBM/VRAM-bandwidth saturation anywhere in v1. The real,
instrumented finding from the gate-magnitude diagnostic (composed-
permutation task) is that recurrent rounds are cheap individually but
the adaptive gate learns to self-collapse (\(g_r\) from 0.82 at round 1
to ~0.01-0.04 by round 5+) -- meaning most of any wall-clock cost past
the first 2-3 rounds is currently being spent on rounds whose own
learned gate says they barely matter. The real, likely bottleneck
class is **sequential recurrence dependency + fragmented/
underutilized compute + redundant per-round work** -- not bandwidth.
Every optimization below should be read against that diagnosis, not a
generic "make the GPU busier" framing.

## 11.2 Classification key

Every item below is tagged:

- **[DO NOW]** -- zero-semantic-change. Provably identical output by
  construction (pure caching/hoisting/preallocation of values that do
  not change). Safe to land without a depth-reasoning-track pause.
- **[BENCH]** -- semantics-preserving in principle, but implementation
  details (fused-kernel numerics, GEMM packing order, compiled-graph
  behavior) can introduce small floating-point differences. Requires
  an exact-equivalence check (section 11.4) before trusting downstream
  results against it.
- **[PARK]** -- architectural experiment. Changes real behavior, not
  just execution mechanics. Explicitly does not run until the
  sequential-reasoning question is resolved.

## 11.3 The concrete optimization sequence

### 1. Cache fixed K/V once per query -- **[DO NOW]** -- DONE, 2026-09-03

Real, verified fact about the current code
(`reference/hz0h_bdh_hzcq_v1_reasoning_workspace_torch.py`): inside
`HZCQReasoningWorkspace.step`, `read_s(H_prev, S)` and
`read_x(H_prev, x_hidden, x_mask)` recompute
\(K_S=W_k^S S,\ V_S=W_v^S S,\ K_x=W_k^x x_{hidden},\ V_x=W_v^x
x_{hidden}\) on **every** call, even though \(S\) and \(x_{hidden}\)
are the same tensors passed unchanged into every round -- only \(Q\)
(a function of the evolving \(H_r\)) actually changes round to round.
Compute \(K_S,V_S,K_x,V_x\) once before the round loop, reuse across
all \(R\) rounds. Exact attention semantics unchanged -- this is
provably the same computation, just not repeated.

**Real result**: implemented in `_ExactCrossAttention.project_kv`/
`.attend` plus `HZCQReasoningWorkspace._step_with_cache`, with `run()`
now computing \(K_S,V_S,K_x,V_x\) once before the round loop instead
of once per round. Verified bit-identical (`torch.equal`, not just
`allclose`) against the old per-round-recompute path via `step()`
called manually in a loop, on a real (B=2, M_S=8, M_H=8, D=32) case at
R=12. All 15 existing structural tests (workspace + memory) still
pass unchanged. Measured (CPU, forward-only, D=80, M_H=8, R=16, B=16,
200 reps after 5 warmup): 1.0027s -> 0.8773s, **1.14x**. Modest at
this R -- expected, since this only removes redundant K/V projection
GEMMs, not the dominant per-round attention/gate cost -- but real,
free, and zero-risk. Item 5 (below) landed in the same change since
it's the same redundant-recompute pattern.

### 2. Factory/pipeline execution -- **[BENCH]**

Restructure the round loop so fixed memory/evidence (\(S\),
\(K_S,V_S,K_x,V_x\) from item 1) stays resident and \(H\) flows through
repeated update stages, rather than the current per-round pattern of
"go back to memory, recompute, write, gate, repeat." Explore
`torch.compile`/Inductor on the whole round function (not just
individual ops) so \(Q\to\text{attention}\to\text{write}\to
\text{gate}\to H_{r+1}\) executes with minimal intermediate
materialization and fewer kernel launches. Needs the equivalence check
(11.4) since compiled/fused execution can reorder floating-point
operations.

### 3. Evidence-cart/waterfall architecture -- **[PARK]**

Future ablation, not a current change: one exact, high-fidelity memory
access produces a small multi-vector evidence bank \(E\); several
cheap \(H\) microsteps then consume/update against that fixed bank
before the next real refresh. This is a genuine architecture change
(changes what each round can see, not just how fast it computes) --
explicitly parked until the sequential-reasoning test (section 8
addendum) resolves. Related to, but distinct from, section 12's
refresh-cadence reduction (8/8 vs 6/8) -- that work is about
CADENCE of exact refreshes on the existing mechanism; this is about
replacing the per-round read target with a cached evidence bank. Keep
these two threads separate when either resumes.

### 4. Fuse tiny ops -- **[BENCH]**

Gate features (RMS, cosine similarity), SiLU, sigmoid, the gated
residual add, and LayerNorm are all small, cheap, elementwise-or-near-
elementwise operations -- real candidates for kernel fusion
(`torch.compile` or manual fusion). Individually cheap, but many small
kernel launches are a real, previously-measured source of overhead
elsewhere in this project (section 2's "KEEP: compiler-friendly dense
computation" -- the same lesson likely applies here). Needs the
equivalence check since fused reductions can change floating-point
rounding order.

### 5. Remove unnecessary per-round work -- **[DO NOW]** for provably
loop-invariant work, **[BENCH]** for anything touching numerics --
the `s_summary` instance DONE, 2026-09-03

Real, verified instance in the current code: `HZCQReasoningWorkspace._gate`
recomputes `s_summary = S.mean(dim=1, keepdim=True)` on **every**
call, even though \(S\) is invariant across the whole `run()` call --
identical redundant-recompute pattern to item 1, just inside the gate
rather than the attention read. Hoist it out once per `run()` call.
**Real result**: `_gate` now takes an optional `s_summary` param;
`run()` computes it once and threads it through `_step_with_cache`.
`step()` itself is untouched (still recomputes on every direct call,
for the diagnostic scripts and tests that call it standalone) --
verified bit-identical against `run()`'s new cached path, same test as
item 1 above. The general sweep for other loop-invariant work (masks,
dtype/device conversions, unnecessary `.clone()`) has NOT been done
yet -- only this one verified instance landed.
General sweep, same treatment: redundant summaries, masks recomputed
from the same inputs, unnecessary dtype/device conversions, and
`.clone()` calls that aren't protecting against real aliasing --
anything genuinely loop-invariant is [DO NOW]; anything that changes
which values flow into a matmul (e.g. an intermediate materialization
that was also serving as a real numerical checkpoint) gets [BENCH]
treatment first.

### 6. Pack compatible projections/GEMMs -- **[BENCH]** -- DONE, 2026-09-03

Where exact semantics allow (e.g. `read_s.q_proj`/`read_x.q_proj` are
both applied to the same \(H\) -- could become one wider GEMM split
after the matmul instead of two separate `nn.Linear` calls). Real,
exact equivalence to verify: concatenated-weight-matrix GEMMs must
produce bit-identical (or float-tolerance-identical) results to the
separate calls before trusting this.

**Real result**: implemented as `HZCQReasoningWorkspace._packed_q`
(concatenates `read_s.q_proj.weight` and `read_x.q_proj.weight` into
one (2D, D) matrix, runs one `F.linear`, splits the result) and
`_ExactCrossAttention.attend_with_q` (accepts a precomputed Q instead
of projecting internally). Uses the exact same `Parameter` tensors as
before -- no weight copies -- so gradients flow to
`read_s.q_proj.weight`/`read_x.q_proj.weight` exactly as they did
pre-packing (verified directly). `step()` and direct `read_s`/`read_x`
calls (tests, diagnostic scripts) are untouched; only `run()`'s
internal `_step_with_cache` uses the packed path. Verified bit-
identical (`torch.equal`) against `step()`'s unpacked path on the same
fixture used for items 1/5, and all 15 structural tests still pass.
Measured (CPU, forward-only, D=80, M_H=8, R=16, B=16): combined with
items 1+5, 1.125x vs the fully-naive baseline -- packing two D×D GEMMs
into one 2D×D barely moves the needle at this small D on CPU, as
expected; the real payoff (fewer kernel launches) should show more on
GPU. Landed anyway since it's real, verified, and free.

### 7. Preallocate recurrent buffers -- **[DO NOW]**

Avoid allocations inside the \(H\) loop -- pure memory-management
change, produces identical numerical results, only removes allocator
overhead. Especially relevant for \(H\)'s fixed-shape
\((B, M_H, D)\) state across every round (section 7's own structural
tests already prove the shape never changes across \(R\) -- exactly
the property that makes preallocation safe here).

### 8. Adaptive early exit -- **[PARK]**

Real, current evidence (gate-magnitude diagnostic, composed-
permutation task): gates collapse to near-zero by round 5, on a task
that does NOT require genuine sequential reasoning (section 8
addendum's own conclusion). That collapse is suggestive but **not
sufficient** evidence to enable hard early-stopping -- it could mean
"this task never needed more than 2 rounds" (a fact about the task) or
"the gate always collapses regardless of real task difficulty" (a
fact that would matter for hard-stopping generally). Do not enable
hard early exit until the genuinely-sequential task (section 8
addendum) establishes whether gate collapse timing tracks real task
difficulty or is a fixed, task-independent behavior.

## 11.4 Profiling checklist -- run per optimization, before/after

Every item above that lands (whether [DO NOW] or after a [BENCH]
verification) gets measured on all of:

- **latency** (ms/batch, matching the real methodology already used
  in section 8's speed comparisons -- warmup passes before timing)
- **throughput** (episodes or tokens/sec)
- **kernel count** (real, not estimated -- via profiler trace)
- **GPU utilization** (%, when running on real CUDA hardware, not the
  local CPU/MPS-only measurements taken so far)
- **peak VRAM** (or peak RSS for local CPU runs)
- **GPU-seconds/task** (the real cost metric from section 1's
  scoreboard, not just raw speed)
- **exact output/accuracy equivalence** -- for every [BENCH] item,
  confirm the optimized path produces the same (or float-tolerance-
  identical) logits/accuracy as the unoptimized baseline on a fixed
  real eval set BEFORE trusting any speed number from it. A faster
  wrong answer is not a real optimization.

---

## 11.5 Cross-platform architecture-level speed plan — CUDA + MPS

Items 1, 5, and 6 above are DONE (bit-identical K/V + `s_summary`
caching, and packed `read_s`/`read_x` Q projections; see their DONE
notes in 11.3) -- real, but small (1.125x combined, CPU-only,
D=80/M_H=8), and only measured on this Mac's CPU so far. That's the
gap this subsection exists to close.

**Restating the current, real evidence this plan is now built on**
(not re-deriving it -- see section 8.5 for the full writeup): S+H
shows real fresh-rule ICL; \(M_H=32\) is a confirmed positive lever
over \(M_H=8\) (+3.04pp mean, n=2000/cell, FSM task); recurrent depth
\(R\) is still essentially flat at both \(M_H=8\) and \(M_H=32\); the
gate diagnostic shows the bottleneck class is sequential-dependency +
fragmented per-round work, not demonstrated HBM/VRAM-bandwidth
saturation (11.1).

**The goal here is explicitly NOT CUDA-only kernel hacking.** The goal
is to change computation *geometry* so the architecture gets faster on
BOTH CUDA and Apple MPS, by reducing:

1. the number of dispatched operations,
2. the number of expensive evidence reads,
3. the number of recurrent rounds actually executed,
4. intermediate materializations,

while preserving exact/high-fidelity Q/K addressing (section 2's
"KEEP: exact/high-fidelity addressing" -- still locked, still not up
for renegotiation by anything in this subsection).

`torch.compile`/Inductor is an optional CUDA accelerator here, **not**
the architectural foundation of this plan -- section 2's "KEEP:
compiler-friendly dense computation" 2x number came from the OLD BDH
architecture, not HZ-CQ-v1, and CUDA-specific compilation wins don't
by themselves demonstrate a real cross-platform geometry win. Every
item below must show its benefit independently on MPS and CUDA before
being trusted.

### SPEED-A — Batched dual-source attention [BENCH FIRST, semantics-preserving target]

\(H\) currently performs two separate attention reads every round --
one against \(S\), one against \(x_{hidden}\) (`read_s`, `read_x` in
`HZCQReasoningWorkspace.step`/`_step_with_cache`). Design an execution
form that runs both as ONE batched backend operation wherever
possible, built on top of the already-landed packed-Q (item 6) and
cached-K/V (item 1) work.

Constraints:
- preserve separate softmax normalization domains for \(S\) and
  \(x_{hidden}\) -- **do not** concatenate them into one softmax
  distribution, that changes the model, not just its execution;
- retain separate learned Q/K/V parameters unless a later, explicitly
  named ablation tests sharing them;
- conceptually stack the two attention problems along a synthetic
  batch/head/source axis, execute through one batched matmul/SDPA-
  style call, then split the two read results back out.

Desired effect: two attention dispatch sequences -> one larger,
backend-friendlier dispatch sequence.

**Promotion gate**: verify output equality (or extremely tight
numerical equivalence) against the current path; verify gradients;
run all structural tests (section 7); benchmark BOTH MPS and CUDA;
record forward latency, training-step latency, throughput, peak
memory, and dispatch/kernel count where measurable (11.4's checklist,
run on real hardware this time, not just CPU).

### SPEED-B — Refresh-and-refine recurrent cell [ARCHITECTURE ABLATION]

The most expensive semantic operation is rereading \(S\) and \(x\)
every round. Introduce a two-timescale recurrent design:

\[
\text{EXPENSIVE outer refresh: } E_j = \text{Read}(H_j, S, x)
\]
\[
\text{CHEAP inner refinement: } H_{j,k+1} = H_{j,k} + \alpha\, g\, \Phi(H_{j,k}, E_j)
\]

where `Read` is the current exact/high-fidelity S/x evidence
attention, \(\Phi\) is a small FULL-DIMENSIONAL local recurrent update
(no low-dimensional BDH-\(\Delta\)-style bottleneck -- section 3's
"DEAD: compressed BDH-\(\Delta\) belief/workspace architecture" is not
being revived here), Q/K addressing stays exact, and \(E\) is reused
for several cheap \(H\)-only microsteps before the next real refresh.

Test refresh ratios \(K=1\) (exact current baseline), \(K=2\),
\(K=4\). Goal: cut expensive S/x cross-attention calls ~2-4x while
still allowing several real state transitions. Do NOT assume the old
cached-evidence results (section 12's 8/8 vs 6/8 refresh-cadence work,
or item 3's parked evidence-cart/waterfall note in 11.3) transfer --
treat this as a new HZ-CQ-v1 ablation. This is the same general
"reduce expensive refresh" family as section 12 and item 3, approached
from a different angle (a two-timescale cell vs a cadence schedule on
the existing per-round mechanism, or a cached evidence bank) -- keep
these three threads clearly labeled and don't merge their results.

**Kill criterion**: if \(K=2\) does not yield a meaningful
cross-platform speed improvement, OR loses more than 1pp reasoning
accuracy without a compensating Pareto gain, stop before \(K=4\).
Measure actual GPU/MPS wall-clock speed, not FLOP estimates.

### SPEED-C — Post-hoc adaptive early exit [ARCHITECTURE/INFERENCE POLICY]

The gate has shown real, confirmed task sensitivity (8.5, and the
gate-diagnostic script): on lookup-like tasks it collapses after a few
rounds; on genuinely sequential FSM tasks it stays open. Section 11.3
item 8 already parked hard early-exit pending exactly this kind of
task-sensitivity evidence -- that evidence now exists, so this is the
follow-up. Section 13 (Adaptive Compute) is the *trained*,
model-decides-\(R(x,S)\) version of this idea, gated on \(R\) first
being shown to help difficult tasks; SPEED-C is deliberately smaller
and comes first -- a POST-HOC stopping rule over a frozen, already-
trained model, not a new halting network.

Use a post-hoc stopping rule based on multiple signals, e.g.:
minimum \(R \geq 2\); mean gate magnitude below threshold for two
consecutive rounds; AND/OR predictive KL/logit change below threshold;
AND/OR \(\cos(H_r, H_{r-1})\) indicating convergence. Evaluate
thresholds on frozen trained trajectories. Do not use gate magnitude
alone as the only stop signal.

Report: accuracy, average realized \(R\), p50/p95 realized \(R\),
latency, GPU/MPS seconds per task.

**Promotion criterion**: same accuracy within statistical noise, with
a substantial reduction in mean realized \(R\) and wall-clock latency.

### SPEED-D — Compressed value/write pathway only [SECONDARY ARCHITECTURE ABLATION]

Only after A-C are characterized. Test reducing the dimensionality of
the VALUE/WRITE side while keeping Q/K/addressing full-dimensional:
\(Q,K\) dimension stays \(D\); \(V\)/write dimension becomes \(D/2\)
or \(D/4\); project back to \(D\) only at the recurrent residual
write. Directly motivated by section 2's "KEEP: value/output-side
compression" (`high fidelity before selection, compression after
selection`) -- this is that same locked lesson, applied to \(H\)'s
write pathway specifically.

Do NOT: compress Q/K; introduce sparse routing (section 3's "DEAD:
addressing-side routing/sparsification" stays dead); create many tiny
GEMMs; use dimensions so small that kernel-launch overhead defeats the
nominal FLOP savings. Benchmark \(D\), \(D/2\), \(D/4\).

### Cross-platform benchmark contract

Fixed benchmark matrix using the current quality-relevant workspace
configuration, especially \(M_H=32\), not only \(M_H=8\). At minimum:
\(M_H=32\); \(R=2,4,8,16\); representative short and long query
lengths; batch 1 and a training-relevant batch size; fp32 where
required on MPS plus the normal production precision on CUDA.

For every speed change, record: MPS latency; CUDA latency;
training-step time; throughput; peak memory; kernel/dispatch count
where available; output/accuracy equivalence; parameter count;
realized recurrent rounds if adaptive exit is involved. Do not call a
change a speed win based only on CPU measurements or theoretical
FLOPs -- the 1.125x measured for items 1/5/6 above is a CPU number and
must not be reported as this section's cross-platform result once real
MPS/CUDA numbers exist.

**Real result, 2026-09-03: priority-1 item DONE on both MPS and CUDA.**
`scripts/hz0h_bdh_hzcq_v1_speed_benchmark_mps_cuda.py`, M_H=32
(confirmed Pareto point), R in {2,4,8,16}, batch in {1,16}, 50 reps
after 5 warmup, real on-device bit-identical equivalence check
(`torch.equal`, max_abs_diff=0.0) on both platforms before trusting
any timing number, per this section's own required discipline.

**MPS** (real Apple Silicon GPU via `torch.backends.mps`): forward-only
mean 1.017x, essentially noise (0.952x-1.072x) -- matches the earlier
CPU finding, packing/caching barely moves forward latency at this
D/M_H. Full training-step (forward+backward+optimizer): mean
**1.096x**, always \(\geq\)1.0x across all 8 combinations, up to
1.162x, batch-size-independent.

**CUDA** (real RTX 5090 via RunPod): a genuinely DIFFERENT, mixed
pattern -- worth reporting exactly as measured, not smoothed over.
Forward-only is the consistent winner here: mean **1.104x**, always
\(\geq\)1.04x. Training-step is **batch-dependent**: at batch=1, real
regressions at 3 of 4 R values (0.805x-1.014x -- the packed-Q GEMM's
backward pass doesn't amortize its own overhead at this batch size);
at batch=16, real gains at all 4 R values (1.026x-1.115x). Mean across
all 8 combinations is close to neutral (0.987x) -- CUDA training-step
speedup is real only at the larger, training-relevant batch size, not
uniformly like MPS.

**Honest overall verdict**: items 1/5/6 are a small, real, free win on
every platform tested for forward-only latency, and a real win for
training-step latency PROVIDED batch size isn't tiny (MPS: always;
CUDA: only at batch=16, not batch=1). Not a uniform win to report
without the batch-size caveat -- exactly the kind of nuance 11.4's
profiling checklist exists to surface.

Both raw results: `results/local/hz0h_bdh_hzcq_v1_speed_benchmark_mps.json`,
`results/local/hz0h_bdh_hzcq_v1_speed_benchmark_cuda.json`.

### Priority order

1. finish/benchmark the already-landed semantics-preserving
   packing/caching (items 1, 5, 6) on real MPS + CUDA, not just CPU
   -- DONE on both MPS and CUDA (see above);
2. SPEED-A batched dual-source attention;
3. SPEED-C post-hoc adaptive early exit;
4. SPEED-B refresh-and-refine;
5. SPEED-D value/write compression.

C before B: adaptive exit can potentially remove whole recurrent
rounds without changing the trained architecture, whereas
refresh-and-refine changes what evidence each round receives -- the
cheaper, less invasive diagnostic goes first.

---

# 12. MAINLINE PHASE 6 — Reduce Expensive Evidence Refresh

Only after v1 quality is stable.

We already have evidence that fewer exact refreshes create a quality/compute frontier.

Previously:

- 8/8 = highest quality
- 6/8 = modest degradation
- 4/8 = larger degradation
- 2/8 had curriculum confounds

So CQ-v1 should later test:

\[
8/8
\]

vs:

\[
6/8.
\]

Do not jump immediately to aggressive 4/8.

If:

\[
6/8
\]

retains nearly all CQ reasoning accuracy while providing meaningful speedup, it becomes the efficiency variant.

---

# 13. MAINLINE PHASE 7 — Adaptive Compute

Only after:

\[
R\uparrow
\]

has been shown to improve difficult tasks.

Then teach the model to decide:

\[
R(x,S).
\]

Easy problem:

\[
R=2-4.
\]

Hard problem:

\[
R=8-16.
\]

This is where we can potentially beat BDH-CQ strongly on cost.

Instead of LOW/MEDIUM/HIGH manually:

\[
\boxed{
\text{model allocates compute according to task difficulty}
}
\]

Possible signals:
- update magnitude,
- gate magnitude,
- state-change norm,
- evidence disagreement,
- prediction confidence.

But avoid irregular GPU branching initially.

Train fixed/static budgets first.

---

# 14. MAINLINE PHASE 8 — Candidate Verification / pass@2

Once pass@1 is respectable.

Generate:

\[
y_1,y_2,\dots
\]

then score them against inferred task memory \(S\).

Possibly:

\[
V(S,x,y_i).
\]

Then return the best candidates.

This attacks BDH-CQ on its own pass@2 evaluation while potentially being cheaper than blindly sampling multiple full solutions.

Later possibility:

\[
\text{reason}
\rightarrow
\text{candidate}
\rightarrow
\text{verify}
\rightarrow
\text{correct}
\]

inside latent space.

---

# 15. Chat-Capable Model

This is a real project milestone, but it is NOT allowed to derail CQ architecture validation.

The existing 150M probe already showed:
- real words,
- local syntax,
- repetition,
- technical/code-heavy generations.

The corpus is the obvious problem: we have not meaningfully trained on conversational English.

Therefore:

\[
\boxed{
\text{chat milestone waits until v1 recurrent architecture is validated}
}
\]

Then train the smallest coherent conversational model using a proper English/chat corpus.

This gives us another human-readable quality benchmark later.

---

# 16. Parking Lot

These ideas are explicitly NOT active unless the mainline reaches the appropriate phase.

### Parked
- Coconut-style latent CoT
- explicit CoT distillation
- RL/verifiable reward
- adaptive halting
- candidate verifier
- dynamic refresh
- LoRA banks
- per-channel gates
- predictor/corrector
- alternative workspace layouts
- larger MoE
- sparse execution
- special CUDA kernels
- chat SFT
- giant synthetic curriculum

They are not deleted.

They are simply forbidden from interrupting the current experiment.

### Parked, with an explicit place and prerequisites (not side branches)

The "Paper-Derived Reasoning Upgrades — Controlled Queue" section
(after section 19) gives several of the above items a real, ordered
path back into the mainline instead of leaving them as an undated
list: adaptive halting -> PAPER-0's forced-exit diagnostic first, then
SPEED-C's post-hoc exit (11.5), then section 13's trained version, in
that order; dynamic refresh -> section 12's 8/8-vs-6/8 cadence work
and SPEED-B's refresh-and-refine (11.5) are the two named, separate
threads, neither of which has started; alternative workspace layouts
-> PAPER-2 through PAPER-4; RL/verifiable reward -> explicitly gated
behind PAPER-6, and only after a supervised value-guided baseline
proves branching latents contain useful diversity, per that section's
own "not permission to immediately add RL" line. None of this changes
their PARK status -- section 19's decision tree still gates entry into
this queue on the v1 negative branch, which is the branch we're
actually on.

---

# 17. Operating Rules to Prevent Drift

## Rule 1 — One main research question at a time

Current question:

\[
\boxed{
\text{Can faithful HZ-CQ-v1 make additional }R\text{ improve reasoning accuracy?}
}
\]

Everything else waits.

---

## Rule 2 — One architecture change per experiment

Never bundle:

\[
\text{new memory}
+
\text{LoRA}
+
\text{refresh}
+
\text{halting}
\]

into one run.

Otherwise we learn nothing.

---

## Rule 3 — Every experiment needs a kill criterion before running

Example:

> If v1 produces <1–2 percentage points of reproducible accuracy improvement from \(R=4\rightarrow8/12\) on deep tasks, do not claim depth reasoning.

---

## Rule 4 — Real task metrics beat proxy losses

Priority:

\[
\text{exact task accuracy}
>
\text{per-byte accuracy}
>
\text{CE}
>
\text{training loss}.
\]

Validation loss is a diagnostic, not the goal.

---

## Rule 5 — Architecture first, optimization second

Do not optimize kernels for an architecture that may be killed tomorrow.

---

## Rule 6 — No expensive scaling before mechanism validation

Do not throw 500M tokens at something that fails a cheap controlled synthetic task.

---

## Rule 7 — Keep a champion

At every stage maintain:

\[
\boxed{\text{current best quality baseline}}
\]

and:

\[
\boxed{\text{current best efficiency baseline}}
\]

Every new candidate must beat one meaningfully.

---

# 18. Immediate Execution Queue

This is the only task list that matters right now.

### STEP 1
Implement real fixed-size persistent \(S\).

### STEP 2
Unit-test \(S\).

### STEP 3
Implement fixed-size recurrent workspace \(H\).

### STEP 4
Integrate adaptive gated updates.

### STEP 5
Verify \(R\) changes compute depth without changing sequence length.

### STEP 6
Run tiny procedural reasoning smoke tests.

### STEP 7
Train minimal v1 with variable \(R\).

### STEP 8
Run paired difficulty × \(R\) evaluation.

Then STOP and inspect the result.

---

# 19. Decision Point

## If v1 shows useful depth scaling

Proceed:

\[
\boxed{
\text{v1}
\rightarrow
\text{reasoning LoRA}
\rightarrow
\text{curriculum}
\rightarrow
\text{systems optimization}
\rightarrow
\text{refresh reduction}
\rightarrow
\text{adaptive compute}
}
\]

## If v1 does NOT show useful depth scaling

Do NOT move to LoRA, giant datasets, RL, or optimization.

Instead:

\[
\boxed{
\text{debug the recurrent state-transition mechanism itself}
}
\]

because the architecture still does not know how to use sequential computation.

Real, current status (2026-09-03): this is the branch we're actually
on -- \(M_H\) capacity is confirmed positive, \(R\) is still flat at
every \(M_H\) tested. The concrete debug plan for this branch is the
"Paper-Derived Reasoning Upgrades — Controlled Queue" section
immediately below.

---

# Paper-Derived Reasoning Upgrades — Controlled Queue

This section translates recent 2026 recurrent-reasoning papers into
falsifiable HatchlingZero experiments, without turning the project
into a collection of simultaneous architecture changes (section 17
Rule 2 -- one architecture change per experiment -- applies to every
item below exactly as it does everywhere else in this plan). This is
the concrete execution plan for section 19's negative branch: "debug
the recurrent state-transition mechanism itself."

## Restating the current evidence this queue is built on

- S+H has demonstrated strong fresh-rule ICL (section 8.5, 8/8.5).
- \(M_H\) capacity is a confirmed positive lever: \(M_H=32 > M_H=8\)
  by +3.04pp on the clean FSM evaluation (n=2000/cell, section 8.5).
- Additional \(R\) is still essentially useless for accuracy at both
  \(M_H=8\) and \(M_H=32\).
- On lookup-like tasks (composed permutation) the adaptive gate
  collapses after a few rounds.
- On genuinely sequential FSM tasks the gate stays approximately fully
  open -- yet accuracy still does not scale materially with \(R\).
- Therefore the problem is **not** simply "the model refuses to
  compute" -- the gate is doing something real and task-sensitive, and
  compute is genuinely happening every round.
- The open problem: **how do we shape \(H_r \to H_{r+1}\) so that
  successive states form a useful reasoning trajectory**, rather than
  R applications of an operator whose extra applications the model has
  learned it doesn't need?

The items below are an ORDERED experimental queue, not a set of
proven improvements to HatchlingZero. None of the source papers have
been reproduced here -- these are HatchlingZero-native, falsifiable
adaptations of their ideas, evaluated against this project's own
tasks, kill criteria, and existing locked findings (section 2/3).

---

### PAPER-0 — Forced-exit trajectory diagnostics

Source: "Adaptive Depth in Looped Transformers: Diagnosing Learned
Halting Gates and Trajectory Readouts" (2026, arXiv:2607.20519).

Before changing \(H\) at all, instrument a SINGLE recurrent
trajectory. For the same episode and same forward trajectory,
decode/read out \(H_1, H_2, H_3, \ldots, H_R\) and measure, at every
round: answer accuracy; correct-answer logit/margin; entropy;
predictive KL between round \(r\) and \(r-1\); \(\cos(H_r, H_{r-1})\);
normalized \(\|H_r - H_{r-1}\|\); gate magnitude.

Critical distinction: do NOT infer trajectory quality only by
rerunning the model with different total \(R\) (that's what the depth
x R sweeps already did). Inspect intermediate states from the SAME
trajectory.

Questions: does correctness progressively improve? Does the state
change while the prediction stays flat? Does the trajectory oscillate?
Does it converge? Are later states destroying useful earlier
information?

Cheapest and FIRST paper-derived experiment. No architecture change.

**Real result, 2026-09-03**
(`scripts/hz0h_bdh_hzcq_v1_paper0_forced_exit_diagnostic.py`, loading
the confirmed M_H=32 checkpoint, depth=16, R=16, n=200 episodes, one
real trajectory, no training): a clean, real, precise mechanistic
signature -- **the READOUT freezes, not the state.**

Rounds 1-4: accuracy fluctuates (0.390 -> 0.380 -> 0.370 -> 0.370),
predictive KL between consecutive rounds is real and shrinking
(0.00197 -> 0.00112 -> 0.00008). By round ~5-6 accuracy locks to
EXACTLY 0.3750 and stays bit-for-bit identical through round 16;
predictive KL between consecutive rounds hits 0.00000 from round 6
onward -- the classifier's output distribution stops changing at all.

But \(\|\Delta H\|\) (the round-to-round state-change norm) does NOT
shrink to match -- it stays large and stable (~1.93) for the entire
remaining 10 rounds, larger than round 1's own \(\Delta H\) in most of
those rounds. \(H\) keeps moving substantially every round; the
answer stops listening. \(\cos(H_r,H_{r-1})\) is consistently
*negative* (~-0.11 to -0.16, stabilizing) -- each round's update has a
real anti-correlated component with the previous state, not a
diminishing one.

**Real, updated mechanistic picture**: the earlier finding (gate stays
~1.0, so "the model isn't refusing to compute") is now sharper --
compute keeps happening (H keeps moving, gate stays open), but by
round ~5-6 that movement lands somewhere the READOUT (the
`rq`/`rk`/`rv`/classifier cross-attention against \(H\)) has become
insensitive to. This is a genuinely new candidate explanation for the
flat-\(R\) result that neither "gate collapse" nor "training budget"
captured: not a stopped computation, but a computation whose products
stop reaching the answer. Directly motivates PAPER-1 next -- is this a
real attractor in the READOUT's effective output space even though raw
\(H\)-space keeps drifting?

Result: `results/local/hz0h_bdh_hzcq_v1_paper0_forced_exit_mh32.json`.

### PAPER-1 — Attractor/convergence diagnostic

Source: "Equilibrium Reasoners: Learning Attractors Enables Scalable
Reasoning" (2026, arXiv:2605.21488).

Test whether \(H\) behaves like a useful task-conditioned dynamical
system. For a fixed problem: create multiple small perturbations of
the initial \(H_0\); run identical recurrence; compare trajectories
for correct and incorrect examples.

Measure: pairwise trajectory distance over \(r\); cosine convergence;
prediction convergence; whether correct runs converge toward a shared
basin/state; whether harder tasks take more iterations to converge.

Desired signature: difficulty up -> convergence time up, and
convergence strength correlates with correctness. If \(H\) never
converges toward solution-aligned regions, that's a real mechanistic
explanation for flat accuracy(\(R\)).

Diagnostic only -- do not copy Equilibrium Reasoners wholesale yet.

**Real result, 2026-09-03**
(`scripts/hz0h_bdh_hzcq_v1_paper1_attractor_diagnostic.py`, same
M_H=32 checkpoint, depth=16, R=16, 60 episodes x 6 perturbations of
\(H_0\) each, perturbation std ~10% of \(H_{init}\)'s own std): \(H\)
IS a real, strongly contracting dynamical system -- and the
contraction is fast and total, independent of correctness.

Mean pairwise \(H\)-distance across the 6 perturbed starts collapses
from 0.0367 (round 1) to 0.0000 (round 11+), roughly halving every
round. Prediction agreement across perturbations hits 100% by round
2 and stays there. Splitting episodes by whether the (unperturbed)
model got the real answer right (n=23) or wrong (n=37) shows **no
meaningful difference**: both groups start at essentially the same
\(H\)-distance (0.0371 vs 0.0364) and both converge to 0.0000 by
round 16, agreement 1.0 either way. EqR's desired signature
(difficulty up -> convergence time up, convergence strength
correlates with correctness) does **not** hold here -- convergence is
fast and complete regardless of whether the answer ends up right.

**Real, combined picture with PAPER-0**: \(H_r\) contracts onto a
single trajectory determined almost entirely by \((S,x)\) within
~10 rounds, independent of \(H_0\) -- but PAPER-0 already showed that
trajectory keeps moving substantially in absolute terms
(\(\|\Delta H\|\approx1.93\), non-shrinking) even after this
contraction and even after the READOUT has frozen (round ~5-6). Put
together: the recurrence is a genuine contracting map (small
perturbations vanish fast, this is a real, nontrivial dynamical
property, not automatic for a gated nonlinear recurrence), but the
attractor it contracts onto is a *moving* trajectory shaped by
\((S,x)\) alone, not a fixed point tied to solving the task -- and the
readout commits to an answer (right or wrong) well before that
trajectory's own long-run behavior is decided. This is consistent
with, and sharpens, section 8.5's verdict: \(R\) doesn't help because
extra rounds mostly ride an already-committed, task-correctness-blind
trajectory, not because the model "refuses" or "runs out of training."

Result: `results/local/hz0h_bdh_hzcq_v1_paper1_attractor_mh32.json`.

### PAPER-2 — Identity-biased / LayerScale \(H\) recurrence

Source: "Thinking Deeper, Not Longer: Depth-Recurrent Transformers for
Compositional Generalization" (2026, arXiv:2603.21676).

The FIRST real architecture ablation in this queue. Current update:
\(H_{r+1} = \operatorname{LN}(H_r + g_r \Delta H_r)\). Potential
problem: the repeated post-update normalization and unconstrained
residual may continuously rewrite the state instead of maintaining an
information highway.

Create ONE controlled alternative with: explicit identity-biased
recurrence; a small learned LayerScale \(\alpha\) initialized close to
zero/small residual; final-answer-only supervision (NOT
same-final-answer-at-every-round, section 3's "DEAD: same-final-answer
supervision at every round" stays dead); same \(S\); same readout;
same \(M_H\); same parameter budget as closely as possible; same
adaptive gate unless mathematically redundant.

Candidate form: \(u_r = F(H_r, S, x)\), \(H_{r+1} = H_r + \alpha\, g_r\, u_r\),
with normalization arranged pre-update or within \(F\) so the direct
identity path \(H_r \to H_{r+1}\) isn't repeatedly renormalized away.

Do NOT change workspace capacity, training data, readout, addressing
fidelity, or evidence sources in the same experiment (Rule 2). Run
paired difficulty x R tests.

**Promotion criterion**: a reproducible difficulty-up -> useful-R-up
effect that exceeds the existing 1-2pp kill criterion (section 8/8.5).
If it only improves fixed-R accuracy but R stays flat, record that
honestly as an optimization/capacity result, NOT recurrent reasoning
-- exactly the same honesty standard the M_H=32 finding was held to.

**Real result, 2026-09-03: FAILED, on both axes.** Same FSM task,
same M_H=32, same 150K steps/n=2000 eval protocol as the LN baseline,
`--identity-biased --layerscale-init 0.1`. Mean accuracy 0.3276 vs the
LN baseline's 0.3774 -- **-4.98pp, a real, clean regression** (noise
floor at depth=16 is 0.57pp, this is ~9x that). Not just "R stays
flat" -- fixed-R accuracy actively got WORSE. Depth=16 R4->R8 +0.55pp,
R4->R12 +0.95pp, both at or below the noise floor -- no real R-effect
here either.

**Real, striking mechanistic finding, not predicted going in**: the
gate did not stay open (as in the LN baseline) or stay collapsed
gradually (as in the composed-permutation task's earlier gate-collapse
finding) -- it **slams shut after round 1, at every depth**: g goes
from ~0.999 (round 1) to ~0.0001 or exactly 0.0 (round 2 onward),
identically across all 5 tested depths. Removing the per-round
LayerNorm means unbounded repeated additions genuinely can compound
without correction -- the model appears to have learned to protect
itself from that by doing all its real writing in round 1, then
closing the gate hard for the remaining 15 rounds. A clean, real,
reproducible behavior, just not the "useful information survives
longer" effect the hypothesis predicted.

**Verdict**: identity-biased/LayerScale recurrence, as specified here,
is not a promotion -- it is a real, honest negative result on both the
accuracy axis and the depth-scaling axis, plus a genuine new
mechanistic observation (gate learns to hard-close under an unbounded
residual) worth keeping in mind for PAPER-3/4's own residual-bounding
designs. Per the queue's own strict order, PAPER-2 has now "failed to
produce useful depth scaling" -- PAPER-3 is unblocked.

Checkpoint:
`results/local/hz0h_bdh_hzcq_v1_fsm_paper2_identity_biased_mh32_checkpoint.pt`.
Result: `results/local/hz0h_bdh_hzcq_v1_fsm_paper2_identity_biased_mh32.json`.

### PAPER-3 — Bounded residual + evidence re-injection

Source: "Latent Recurrent Thoughts: Recurrent Refinement of Proposed
Latents for Reasoning with Frozen LLMs" (2026, arXiv:2609.01117).

Only if PAPER-2 fails to produce useful depth scaling. Borrow the
PRINCIPLE, not the exact architecture: bounded residual latent
corrections; continually re-anchor recurrence to the original
task/query evidence; do not allow \(H\) to drift arbitrarily far from
its evidence-conditioned starting representation.

Do NOT reproduce the previous BDH-\(\Delta\) low-dimensional
bottleneck (section 3, still dead) -- all reasoning states stay
full-dimensional \(D\). Test first as \(H_{r+1} = H_{base} +
\text{bounded\_correction}(H_r, S, x)\) or an equivalent bounded
residual formulation. One change at a time.

**Real result, 2026-09-03: FAILED, same as PAPER-2, but a genuinely
different mechanism.** Implemented as \(H_{r+1}=H_{base}+g_r\cdot
\text{bound\_scale}\cdot\tanh(\Delta H_r)\), \(H_{base}\) = \(H_{init}\)
cross-attended once against \(S\) (fixed every round, zero new
parameters -- reuses `read_s`), correction hard-capped via tanh
regardless of \(\Delta H_r\)'s own scale. Same FSM task, M_H=32, 150K
steps, n=2000/cell, same protocol as baseline and PAPER-2. Trained
locally on this Mac's CPU in 6320s (~105min) -- real finding from
earlier tonight held: CPU beats a RunPod RTX 5090 for this tiny
sequential workload, so no GPU dispatch was needed here.

Mean accuracy: 0.3285, essentially indistinguishable from PAPER-2's
0.3276 (+0.09pp, noise) and still -4.89pp below the LN baseline's
0.3774. R still shows no real effect (depth=16 R4->R8 -0.35pp,
R4->R12 +0.40pp, both at/below the 0.32pp noise floor).

**The real mechanistic difference from PAPER-2**: the gate did NOT
slam shut here -- it settled at a stable, moderate ~0.08-0.09 at
every round past round 1, at every depth (vs baseline's ~1.0 and
PAPER-2's ~0.0001). Bounded re-anchoring genuinely avoided the hard
gate-collapse failure mode PAPER-2 produced. But this didn't help
accuracy at all -- the most likely explanation: re-anchoring EVERY
round to the SAME fixed \(H_{base}\) means each round's correction is
computed relative to that fixed point, not relative to what the
previous rounds actually wrote -- so information can't meaningfully
compound across rounds even though the correction itself survives
(isn't destroyed by renormalization or gate-collapse). The state
literally cannot accumulate a multi-round computation this way; each
round is close to an independent, small nudge away from the same
anchor, not a chained refinement.

**Verdict**: PAPER-2 and PAPER-3 have now both failed on the accuracy/
depth-scaling axes, via two different, real, well-understood
mechanisms (unbounded-residual gate-collapse vs bounded-but-static-
anchor no-accumulation). Per the queue's strict order, PAPER-4 (fast
scratch / slow integrator) is next -- and is now better motivated than
before: it explicitly separates a fast, disposable per-round
computation from a slow, persistent integrator that's allowed to
actually accumulate across rounds, which PAPER-3's fixed-anchor design
structurally prevented.

Checkpoint:
`results/local/hz0h_bdh_hzcq_v1_fsm_paper3_bounded_residual_mh32_checkpoint.pt`.
Result: `results/local/hz0h_bdh_hzcq_v1_fsm_paper3_bounded_residual_mh32.json`.

### PAPER-3b — Bounded, accumulating recurrence (inserted control, 2026-09-03)

Real, deliberate insertion between PAPER-3 and PAPER-4, not from any
source paper -- PAPER-2 and PAPER-3 each changed TWO things relative
to the default at once, in different combinations, so neither cleanly
answers "was PAPER-2 bad specifically because the residual was
unbounded?" PAPER-2 kept real accumulation (\(H_r \to H_{r+1}\) builds
on \(H_r\)) but left the correction unbounded (gate hard-collapsed);
PAPER-3 bounded the correction but re-anchored every round to a fixed
\(H_{base}\) (no accumulation at all). This isolates exactly the
missing cell: real accumulation AND a hard bound, together.

\[
H_{r+1} = H_r + \beta\,\tanh(g_r \Delta H_r)
\]

\(\beta\) is a FIXED (not learned) hyperparameter, deliberately --
PAPER-2's learned \(\alpha\) drifted from 0.1 to 0.359 during training,
which could itself have contributed to the instability; fixing
\(\beta\) removes that confound from this specific comparison. No
post-update LN, no second state, zero new parameters (verified:
identical param count to the default path). Same S, same readout,
same M_H, same everything else (Rule 2).

Real interpretation going in:
- if accuracy recovers toward the LN baseline -> the missing
  distinction was found: bounded-but-accumulating is what PAPER-2 and
  PAPER-3 each independently got half of;
- if accuracy recovers but \(R\) still stays flat -> recurrence
  stability improved, the underlying "no genuine multi-round reasoning"
  problem is still open;
- if it also lands around ~33% (PAPER-2/3's shared result) -> stop
  iterating on residual-update variants and move to PAPER-4's
  structurally different two-state design instead.

### PAPER-4 — Fast scratch / slow integrator

Inspired by the two-timescale direction in Latent Recurrent Thoughts
and related hierarchical recurrent-reasoning work. Only after
PAPER-2/3. Introduce two FULL-DIMENSIONAL state roles: \(H_{fast}\)
(temporary scratch computation) and \(H_{slow}\) (persistent
integrated reasoning state). Example: \(H_{fast} \leftarrow
F_{fast}(H_{fast}, H_{slow}, S, x)\) several times, then \(H_{slow}
\leftarrow H_{slow} + \alpha\, F_{slow}(H_{slow}, H_{fast})\).

Constraints: no compressed 384-d belief state; no 8x96 BDH-\(\Delta\)
recreation; no new low-dimensional latent coordinate system; weight
tying preferred; compare against a parameter-matched baseline.

Hypothesis: temporary computation should not repeatedly overwrite the
integrated reasoning state. Promotion criterion remains difficulty ->
larger useful compute depth.

### PAPER-5 — Breadth before extreme depth

Source: "Generative Recursive Reasoning" / GRAM (2026,
arXiv:2605.19376). A v2-level experiment, NOT immediate mainline.

Current \(H\) is deterministic: \(H_r \to H_{r+1}\). GRAM motivates
testing \(H_r \to \{H_{r+1}^{(1)}, \ldots, H_{r+1}^{(B)}\}\) -- multiple
latent hypotheses explored in parallel. Do NOT start with a large
branching tree.

Initial experiment: breadth \(B \in \{1,2,4\}\); shallow recurrent
depth; shared transition parameters; simple stochastic proposal; fixed
total compute comparison against deeper deterministic recurrence.

Question: at the SAME inference compute, is breadth > depth for tasks
where the deterministic trajectory commits to the wrong latent
solution? Do not promote without a compute-matched comparison.

### PAPER-6 — Value-guided latent search

Source: "Q-Learning With World Models" (2026, arXiv:2608.17163),
combined conceptually with GRAM. Only after PAPER-5 shows useful
diversity among latent trajectories.

Conceptual HatchlingZero analogue: proposal \(F(H_r, z_i, S, x) \to
H_{r+1}^{(i)}\); evaluator \(Q_\phi(H_r, z_i, S, x) \to\) predicted
downstream solution value. Generate several candidate latent updates,
score them, retain only the best 1-4.

Important: this is NOT permission to immediately add RL; first test
supervised/verifiable value prediction on procedural tasks where
correctness is known; keep candidate breadth tiny; compare against
deterministic \(H\) at matched GPU-seconds; measure whether additional
test-time compute actually improves correctness. Only consider
RL/Q-learning after a supervised value-guided search baseline proves
branching latent states contain useful selectable diversity.

---

## Strict experiment order

\[
\boxed{
\begin{aligned}
&1.\ \text{PAPER-0 forced-exit probe} \\
&2.\ \text{PAPER-1 attractor probe} \\
&3.\ \text{finish the current } M_H \text{ capacity curve / current controlled work} \\
&4.\ \text{PAPER-2 identity-biased LayerScale recurrence} \\
&5.\ \text{PAPER-3 bounded residual/evidence anchoring} \\
&6.\ \text{PAPER-4 fast-scratch / slow-integrator} \\
&7.\ \text{PAPER-5 stochastic breadth} \\
&8.\ \text{PAPER-6 value-guided latent search}
\end{aligned}
}
\]

Never combine two paper-derived architectural mechanisms in the first
experiment testing either one (Rule 2, again).

## Required scoreboard for every paper-derived architecture

Always record: accuracy by task difficulty; accuracy by \(R\);
fixed-compute accuracy; parameter count; wall-clock latency; peak
memory; gate trajectory; intermediate-state/forced-exit accuracy;
convergence metrics; MPS and CUDA performance once the mechanism
survives local validation (see 11.5's cross-platform benchmark
contract -- the same discipline applies here).

The PRIMARY success criterion remains:

\[
\boxed{
\text{harder task} \rightarrow \text{larger useful recurrent/test-time compute}
}
\]

A plain fixed-R accuracy improvement is useful but does NOT prove
reasoning-depth scaling.

## Explicit anti-drift rules

Do not: revive Q/K compression (section 2, locked); revive sparse
routing (section 3, dead); recreate BDH-\(\Delta\)'s low-dimensional
belief/workspace (section 3, dead); use same-final-answer supervision
at every round (section 3, dead); add LoRA, RL, branching search, a
new readout, a new workspace, and a new loss simultaneously; call
stochastic breadth a win without compute matching; call an
R-dependent loss improvement reasoning unless correctness also
improves.

---

# 20. Project in One Line

For the foreseeable future:

\[
\boxed{
\textbf{Build faithful CQ recurrence → prove more compute creates more reasoning → then make it cheaper.}
}
\]

Everything else is secondary until that chain works.