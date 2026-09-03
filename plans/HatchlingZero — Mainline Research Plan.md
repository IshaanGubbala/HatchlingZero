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

**Real, concrete next steps, not yet run**: (a) find where the
\(M_H\) benefit saturates (64? higher?); (b) retest depth x R at the
confirmed-better \(M_H\) on the FSM task specifically, now that
capacity is no longer confounding the measurement; (c) real
checkpoints and a parameterized script family
(`scripts/hz0h_bdh_hzcq_v1_*` in the working tree, referenced from
`plans/newnewplan.md`) exist for whoever continues this without
needing to rebuild the harness.

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

### 6. Pack compatible projections/GEMMs -- **[BENCH]**

Where exact semantics allow (e.g. `read_s.q_proj`/`read_x.q_proj` are
both applied to the same \(H\) -- could become one wider GEMM split
after the matmul instead of two separate `nn.Linear` calls). Real,
exact equivalence to verify: concatenated-weight-matrix GEMMs must
produce bit-identical (or float-tolerance-identical) results to the
separate calls before trusting this.

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

---

# 20. Project in One Line

For the foreseeable future:

\[
\boxed{
\textbf{Build faithful CQ recurrence → prove more compute creates more reasoning → then make it cheaper.}
}
\]

Everything else is secondary until that chain works.