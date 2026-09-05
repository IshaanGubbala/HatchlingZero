# Hatchling World

**Status:** Proposed HatchlingZero mainline branch  
**Date:** 2026-09-04  
**Purpose:** Test whether HatchlingZero's persistent-memory + recurrent-workspace design becomes substantially more useful when it learns through interaction, consequences, retrieval, curriculum, and verifiable reward — while redesigning the hot loop for efficient training and inference on both Apple MPS and NVIDIA CUDA.

---

# 0. North Star

Hatchling World is **not** "put HZ in a game and hope intelligence emerges." It is a controlled research program around one hypothesis:

\[
\boxed{\text{HZ may be better matched to interactive, stateful learning than static one-shot supervision.}}
\]

The existing HatchlingZero target remains:

\[
\boxed{\text{harder task} \Rightarrow \text{more useful recurrent compute}}
\]

but the learning setting changes from:

\[
x \rightarrow y
\]

to:

\[
o_t \rightarrow \text{reason} \rightarrow a_t \rightarrow o_{t+1} \rightarrow \text{update memory} \rightarrow \text{reason again}.
\]

Long-term success means a better **quality-compute Pareto frontier**:

- higher task success,
- better action efficiency,
- fewer environment interactions needed to learn,
- useful recurrent-depth scaling,
- persistent within-lifetime learning,
- lower training wall-clock,
- lower inference latency,
- lower memory/VRAM,
- good performance on both MPS and CUDA.

---

# 1. Do Not Abandon Hatchling World After One Bad Run

Recent HZ work showed that a single architecture result can be misleading, but repeated controlled failures are meaningful. Hatchling World therefore gets a **bounded rescue ladder**.

A first negative result does **not** mean:

> Interactive learning does not work for HZ.

But this branch also cannot become unfalsifiable.

## 1.1 Verdict classes

Every serious experiment receives one of four verdicts:

1. **PASS** — clears the predeclared criterion.
2. **PROMISING / UNDERPOWERED** — useful directional signal, but evaluation/training is not decisive.
3. **FAIL-DIAGNOSE** — fails, but a concrete plausible confound gets a bounded rescue attempt.
4. **KILL / PARK** — fails after the rescue ladder and matched controls.

## 1.2 Rescue ladder

Before killing a major Hatchling World hypothesis, test failure classes in this order.

### A. Environment validity

- Is the task actually solvable?
- Is the oracle planner correct?
- Are rewards correct?
- Are train/test worlds truly distinct?
- Does difficulty really increase dependency depth/horizon?
- Can a simple reference agent learn above chance?

### B. Optimization

- Is HZ undertrained?
- Is reward too sparse?
- Is behavior cloning competence too weak before RL?
- Are LR/batch/rollout settings unstable?
- Are gradients reaching policy, memory, readout, and recurrent state paths?

### C. Curriculum / data

- Did difficulty ramp too quickly?
- Are early tasks easy enough to establish competence?
- Is the model seeing enough successful and failed trajectories?
- Does the train distribution contain the skills required at evaluation?

### D. Architecture-task interface

- Does \(S\) actually receive the information needed for within-world learning?
- Does the policy/readout consume later \(H_r\)?
- Does the environment force persistent memory to matter?
- Do later rounds receive information that could change the action?

## 1.3 Anti-rationalization kill rule

A major Hatchling World hypothesis may be parked only after:

- at least **3 independent procedural task families** fail,
- across at least **2 meaningful difficulty/horizon regimes**,
- after a verified learnable baseline succeeds,
- after obvious implementation/measurement bugs are ruled out,
- after one bounded optimization/curriculum rescue pass,
- and HZ shows no meaningful advantage in quality, sample efficiency, persistent memory use, depth scaling, or compute efficiency.

This prevents premature abandonment **without** turning the project into an endless rescue loop.

---

# 2. Inherited HZ Findings

Hatchling World starts from the current best-supported architecture. Do not redesign recurrence in the first world phases.

## KEEP: original LN recurrence

\[
H_{r+1}=\operatorname{LN}(H_r+g_r\Delta H_r)
\]

Current controlled sequence:

| Variant | Mean accuracy |
|---|---:|
| Original LN baseline | **0.3774** |
| PAPER-2 identity residual | 0.3276 |
| PAPER-3 bounded fixed-anchor | 0.3285 |
| PAPER-3b bounded accumulating | 0.3060 |
| PAPER-4 fast/slow | 0.2841 |

The update-rule branch has now gone 0-for-4. **Do not change the H update rule inside early Hatchling World experiments.**

## KEEP: \(M_H=32\)

Current capacity Pareto point:

- real gain over \(M_H=8\),
- no meaningful gain from \(M_H=64\).

## KEEP: full-fidelity Q/K

Do not compress or sparsify the selection/addressing side.

## KEEP: D/2 value/write

The half-width value/write path preserved quality:

\[
0.3786 \text{ vs } 0.3774
\]

while reducing roughly:

- **9.6% total model parameters**, and
- **23.6% workspace parameters**.

Use it as the default HZ-World efficiency configuration unless a world-specific test disproves it.

## KEEP: cached static context + packed Q

Retain:

- cached \(K_S,V_S\),
- cached \(K_x,V_x\) where semantically valid,
- cached \(S\) summary,
- packed dual Q projection.

## KEEP: compiler-friendly dense computation

Existing profiling repeatedly says HZ is often:

\[
\boxed{\text{dispatch/launch/Python-overhead bound, not FLOP bound}}
\]

So:

\[
\boxed{\text{reduce dispatch count before chasing tiny FLOP reductions}}
\]

---

# 3. The Hatchling World Learning Model

There are two timescales.

## 3.1 World time

\[
W_{t+1}=T(W_t,a_t)
\]

Observation:

\[
o_t=O(W_t)
\]

Action:

\[
a_t\sim\pi_\theta(a_t\mid o_t,S_t,H_{t,R})
\]

Environment returns:

\[
(o_{t+1},r_t,d_t)
\]

where \(d_t\) is termination.

## 3.2 Reasoning time

Before each external action:

\[
H_{t,0}=H_{\text{init}}(o_t,S_t)
\]

\[
H_{t,r+1}=F_\theta(H_{t,r},S_t,o_t)
\]

for \(r=0,\dots,R-1\).

Then:

\[
a_t=\operatorname{Policy}(H_{t,R},S_t,o_t)
\]

Thus:

\[
\boxed{\text{world steps }t \neq \text{reasoning steps }r}
\]

## 3.3 Persistent within-lifetime memory

After observing consequences:

\[
S_{t+1}=U_\theta(S_t,o_t,a_t,r_t,o_{t+1})
\]

Interpretation:

\[
\boxed{S_t=\text{what this agent has learned about this particular world}}
\]

During lifetime-memory evaluations:

\[
\theta \text{ is frozen}
\]

and only state may adapt.

---

# 4. HZ-World-0: Minimal Procedural Sandbox

Do **not** start with Minecraft, 3D vision, unrestricted language, or robotics simulation.

W0 should be:

- symbolic,
- deterministic where possible,
- procedurally generated,
- fully verifiable,
- fixed-shape/tensorizable,
- vectorizable across many parallel worlds.

## 4.1 World state

Represent each world with fixed-shape tensors describing:

- agent location,
- rooms/nodes,
- object locations,
- inventory,
- doors/locks,
- machines,
- switches,
- resources,
- hidden rule table,
- goal,
- discovered facts,
- library state later.

Conceptually:

\[
W_t=(P_t,I_t,O_t,D_t,M_t,R_{\text{hidden}},G)
\]

## 4.2 Initial actions

Keep the action vocabulary small:

- `MOVE(destination)`
- `PICKUP(object)`
- `DROP(object)`
- `USE(object, target)`
- `PRESS(target)`
- `INSPECT(target)`
- later: `READ(query)`

No free-form language actions in W0.

## 4.3 Procedural rules

Rules change between episodes.

Examples:

- colored keys open different door classes,
- two ingredients create a tool,
- switches power particular machines,
- tokens activate specific portals,
- machines require different resources,
- object effects change across worlds.

The same surface objects should support different mappings across episodes so model weights alone cannot simply memorize the solution.

---

# 5. School: Curriculum Generator

School is not just a place containing text. It is the controlled difficulty generator.

## S0 — Cause/effect

Horizon: 1–2 meaningful actions.

Examples:

- press switch -> light activates,
- pick up key -> inventory changes,
- use key -> door opens.

## S1 — Short composition

Horizon: 2–4 actions.

\[
\text{key}\rightarrow\text{door}\rightarrow\text{goal object}
\]

## S2 — Multi-step planning

Horizon: 5–8 actions.

\[
\text{resource}\rightarrow\text{tool}\rightarrow\text{room}\rightarrow\text{machine}\rightarrow\text{goal}
\]

## S3 — Hidden rules

Episode-specific rules must be inferred.

World A:

- purple keys open triangle doors.

World B:

- green keys open triangle doors.

This is where persistent \(S\) should become necessary.

## S4 — Experiment-driven learning

Some rules are not given at all.

The agent must try an action and learn from the result.

Example:

\[
\operatorname{USE}(\text{red crystal},\text{machine})\rightarrow\text{failure}
\]

Later behavior should depend on remembering that failed experiment.

## S5 — Long-horizon planning

Horizon: 10–30+ meaningful actions.

Use dependency chains with multiple subgoals.

This is the primary candidate for useful \(R\)-scaling.

---

# 6. Library: Paid External Knowledge

Introduce Library only after basic interactive competence works.

Add:

\[
a_t=\operatorname{READ}(q)
\]

which returns a bounded fact/document fragment.

Reading has a small cost:

\[
r_{\text{read}}<0
\]

so the agent learns when to:

- act from memory,
- experiment,
- or retrieve information.

Example facts:

- "Generators require charged power cells."
- "Triangle tokens unlock laboratory doors."
- "Copper and resin form a conductor."

## Library metrics

- task success,
- reads per successful task,
- unnecessary reads,
- failed actions avoided after retrieval,
- retention of facts in \(S\),
- use of a fact many steps after reading it.

Long-term behavior:

\[
\boxed{\text{remember}\leftrightarrow\text{reason}\leftrightarrow\text{retrieve}\leftrightarrow\text{act}}
\]

---

# 7. Learning Curriculum

## W0 — Environment validation

Before serious HZ training:

- procedural generator,
- deterministic transition engine,
- oracle solver,
- reward verifier,
- train/test seeds,
- horizon/difficulty labels,
- solvability tests.

## W1 — World prediction

Auxiliary objective:

\[
(o_t,a_t)\rightarrow\hat{o}_{t+1}
\]

or a compact latent transition target.

Purpose:

- teach causal structure,
- test whether state contains action-relevant information.

Do **not** treat prediction accuracy as the final intelligence metric.

## W2 — Behavior cloning warm-start

Use oracle trajectories.

Train:

\[
\pi(a_t\mid o_t,S_t,H_t)
\]

before sparse-reward RL.

## W3 — Persistent-memory challenge

Freeze weights during an episode.

Construct tasks where a rule is learned early and required much later.

Ablate:

- normal \(S\),
- zeroed \(S\),
- reset \(S\),
- optionally shuffled \(S\).

Promotion requires a real drop when memory is destroyed.

## W4 — Recurrent-depth challenge

Sweep:

\[
R\in\{1,2,4,8,16\}
\]

by task horizon and dependency depth.

Target:

\[
\boxed{\text{longer/harder tasks show larger useful }R}
\]

Use task success, not just loss.

## W5 — RL with verifiable rewards

After BC establishes competence, optimize actual outcomes.

Base reward:

\[
r_{\text{goal}}=\mathbb{1}[\text{goal achieved}]
\]

plus carefully bounded shaping such as:

- small action cost,
- invalid-action penalty,
- optional subgoal rewards only if they cannot shortcut the real objective.

No reward for "looking thoughtful."

## W6 — Group-relative trajectory optimization

For one starting world state, sample:

\[
\tau_1,\dots,\tau_G
\]

and score them with the verifier.

Optimize relative trajectory quality.

Track:

\[
\boxed{\text{reward gain per environment step and GPU-second}}
\]

not merely final success.

## W7 — Library curriculum

Add paid retrieval.

## W8 — Off-policy replay

Store:

\[
(o_t,S_t,a_t,r_t,o_{t+1},\text{world id},R,\text{success})
\]

Only test replay after the on-policy loop is stable.

## W9 — PAPER-5 revived in the right setting

PAPER-5 remains **parked, not killed**.

Revive stochastic breadth only when the environment creates meaningful competing plans.

Then breadth means:

\[
\boxed{\text{candidate plans/action strategies}}
\]

rather than arbitrary branching on a static FSM benchmark.

## W10 — PAPER-6 value-guided search

Only after PAPER-5 proves useful diversity.

Learn:

\[
Q(H,S,o,a)\approx P(\text{eventual success})
\]

and allocate compute to promising branches.

---

# 8. Frozen-Trajectory Readout Diagnostic

Run this in parallel with HZ-World.

Freeze the current good LN recurrence.

Collect:

\[
H_1,H_2,\dots,H_R
\]

from the same trajectory.

Train only probes/readouts:

1. current readout on \(H_r\),
2. linear probe per \(H_r\),
3. small MLP probe,
4. trajectory pooling over \(H_1\dots H_r\).

Interpretation:

If late states become more decodable:

\[
\boxed{\text{useful information exists; the readout is failing to exploit it}}
\]

If not:

\[
\boxed{\text{later recurrence itself is not adding solution information}}
\]

Do this before another recurrence redesign.

---

# 9. Baselines

HZ-World needs matched controls.

## Baseline A — Current best HZ

- original LN recurrence,
- \(M_H=32\),
- D/2 value/write,
- exact Q/K.

## Baseline B — Small Transformer policy

Match approximately on:

- params,
- training interactions,
- observation access,
- inference compute.

## Baseline C — Simple recurrent control

A GRU/LSTM-style agent or similarly cheap recurrent baseline.

Purpose: distinguish "interactive tasks help any recurrent model" from a real advantage of HZ's \(S+H\) structure.

---

# 10. Metrics

## Intelligence / behavior

- task success,
- success vs horizon,
- success vs hidden-rule count,
- success vs \(R\),
- action efficiency,
- recovery after failed experiments,
- generalization to unseen rule combinations.

## Persistent learning

- normal \(S\),
- reset \(S\),
- zeroed \(S\),
- delayed use of learned facts,
- facts retained across long episodes.

## Sample efficiency

- environment steps to threshold success,
- trajectories required,
- reward per 1M interactions.

## Compute

- train steps/sec,
- environment steps/sec,
- trajectories/sec,
- inference latency/action,
- latency vs \(R\),
- peak memory,
- parameter count,
- GPU utilization,
- GPU-seconds per successful task.

Core Pareto metrics:

\[
\boxed{\text{success per GPU-second}}
\]

and

\[
\boxed{\text{success per environment interaction}}
\]

---

# 11. Systems Objective: Make Hatchling World Faster Than the Current Pipeline

Current evidence indicates HZ's tiny pipeline is often dominated by:

- Python overhead,
- many small launches,
- sequential recurrence,
- CPU-side episode generation,
- host-to-device transfers,
- poor accelerator amortization.

A critical existing observation is that MPS can be slower than CPU when each step generates CPU-side data and transfers it to the GPU.

Therefore:

\[
\boxed{\text{the environment itself must be designed as an accelerator-friendly system}}
\]

---

# 12. Shared MPS + CUDA Speed Architecture

## SPEED-W0 — Vectorized worlds

Never run one Python environment per agent.

Represent many worlds as batched tensors:

\[
W_t\in\mathbb{R}^{B\times\cdots}
\]

and apply transitions in parallel.

Initial batch sweep:

\[
B_{\text{env}}\in\{32,128,512\}
\]

subject to memory.

## SPEED-W1 — Eliminate per-step host/device copies

Bad:

```text
CPU Python generates sample
→ CPU tensors
→ tiny GPU transfer
→ tiny model step
→ repeat
```

Desired:

```text
world tensors already on device
→ HZ action
→ tensorized environment transition on device
→ next HZ action
→ repeat
```

If full device-side transitions are initially impractical:

- pre-generate large rollout chunks,
- transfer batches rather than individual steps,
- reuse persistent buffers,
- avoid Python tensor construction inside the hot loop.

This is especially important for MPS because current HZ measurements already exposed host-to-device overhead as a real limiter.

## SPEED-W2 — Fixed-shape world buckets

Use a few fixed configurations:

- small,
- medium,
- long-horizon.

Keep observation/action shapes static and \(R\) drawn from a small fixed set.

Benefits:

- fewer recompiles,
- easier kernel fusion,
- easier graph replay,
- predictable memory.

## SPEED-W3 — Parallelize across worlds, not time

World time is inherently sequential within one episode.

Parallelize:

\[
\{W_t^{(1)},W_t^{(2)},\dots,W_t^{(B)}\}
\]

across independent worlds.

This preserves causal interaction while creating accelerator-sized work.

## SPEED-W4 — Keep D/2 value/write

It already preserved quality and reduced memory/params.

It did not accelerate the tiny FSM workload, but larger batched world training may move HZ toward a compute regime where smaller projection matrices matter more.

Do not claim a speed win until measured.

## SPEED-W5 — SPEED-A: batched dual-source attention

Current H reads separately from:

- persistent memory \(S\),
- current observation/query \(x\).

Implement one batched backend operation **while preserving separate softmax normalization domains**.

Do not concatenate \(S\) and \(x\) into a single softmax.

Goal:

\[
\boxed{\text{fewer dispatches without changing addressing semantics}}
\]

Promotion requires:

- output equivalence,
- gradient equivalence,
- fewer launches,
- real MPS + CUDA speedup,
- no quality regression.

## SPEED-W6 — K=2 evidence refresh

Test an architecture that performs one expensive evidence read, then two cheap refinements:

\[
E_j=\operatorname{Read}(H_j,S,o_t)
\]

\[
H_{j,1}=F(H_j,E_j)
\]

\[
H_{j,2}=F(H_{j,1},E_j)
\]

then refresh evidence.

At \(R=16\):

\[
16 \text{ expensive reads}\rightarrow 8
\]

Test \(K=2\) only first.

Kill/revise if:

- task success drops materially,
- or wall-clock improvement is negligible.

## SPEED-W7 — Adaptive early exit

Only after useful depth actually exists.

Do not use gate magnitude alone.

Candidate signals:

- action-distribution KL,
- action margin,
- hidden-state displacement,
- value confidence,
- consecutive-round agreement.

Promotion:

\[
\text{same task success within noise}
\]

with substantial reduction in:

\[
\mathbb{E}[R]
\]

---

# 13. CUDA-Specific Plan

CUDA has an additional high-value lever: graph replay.

## CUDA-1 — Compile stable model sections

Benchmark:

- eager,
- `torch.compile(..., mode="default")`,
- `mode="reduce-overhead"`,
- `mode="max-autotune"`.

Do not assume one mode wins everywhere.

Current PyTorch documentation explicitly describes `reduce-overhead` as a mode intended to reduce Python overhead using CUDA Graphs where applicable, and `max-autotune` as another GPU-oriented optimization mode.

## CUDA-2 — CUDA Graph capture

HZ-World should intentionally use static buffers/shapes so repeated rollout/model steps can be captured and replayed.

This is particularly aligned with HZ's current bottleneck:

\[
\boxed{\text{many small repeated kernels + CPU launch overhead}}
\]

Candidate capture unit:

```text
recurrent reasoning
→ policy logits
→ stable action-selection path where capturable
→ training forward/backward region where safe
```

Measure:

- launches/action,
- CPU launch time,
- latency/action,
- throughput,
- graph memory overhead.

## CUDA-3 — Persistent rollout buffers

Preallocate on device:

- observations,
- \(S\),
- \(H\),
- actions,
- rewards,
- dones,
- trajectory metadata.

Avoid allocation inside the hot loop.

## CUDA-4 — Separate inference and training benchmarks

Report separately:

- batch-1 action latency,
- batch 8/16,
- rollout-training batch,
- large vectorized environment batch.

A training optimization is not automatically an inference optimization.

---

# 14. Apple MPS-Specific Plan

MPS requires a different strategy from CUDA.

The MPS backend maps PyTorch operations onto Metal Performance Shaders / MPS Graph and tuned Metal kernels. The first priority is therefore to give MPS **large, stable, device-resident work** rather than repeatedly moving tiny tensors from CPU.

## MPS-1 — Device-side or chunked environment transitions

Highest priority.

Test:

1. tensorized transitions directly on MPS,
2. versus large pre-generated chunks transferred infrequently.

## MPS-2 — Larger vectorized world batches

Increase world count until:

- GPU utilization rises,
- per-world throughput stops improving,
- memory becomes limiting.

Report:

- total env steps/sec,
- env steps/sec/world,
- action latency,
- model steps/sec.

## MPS-3 — Profile before custom kernels

Use MPS profiling to identify:

- recurrent-cell hotspots,
- launch fragmentation,
- host/device synchronization,
- allocator overhead.

Only then consider custom Metal-backed PyTorch operations.

## MPS-4 — Custom fused recurrent op only if justified

Apple supports custom PyTorch operations backed by Metal kernels.

If profiling shows standard MPS Graph execution still fragments a stable recurrent sequence into too many tiny launches, prototype one fused operation covering a narrow hot region such as:

```text
packed Q
→ attention score transforms
→ value read
→ write projection
→ gate/residual pieces
```

Promotion requires:

- numerical equivalence,
- backward correctness,
- >15% repeated-step speed improvement,
- manageable implementation complexity.

Do not prematurely rewrite the whole model in Metal.

## MPS-5 — Synchronization discipline

Use `torch.mps.synchronize()` around timing boundaries.

Avoid synchronization inside the hot loop unless correctness requires it.

---

# 15. Train-Time Pipeline Redesign

Interactive learning introduces rollout generation, which may become more expensive than gradient updates.

Separate:

\[
\text{rollout generation}
\]

from:

\[
\text{gradient training}
\]

## Phase A — synchronous reference

Start with one correct, reproducible process.

## Phase B — batched rollout/training

Once correct:

- collect many vectorized trajectories,
- stack into training batches,
- perform several optimizer updates per rollout window.

Measure accelerator idle time.

## Phase C — asynchronous only later

Only after the reference system works should rollout workers and trainer be decoupled.

Do not introduce distributed/off-policy complexity before the single-node signal is established.

---

# 16. Inference-Time Speed Target

Interactive inference is measured in **time per action**, not tokens/sec.

\[
T_{\text{action}}=T_{\text{observe}}+T_S+T_{H,R}+T_{\text{policy}}
\]

Measure each component.

Report:

- batch-1 latency,
- p50/p95,
- latency vs \(R\),
- realized \(R\) under early exit,
- successful actions/sec,
- memory footprint.

Target:

\[
\boxed{\text{maximum verified task success per unit latency}}
\]

not minimum latency at any cost.

---

# 17. Speed Experiment Order

Do not combine all optimizations at once.

1. Vectorize world state and transitions.
2. Eliminate per-step host/device transfers.
3. Establish CPU vs MPS vs CUDA reference numbers.
4. Batch many environments.
5. Keep D/2 value/write.
6. SPEED-A batched dual-source attention.
7. K=2 evidence refresh.
8. CUDA: compile + CUDA Graph replay.
9. MPS: profile; custom Metal fusion only if justified.
10. Adaptive early exit only after useful depth exists.

Each step gets an ablation table.

No cumulative speed claim without individual measurements.

---

# 18. Cross-Platform Benchmark Contract

Every performance result must report:

## Hardware

- device,
- exact GPU/Apple chip,
- PyTorch version,
- dtype.

## Workload

- environment batch,
- observation shape,
- \(M_H\),
- \(D\),
- value_dim,
- \(R\),
- horizon,
- library size if applicable.

## Training

- environment transition time,
- forward,
- backward,
- optimizer,
- total step,
- env steps/sec,
- trajectories/sec.

## Inference

- batch-1 action latency,
- batch-N latency,
- p50/p95,
- success/task,
- actions/task.

## Memory

- peak device memory,
- params,
- rollout-buffer memory.

## Correctness

Every semantics-preserving optimized path must pass numerical/behavioral equivalence before timing is trusted.

---

# 19. First Critical Experiments

## EXP-HW-0 — Is the benchmark learnable?

Train a simple reference agent and oracle-backed baseline.

If nobody learns, fix the environment before blaming HZ.

## EXP-HW-1 — Can HZ learn basic world operation?

Current best HZ only.

Promotion:

- well above random,
- improving with training,
- held-out procedural-world generalization.

Initial failure triggers rescue ladder, not branch death.

## EXP-HW-2 — Does persistent \(S\) matter?

Compare:

- normal \(S\),
- reset \(S\),
- zeroed \(S\),
- optionally shuffled memory.

Success:

\[
\boxed{\text{real performance loss when persistent memory is destroyed}}
\]

on tasks designed to require past experience.

## EXP-HW-3 — Does horizon create useful \(R\)?

Sweep:

\[
R=1,2,4,8,16
\]

for each horizon bucket.

Primary target:

\[
\boxed{R^*_{\text{long}}>R^*_{\text{short}}}
\]

with reproducible task-success gains beyond noise.

## EXP-HW-4 — Interaction vs static supervision

Create matched information in two forms.

### Static

All relevant transitions/facts supplied as input.

### Interactive

Agent must act, observe consequences, and update \(S\).

Compare:

- task success,
- sample efficiency,
- adaptation to changed rules,
- compute.

This directly tests Hatchling World's central hypothesis.

## EXP-HW-5 — RLVR after imitation

Only after EXP-HW-1 works.

Test whether verified-reward exploration improves:

- held-out success,
- recovery after mistakes,
- long-horizon planning.

## EXP-HW-6 — Systems pass

Repeat the same workload on:

- CPU,
- MPS,
- CUDA,

then apply the speed ladder in order.

---

# 20. Early Success Does Not Require Immediately Beating Every Transformer

Hatchling World survives early phases if it produces any reproducible, distinctive useful signal such as:

1. HZ uses persistent \(S\) substantially better than controls.
2. HZ generalizes better to changed hidden rules.
3. Harder tasks benefit from larger \(R\).
4. HZ learns from failed actions within an episode.
5. HZ needs fewer interactions to adapt to a new world.
6. HZ reaches a better quality/memory Pareto point.
7. HZ matches success with lower inference compute.

Long-term promotion still requires:

\[
\boxed{\text{a meaningful quality-compute advantage over matched baselines}}
\]

not merely interesting behavior.

---

# 21. What Finally Counts as Failure?

Do **not** kill Hatchling World after one disappointing number.

Park/kill only after:

- environment validity is proven,
- baseline learnability is proven,
- reward/curriculum bugs are ruled out,
- one bounded optimization rescue is completed,
- at least three task families are tested,
- persistent-memory and depth-specific tasks are included,
- and HZ still shows no useful advantage or distinctive useful behavior.

---

# 22. Immediate Implementation Checklist

## Phase 1 — environment

- [x] Create `hatchling_world/`. (commit a20bc30, 2026-09-04)
- [x] Fixed-shape world schema. (`state.py`: batched WorldState/WorldConfig)
- [x] Batched vectorized transition engine. (`transition.py`, `vector_env.py`)
- [x] Oracle planner. (`oracle.py`, real BFS over the exact transition semantics)
- [x] Reward verifier. (`rewards.py`)
- [x] Train/test procedural seeds. (`curriculum.py`, disjoint seed-space split)
- [x] Difficulty/horizon generator. (`curriculum.py`, School levels S0/S1/S2/S3/S5 --
      S4's "learn from a failed experiment" mechanic honestly not built yet,
      this W0 sandbox has no experimentable/failable action to hang it on)
- [x] Unit tests for transitions and solvability. (16 tests, 4 files, incl. a
      real 400/400 solvability stress sweep and oracle-plan-replay-through-
      the-real-env checks)
- [x] Real-time live viewer, added on top of the plan's own checklist per
      explicit request: `scripts/hz_world_live_view.py` (local HTTP server,
      stdlib only, SVG room-graph render) + `scripts/hz_world_rollout_demo.py`
      (feeds it -- oracle-driven today, verified live end to end). Phase 2's
      real HZ policy will plug into the identical snapshot schema.

## Phase 2 — HZ adapter

- [x] Original LN recurrence only. (`reference/hz_world_agent_torch.py`'s
      `HZWorldAgent` uses `HZCQReasoningWorkspaceConfig` with
      identity_biased/bounded_residual/bounded_accumulating all False --
      verified directly by test)
- [x] \(M_H=32\). (default `workspace_slots`)
- [x] D/2 value/write. (`value_dim = d_model // 2`, verified by test)
- [x] Exact Q/K. (unchanged from the validated `HZCQReasoningWorkspace`)
- [x] Persistent \(S\) update after action consequences.
      (`update_memory()`, real section-3.3 \(S_{t+1}=U_\theta(S_t,o_t,a_t,r_t,o_{t+1})\))
- [x] Fixed action head. (`rq/rk/rv` cross-attention readout + `action_head`
      Linear, same pattern as the FSM harness's readout)
- [x] No new recurrence experiments. (zero architecture changes to
      `HZCQPersistentMemory`/`HZCQReasoningWorkspace`, only new glue code)

## Phase 3 — behavior cloning

- [x] Oracle trajectories. (`hatchling_world.oracle.solve`, real BFS plans)
- [ ] World-prediction auxiliary target. (not implemented -- BC alone
      already produced real learning, see result below; this is a real,
      disclosed gap, not yet needed)
- [x] Policy imitation. (`scripts/hz_world_behavior_clone.py`, real
      teacher-forced BC with full-episode BPTT through S and every
      step's H rounds)
- [x] Held-out-world baseline. (`split="test"` live-eval episodes,
      self-driven, never trained on -- real result below)

**Real result, 2026-09-04: genuine learning confirmed, reproduced across
two seeds.** S0_cause_effect, 3000 BC episodes, `d_model=64`, `M_H=32`
(D/2 value/write), `n_rounds=8`, real teacher-forced BPTT through the
whole episode. Per-step action accuracy (train split): ~42% at episode
50 -> ~56% at episode 200 -> ~88-93% by episode 3000. Real, self-driven
held-out (`split="test"`) evaluation episodes -- the agent's OWN
argmax actions, no oracle forcing -- reach a 90% success rate over the
last 10 live evals by the end of training (seed 1: 9/10 successes,
climbing from the first eval already at 100% on this easy level to a
stable ~90% band). This is the first real evidence that Hatchling
World's actual pipeline (environment -> HZWorldAgent -> live viewer)
produces genuine, watchable learning, not just a working demo of the
oracle.

Real, disclosed limitation: this result is on S0 (the easiest level,
horizon 1-2 actions) only, with `--eval-step-delay 0` for the speed of
this initial check -- the real EXP-HW-1/EXP-HW-3 questions (does this
generalize to S1-S5, does horizon create useful \(R\)) are still open,
not yet run at scale, and are the natural next real experiments.

## Phase 4 — memory

- [ ] Frozen-weight lifetime evaluation.
- [ ] \(S\) reset/zero ablations.
- [ ] Delayed-use tasks.

## Phase 5 — depth

- [ ] Horizon buckets.
- [ ] \(R\in\{1,2,4,8,16\}\).
- [ ] Success vs \(R\).
- [ ] Action efficiency vs \(R\).

## Phase 6 — systems

- [ ] CPU baseline.
- [ ] MPS reference.
- [ ] CUDA reference.
- [ ] Remove per-step transfers.
- [ ] Device/vectorized worlds.
- [ ] SPEED-A.
- [ ] K=2 evidence refresh.
- [ ] CUDA Graph benchmark.
- [ ] MPS profiler pass.

## Phase 7 — RL

- [ ] Verifiable reward loop.
- [ ] On-policy baseline.
- [ ] Group-relative trajectory optimization.
- [ ] Replay/off-policy later.

## Phase 8 — Library

- [ ] `READ(query)` action.
- [ ] Retrieval cost.
- [ ] Bounded fact response.
- [ ] Long-delay memory evaluation.

---

# 23. Suggested Repository Structure

```text
hatchling_world/
    __init__.py
    state.py
    actions.py
    transition.py
    generator.py
    oracle.py
    rewards.py
    vector_env.py
    curriculum.py
    library.py

reference/
    hz_world_agent_torch.py

scripts/
    hz_world_validate.py
    hz_world_behavior_clone.py
    hz_world_depth_sweep.py
    hz_world_memory_ablation.py
    hz_world_rlvr.py
    hz_world_grpo.py
    hz_world_speed_benchmark.py

tests/
    test_hz_world_transition.py
    test_hz_world_oracle.py
    test_hz_world_vector_env.py
    test_hz_world_memory.py
    test_hz_world_agent.py

results/
    hatchling_world/
```

---

# 24. Commit Discipline

Suggested sequence:

1. `Add deterministic vectorized Hatchling World environment`
2. `Add oracle solver and verifiable rewards`
3. `Connect HZ persistent memory and policy to Hatchling World`
4. `Establish behavior-cloning and held-out-world baseline`
5. `Measure persistent-memory ablations in Hatchling World`
6. `Measure horizon-by-recurrent-depth scaling`
7. `Vectorize Hatchling World for MPS and CUDA`
8. `Reduce Hatchling World dispatch and evidence-refresh cost`
9. `Add verifiable-reward post-training`
10. `Add paid Library retrieval curriculum`

Do not bundle environment design, RL, recurrence redesign, and systems optimization into one commit.

---

# 25. External Systems Notes

Current platform documentation supports the following systems directions:

- PyTorch `torch.compile` exposes `default`, `reduce-overhead`, and `max-autotune` modes; `reduce-overhead` is specifically intended to reduce Python overhead with CUDA Graphs where applicable.
- CUDA Graphs are designed to reduce repeated kernel-launch overhead by replaying a captured static execution graph.
- PyTorch's MPS backend maps PyTorch operations onto Metal Performance Shaders / MPS Graph and tuned Metal kernels.
- Apple supports custom PyTorch operations backed by Metal kernels, which makes a later fused recurrent primitive possible if profiling justifies it.

References:

- https://docs.pytorch.org/docs/stable/generated/torch.compile
- https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html
- https://developer.apple.com/metal/pytorch/
- https://docs.pytorch.org/docs/stable/notes/mps.html
- https://developer.apple.com/documentation/Metal/customizing-a-pytorch-operation

---

# 26. Final Thesis

Hatchling World tests a different explanation for HZ's repeated flat-\(R\) results:

> Maybe the state machinery is not fundamentally incapable of useful recurrent reasoning. Maybe the static tasks used so far reward one-shot function approximation strongly enough that extra latent computation has little reason to become useful.

Hatchling World creates a setting where:

- information arrives over time,
- actions have consequences,
- mistakes reveal information,
- persistent memory can matter,
- retrieval has a cost,
- future observations depend on previous decisions,
- and long-horizon success is objectively verifiable.

The branch answers two independent questions:

\[
\boxed{\textbf{Q1: Does interaction make HZ's stateful architecture more useful?}}
\]

and

\[
\boxed{\textbf{Q2: Can that interaction loop be made accelerator-efficient enough to matter?}}
\]

Q1 is measured by:

- task success,
- within-lifetime learning,
- memory ablations,
- horizon-vs-\(R\) scaling,
- verified-reward improvement.

Q2 is measured by:

- vectorized worlds,
- device-resident rollout state,
- reduced dispatch,
- fewer evidence refreshes,
- CUDA Graph replay,
- MPS-specific profiling/fusion,
- real wall-clock benchmarks.

Do not claim success until both axes are measured.

But equally:

\[
\boxed{\textbf{one weak initial experiment is a diagnostic, not the end of Hatchling World.}}
\]

The branch earns multiple controlled, falsifiable attempts before it is killed.
