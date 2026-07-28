# HZ-0D Total Restart Plan

## Starting Point

This plan assumes no working HZ-0D code exists.

Only the README, master plan, and Git history are available.

Everything related to adaptive recurrence must be rebuilt:

- controller equations
- compute policy
- halting logic
- budget enforcement
- training losses
- tests
- PMetal implementation
- evaluation

---

## Objective

HZ-0D adds adaptive recurrence and dynamic compute to a completed HZ-0C model.

HZ-0D should answer:

> Can the model allocate more recurrent computation to difficult tokens or reasoning steps and less computation to easy ones while improving quality per unit of compute?

HZ-0D must not be merely:

- a deeper fixed model
- speculative decoding
- token skipping without validation
- uncontrolled recurrence
- an inference-only heuristic

The compute policy must be trainable, bounded, measurable, and reproducible.

---

# Phase D0 — Git-history audit

Inspect Git history for:

- prior adaptive-depth ideas
- recurrence controllers
- halting criteria
- confidence scores
- compute penalties
- failed experiments
- stability concerns
- intended metrics

Classify findings as confirmed, uncertain, or rejected.

## Deliverables

- `docs/restart/hz0d_history_audit.md`
- `docs/restart/hz0d_recovered_requirements.md`

## Exit gate

Adaptive-compute behavior is defined independently of old code.

---

# Phase D1 — Define the compute contract

Specify:

- minimum recurrent steps
- maximum recurrent steps
- per-token versus per-sequence decisions
- whether state is shared across steps
- controller inputs
- halting threshold
- deterministic inference policy
- training-time stochasticity
- global compute budget
- fallback behavior
- reset behavior

A first policy may use:

```text
continue_probability_t = controller(hidden_t, state_t, uncertainty_t)
```

and stop when cumulative halting mass or a threshold is reached.

Set hard maximums. No input may trigger unbounded compute.

## Exit gate

Every token has a deterministic, bounded compute path at inference.

---

# Phase D2 — Build a tiny adaptive-compute simulator

Use synthetic tasks where difficulty is controllable.

Tasks:

1. easy versus hard arithmetic depth
2. nested parentheses
3. variable-chain length
4. sequence-copy distance
5. multi-hop retrieval depth
6. code-trace depth
7. ambiguous versus obvious classification

Verify that:

- easy cases use fewer steps
- hard cases use more steps
- compute does not collapse to minimum
- compute does not saturate at maximum
- output remains stable

## Exit gate

Adaptive computation correlates with known task difficulty.

---

# Phase D3 — Train the controller

Possible losses:

```text
task loss
expected compute penalty
budget violation penalty
halting entropy regularizer
minimum-utilization penalty
maximum-saturation penalty
consistency loss
```

Train in stages:

1. supervised target step counts
2. soft compute penalty
3. latent compute allocation
4. fixed average-budget training

Compare controller forms:

- scalar halt gate
- discrete step classifier
- monotonic cumulative halting
- value-of-computation predictor

## Exit gate

The controller allocates compute nontrivially and remains within budget.

---

# Phase D4 — Establish fair baselines

Compare against:

- fixed minimum steps
- fixed average steps
- fixed maximum steps
- shallower model
- deeper model
- confidence-based heuristic
- early-exit transformer
- equal-FLOP static model

Evaluate at matched average compute, not only matched parameter count.

## Exit gate

HZ-0D can be compared fairly on quality per FLOP and latency.

---

# Phase D5 — Wait for stable HZ-0C

Full integration waits until HZ-0C has:

- stable session-state semantics
- reliable reset
- PMetal implementation
- trained checkpoint
- adaptation benchmarks

The adaptive-compute simulator may proceed independently.

## Exit gate

A frozen HZ-0C checkpoint is available.

---

# Phase D6 — Integrate with frozen HZ-0C

Begin by applying adaptive recurrence to a small subset of layers.

Keep HZ-0C frozen initially.

Train only:

- halt controller
- uncertainty estimator
- step embedding
- compute-budget controller

Ensure that HZ-0B memory and HZ-0C fast-state updates occur under well-defined step semantics.

Specify whether:

- memory writes happen once per token
- fast weights update once per token
- internal recurrent refinement steps are read-only

The safest first design is:

> extra internal steps may refine hidden state, but external memory and fast weights update only once after halting.

## Exit gate

Dynamic steps do not duplicate or corrupt memory and fast-state updates.

---

# Phase D7 — Natural curriculum

Train on mixtures with variable difficulty:

- short and long reasoning chains
- easy and hard code completion
- shallow and deep tool planning
- variable retrieval hops
- short and long dependency spans
- structured output with varying constraints

Use explicit compute budgets during training.

## Exit gate

The model uses extra compute on difficult examples without universal saturation.

---

# Phase D8 — PMetal implementation

Potential PMetal components:

```text
adaptive_recurrence_loop
halt_controller
active_token_compaction
step_state_update
budget_enforcement
```

Optimization goals:

- avoid running halted tokens
- compact active tokens efficiently
- minimize control-flow synchronization
- preserve deterministic ordering
- support batched sequences with different step counts

Keep a slow reference loop.

## Exit gate

PMetal matches reference outputs and step decisions.

---

# Phase D9 — Evaluation

Measure:

- quality at matched average FLOPs
- average steps per token
- step distribution
- easy versus hard allocation
- p50/p95/p99 latency
- throughput
- memory
- controller overhead
- budget violations
- quality under forced budgets
- robustness to adversarial inputs

## Exit gate

HZ-0D improves quality per compute or reduces compute at matched quality.

---

# HZ-0D completion definition

HZ-0D is complete when:

1. Compute policy is explicit and bounded.
2. Difficulty-sensitive allocation works in isolation.
3. The controller does not collapse or saturate.
4. Integration preserves memory and fast-weight semantics.
5. It beats fixed-step baselines at matched compute.
6. PMetal execution matches the reference.
7. Tail latency and worst-case compute are documented.
8. Inference behavior is deterministic under a fixed policy.

---

# First concrete milestone

> A tiny adaptive-recurrence model that uses fewer steps on easy synthetic examples, more steps on hard examples, and beats fixed-step baselines at the same average compute.
