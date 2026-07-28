# HZ-0C Total Restart Plan

## Starting Point

This plan assumes the active branch contains only:

- `README`
- the HATCHLING-ZERO master plan
- `.git` history

It assumes no working HZ-0C implementation remains.

The following must be rebuilt or explicitly recovered:

- fast-weight equations
- update rules
- reset behavior
- training curriculum
- tests
- integration code
- PMetal implementation
- evaluation harness

Git history may be inspected for ideas and historical evidence, but old code must not be trusted until reproduced and tested.

---

## Objective

HZ-0C adds session-local fast weights to a completed HZ-0B model.

HZ-0C should answer:

> Can the model rapidly adapt its internal computation within a session without changing its permanent pretrained weights and without corrupting base-model behavior?

HZ-0C must remain distinct from:

- ordinary hidden state
- HZ-0B associative memory
- optimizer-based fine-tuning
- LoRA adapters
- permanent personalization
- external retrieval

The intended behavior is temporary, session-local adaptation that resets cleanly.

---

# Phase C0 — Git-history audit

Inspect Git history for:

- prior fast-weight equations
- update rules
- learned update controllers
- reset and persistence semantics
- any successful toy tasks
- instability modes
- prior claims about adaptation
- integration attempts
- parameter-count assumptions

Classify findings as:

### Confirmed

Supported by tests or repeated evidence.

### Uncertain

Mentioned but not validated.

### Rejected

Broken, ambiguous, or misleading approaches.

## Deliverables

- `docs/restart/hz0c_history_audit.md`
- `docs/restart/hz0c_recovered_requirements.md`

## Exit gate

The new fast-weight mechanism is specified independently of archived code.

---

# Phase C1 — Define the fast-weight contract

Specify exactly what changes during a session.

Potential fast-weight state:

```text
fast matrices
fast biases
update confidence
decay rate
age
reset token or boundary metadata
```

Define:

- which layers receive fast weights
- whether updates apply to attention, GDN-2, MLP, or adapters
- rank and dimensionality
- update frequency
- update rule
- decay
- clipping
- normalization
- session reset
- serialization policy
- whether gradients flow through update steps
- maximum fast-state memory

Start with low-rank fast adapters rather than full dense matrices.

A practical first form:

```text
W_effective = W_base + A_fast @ B_fast
```

where `A_fast` and `B_fast` are session-local and resettable.

## Exit gate

All state tensors, update equations, boundaries, and reset semantics are documented.

---

# Phase C2 — Build an isolated simulator

Before integrating with the language model, test fast adaptation on synthetic systems.

Tasks:

1. rapid linear mapping adaptation
2. symbol remapping
3. temporary vocabulary substitution
4. few-example classification
5. temporary formatting rule
6. temporary API schema adaptation
7. contradictory session rules
8. decay after inactivity
9. reset to baseline
10. adaptation under noisy examples

Measure:

- adaptation speed
- number of examples required
- retained performance
- interference
- decay behavior
- reset fidelity
- norm growth
- stability over long sessions

## Exit gate

The fast-weight system learns temporary mappings and resets exactly.

---

# Phase C3 — Choose the update mechanism

Compare at least three mechanisms:

1. Hebbian or outer-product update
2. learned gradient-like update controller
3. low-rank delta prediction

A candidate learned update:

```text
delta_t = controller(hidden_t, error_signal_t, context_t)
fast_state_t = decay * fast_state_(t-1) + gated(delta_t)
```

The update controller should predict:

- whether to update
- where to update
- update magnitude
- decay or retention
- optional confidence

Add constraints:

- norm clipping
- update sparsity
- stability penalty
- reset-consistency penalty
- interference penalty

## Exit gate

One mechanism clearly outperforms simple baselines on isolated adaptation tasks.

---

# Phase C4 — Establish fair baselines

Compare against:

- no adaptation
- longer prompt context
- HZ-0B memory only
- retrieval of examples
- ordinary in-context learning
- LoRA fine-tuning
- gradient descent on a small adapter
- equal-parameter static adapter

This determines whether fast weights add value beyond keeping examples in context.

## Exit gate

The evaluation harness can distinguish fast adaptation from ordinary prompting and retrieval.

---

# Phase C5 — Wait for stable HZ-0B

Full HZ-0C integration should wait until HZ-0B has:

- a frozen memory contract
- reliable reset behavior
- stable PMetal integration
- a trained checkpoint
- memory-specific evaluation baselines

The isolated HZ-0C simulator may proceed in parallel.

## Exit gate

A reproducible HZ-0B checkpoint is available.

---

# Phase C6 — Integrate with frozen HZ-0B

Start with HZ-0B frozen.

Add fast weights only to a narrow location, preferably:

- upper-layer MLP adapters, or
- memory read/write controllers

Train only:

- fast-weight generator
- update gate
- decay controller
- reset controller

Compare:

```text
HZ-0B
HZ-0B plus static adapter
HZ-0B plus fast adapter
```

## Exit gate

Fast weights improve session adaptation without degrading normal behavior when inactive.

---

# Phase C7 — Training curriculum

## Stage 1 — Explicit adaptation targets

The data directly indicates the temporary rule.

## Stage 2 — Few-shot adaptation

The model infers a rule from demonstrations.

## Stage 3 — Rule switches

The session changes rules and the model must update.

## Stage 4 — Natural tasks

Use:

- temporary user formatting preferences
- temporary tool schemas
- codebase-specific conventions
- new variable bindings
- temporary terminology
- task-specific output structures

## Stage 5 — Adversarial stability

Test:

- contradictory examples
- malicious adaptation attempts
- long sessions
- repeated rule switching
- reset boundaries
- irrelevant demonstrations

## Exit gate

The model adapts sparsely, quickly, and reversibly.

---

# Phase C8 — PMetal implementation

Port proven operations to PMetal.

Potential operators:

```text
fast_weight_apply
fast_weight_update
fast_weight_decay
fast_weight_reset
low_rank_delta_matmul
```

Handle:

- batched session states
- gradient flow through update sequences
- checkpointing
- BF16 stability
- update ordering
- long-session memory use

Keep a pure reference implementation.

## Exit gate

PMetal outputs, state transitions, and gradients match the reference implementation.

---

# Phase C9 — Evaluation

Evaluate:

- few-shot adaptation accuracy
- examples required to adapt
- time to adapt
- temporary rule retention
- rule switching
- interference with prior rules
- reset fidelity
- session length scaling
- latency overhead
- memory overhead
- general-language degradation

Compare with all fair baselines.

## Exit gate

HZ-0C demonstrates adaptation that is faster or cheaper than fine-tuning and better than prompting alone.

---

# HZ-0C completion definition

HZ-0C is complete when:

1. Fast-weight state and update semantics are explicit.
2. Isolated adaptation and reset tests pass.
3. It outperforms prompting, retrieval, and static adapters on matched tasks.
4. Integration with frozen HZ-0B preserves base quality.
5. Session rule changes and resets are reliable.
6. PMetal kernels match the reference.
7. Overhead and limitations are documented honestly.
8. No permanent weight changes occur during ordinary use.

---

# First concrete milestone

> A standalone low-rank fast-weight simulator that learns a temporary mapping from a few examples, survives controlled interference, and returns exactly to baseline after reset.
