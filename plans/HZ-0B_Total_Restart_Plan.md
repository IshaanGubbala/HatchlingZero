# HZ-0B Total Restart Plan

## Starting Point

This plan assumes the active branch contains only:

- `README`
- the HATCHLING-ZERO master plan
- `.git` history

It assumes no working HZ-0B code remains.

The following must be rebuilt or explicitly recovered:

- memory equations
- memory-state definitions
- read and write logic
- protection behavior
- overwrite behavior
- simulator
- training curriculum
- tests
- integration code
- PMetal kernels
- evaluation harness

Git history may be inspected for ideas and historical evidence, but old behavior must be reproduced before it is trusted.

---

## Objective

HZ-0B adds an explicit, controllable, session-local associative memory to a finished and frozen HZ-0A base.

HZ-0B should answer:

> Can the model store, retrieve, protect, update, and forget useful information beyond its normal recurrent state without damaging base-model quality?

HZ-0B must provide a clearly separate memory state with observable operations.

It must not be confused with:

- the ordinary GDN-2 recurrent state
- a larger hidden dimension
- longer context alone
- external vector search
- permanent user-profile storage
- HZ-0C fast weights

---

# Phase B0 — Git-history audit

Inspect Git history for:

- previous memory-state shapes
- read equations
- write equations
- protection logic
- overwrite experiments
- synthetic tasks
- test outcomes
- known failures
- integration attempts
- naming conventions

Create a recovery report with:

### Confirmed

Ideas that were supported by tests or repeated evidence.

### Uncertain

Ideas mentioned but not adequately validated.

### Rejected

Mechanisms that failed, were ambiguous, or produced misleading results.

Historical lessons currently known include:

- basic recall appeared promising
- protection behavior appeared promising
- overwrite behavior remained unresolved

These remain hypotheses until independently reproduced.

## Deliverables

- `docs/restart/hz0b_history_audit.md`
- `docs/restart/hz0b_recovered_requirements.md`

## Exit gate

The new memory design is based on an explicit specification rather than assumptions about archived code.

---

# Phase B1 — Define the memory contract

Specify the memory independently from any language model.

The memory state should include explicit tensors such as:

```text
keys
values
usage or confidence
age
protection strength
write metadata
```

Required operations:

```text
read(query)
write(key, value, strength)
reinforce(existing memory)
update(existing memory)
protect(memory)
forget or decay(memory)
delete(memory)
reset()
serialize()
restore()
```

Decide explicitly:

- fixed-slot or matrix memory
- number of slots
- key dimension
- value dimension
- soft versus hard addressing
- top-k behavior
- write ordering
- collision behavior
- capacity overflow
- eviction policy
- session-local lifecycle
- per-layer or shared memory
- whether writes are differentiable
- how memory output enters the residual stream

For the first implementation:

- use fixed-capacity, session-local memory
- do not implement persistent cross-user storage
- keep memory reset explicit
- keep the memory state serializable

## Exit gate

Every memory tensor, operation, shape, and lifecycle rule is documented.

---

# Phase B2 — Build a pure reference simulator

Before integrating with HZ-0A, build an isolated memory simulator in simple Python or MLX.

The simulator must expose exact operations and state transitions.

Initial tasks:

1. store one key-value pair
2. retrieve an exact key
3. retrieve from a noisy key
4. store multiple independent keys
5. handle similar keys
6. overwrite an existing fact
7. reinforce an existing fact
8. protect one memory
9. overwrite a different memory
10. forget by age
11. overflow capacity
12. reset and restore
13. chained retrieval
14. conflicting writes

Track:

- recall accuracy
- false retrieval rate
- overwrite success
- reinforcement success
- protection survival
- deletion success
- collision rate
- capacity degradation
- memory norm stability
- deterministic behavior

## Exit gate

The simulator passes predefined tests without any language model attached.

---

# Phase B3 — Solve memory semantics

Separate these events explicitly:

```text
new write
reinforcement
contradictory update
temporary scratch value
protected fact
intentional deletion
stale memory
```

Do not rely on one similarity threshold for all behaviors.

Define control signals such as:

- write probability
- update probability
- reinforce probability
- erase strength
- protection strength
- slot-selection distribution
- memory-validity confidence

Add penalties or constraints for:

- excessive writes
- uncontrolled memory growth
- protected-memory corruption
- repeated duplicate memories
- unstable key or value norms

Potential losses:

```text
retrieval loss
write-decision loss
update-classification loss
protection violation penalty
write sparsity penalty
memory norm penalty
language-model loss after integration
```

## Exit gate

New writes, reinforcement, updates, deletion, and protection produce observably different behavior.

---

# Phase B4 — Establish fair baselines before integration

Implement non-HZ-0B baselines:

- no memory
- larger recurrent state
- longer context
- simple key-value cache
- external vector retrieval
- equal-parameter feed-forward adapter

Use the same synthetic tasks.

This prevents the project from crediting HZ-0B for improvements that come merely from more capacity.

## Exit gate

The evaluation harness can compare HZ-0B against simpler alternatives.

---

# Phase B5 — Wait for a stable HZ-0A checkpoint

Full integration must not begin until HZ-0A has:

- a frozen architecture
- a reproducible tokenizer
- a stable PMetal implementation
- verified gradients
- reliable checkpoint loading
- a trained checkpoint
- a known no-memory evaluation baseline

The isolated HZ-0B simulator can proceed in parallel.

## Exit gate

A frozen HZ-0A checkpoint and evaluation suite are available.

---

# Phase B6 — Read-only integration with frozen HZ-0A

Integrate memory in the safest order.

First add only:

1. hidden-state-to-query projection
2. memory read
3. gated memory contribution into the residual stream

Keep HZ-0A frozen.

Do not allow writes yet.

Compare:

```text
HZ-0A frozen, no memory
HZ-0A frozen, read-only memory
```

Verify:

- memory read does not destabilize logits
- empty memory behaves like no memory
- reset returns to baseline behavior
- unrelated memories do not corrupt output

## Exit gate

Read-only memory improves memory-specific tasks without materially degrading general held-out loss.

---

# Phase B7 — Controlled writes

Add explicit write controls while keeping HZ-0A frozen.

Start with supervised writes where the training data specifies:

- whether to write
- the key
- the value
- whether to protect
- whether to update
- whether to delete

Train only:

- query projection
- read gate
- write controller
- update controller
- protection controller
- memory adapters

Compare:

```text
read only
read plus supervised write
read plus write plus update
```

## Exit gate

The model can store and retrieve supervised memories reliably.

---

# Phase B8 — Memory curriculum

## Stage 1 — Explicit operation supervision

Examples directly label memory operations.

## Stage 2 — Delayed recall

The model must store information and use it much later.

## Stage 3 — Latent write decisions

Only final behavior is supervised.

## Stage 4 — Natural sequences

Use:

- multi-turn conversations
- evolving constraints
- tool results needed later
- variable assignments
- code symbols
- document facts
- changing user preferences within a session

## Stage 5 — Adversarial memory

Test:

- contradictory later information
- distractors
- malicious overwrite attempts
- near-identical keys
- stale memories
- capacity pressure
- reset boundaries

## Exit gate

The model learns sparse, useful writes rather than writing every token.

---

# Phase B9 — Partial and full fine-tuning

After frozen-base integration works:

1. unfreeze only memory-adjacent projections
2. unfreeze selected upper HZ-0A layers
3. consider full fine-tuning only if necessary

At each stage, measure:

- general language loss
- memory-task performance
- write frequency
- memory interference
- catastrophic degradation

## Exit gate

Memory improvements survive limited fine-tuning without destroying HZ-0A quality.

---

# Phase B10 — PMetal implementation

Once reference semantics are stable, port the memory operations to PMetal.

Potential operators:

```text
associative_memory_read
associative_memory_write
memory_similarity
top_k_addressing
protected_update
memory_decay
memory_delete
```

Concurrency and backward design must handle:

- multiple tokens targeting the same slot
- deterministic write ordering
- atomic versus staged updates
- sequence parallelism
- BF16 stability
- checkpointed memory state
- gradients through read and write decisions

Keep the pure reference implementation permanently.

## Exit gate

PMetal memory operations match the reference implementation on state transitions, outputs, and gradients.

---

# Phase B11 — Evaluation

Evaluate on:

- exact associative recall
- noisy associative recall
- multi-hop retrieval
- passkey tasks
- long-conversation consistency
- state-variable tracking
- tool-result reuse
- code symbol and value tracking
- overwrite accuracy
- reinforcement accuracy
- protection retention
- forgetting accuracy
- reset behavior
- serialization and restoration
- capacity scaling
- adversarial interference

Compare against:

- HZ-0A alone
- longer-context HZ-0A
- expanded recurrent state
- external vector retrieval
- equal-parameter no-memory model

Measure cost:

- extra parameters
- bytes per slot
- read latency
- write latency
- training-memory overhead
- inference-memory overhead
- throughput degradation

## Exit gate

HZ-0B provides a measurable advantage that cannot be explained only by more parameters or more context.

---

# HZ-0B completion definition

HZ-0B is complete when:

1. The memory contract is explicit and versioned.
2. The isolated simulator passes recall, overwrite, protection, forgetting, deletion, and collision tests.
3. Read-only integration works with frozen HZ-0A.
4. Controlled writes work with frozen HZ-0A.
5. Latent write decisions work on natural sequences.
6. General HZ-0A quality is preserved.
7. HZ-0B beats fair no-memory, longer-context, and retrieval baselines.
8. PMetal kernels match the reference implementation.
9. Session-local reset, serialization, and restoration are reliable.
10. Memory costs and limitations are documented honestly.

---

# Recommended branch structure

```text
restart/pmetal-hz0b
├── docs/restart
├── docs/memory
├── src/reference/memory
├── src/simulator
├── src/curriculum
├── src/integration
├── src/pmetal/memory
├── tests/simulator
├── tests/integration
├── evals
└── configs
```

---

# First concrete milestone

The first HZ-0B milestone is not integration with the 300M model.

It is:

> A standalone, deterministic memory simulator with explicit read, write, update, protect, forget, delete, reset, and restore operations that passes synthetic tests for recall, overwrite, collision, and protection.
