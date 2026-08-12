# HZ-0D Total Restart Plan

## Objective

HZ-0D adds **bounded session-local fast weights** with snapshot, rollback, reset, and strict session isolation to a completed HZ-0C model.

It must answer:

> Can the model temporarily adapt its computation during a session without changing permanent pretrained weights, while remaining bounded and reversible?

HZ-0D is distinct from HZ-0B memory, HZ-0C surprise-triggered attention, LoRA fine-tuning, and persistent personalization.

## Starting point

Assume no working HZ-0D implementation survives. Git history is reference material only.

## D0 — Audit Git history

Recover and classify fast-weight equations, update controllers, reset and isolation rules, snapshot/rollback ideas, prior toy tasks and failures, and integration attempts.

Create:

- `docs/restart/hz0d_history_audit.md`
- `docs/restart/hz0d_recovered_requirements.md`

**Exit gate:** the mechanism is specified independently of archived code.

## D1 — Define the contract

Specify which layers receive fast weights, low-rank dimensions, update frequency and budget, decay, clipping, normalization, snapshot, rollback, reset, serialization, gradient flow, and maximum fast-state memory.

Start with low-rank adapters:

```text
W_effective = W_base + A_fast @ B_fast
```

Permanent model weights never change during ordinary use.

**Exit gate:** every state tensor and lifecycle operation is documented.

## D2 — Isolated simulator

Test rapid mapping adaptation, symbol remapping, temporary formatting rules, temporary API schemas, few-example classification, contradictory rule changes, decay, snapshot/rollback, reset, noisy updates, and malicious updates.

Measure adaptation speed, interference, state norms, rollback fidelity, and reset fidelity.

**Exit gate:** temporary mappings work and prior state is restored exactly.

## D3 — Choose the update mechanism

Compare Hebbian updates, learned gradient-like updates, low-rank delta prediction, and error-conditioned adapter updates.

The controller predicts whether, where, and how strongly to update.

Add update sparsity, norm clipping, stability penalties, rollback consistency, interference penalties, and a per-session update budget.

**Exit gate:** one bounded method clearly beats simple alternatives.

## D4 — Fair baselines

Compare with no adaptation, ordinary in-context learning, longer context, HZ-0B memory only, HZ-0C only, retrieval, static adapters, gradient-updated adapters, and permanent LoRA.

**Exit gate:** gains are attributable to temporary fast adaptation.

## D5 — Dependency gate

Full integration waits for a frozen HZ-0C with stable HZ-0B memory, surprise controller, triggered attention, PMetal implementation, trained checkpoint, and baselines.

The isolated simulator may proceed earlier.

## D6 — Frozen-backbone integration

Start with fast adapters only in narrow locations such as upper MLP blocks, memory controllers, or anchor-attention output projections.

Avoid modifying the core GDN-2 update first.

**Exit gate:** inactive fast weights reproduce HZ-0C behavior; active fast weights improve adaptation.

## D7 — Define state ordering

Per token:

1. read HZ-0B memory
2. run the recurrent backbone
3. compute HZ-0C surprise
4. optionally run anchor attention
5. produce output
6. perform at most one memory write
7. perform at most one fast-weight update

Prevent duplicate writes and feedback loops.

**Exit gate:** state transitions are deterministic and unambiguous.

## D8 — Curriculum

Progress through explicit update supervision, few-shot rule inference, rule switching, natural temporary preferences and schemas, and adversarial update/rollback tasks.

**Exit gate:** adaptation is sparse, quick, and reversible.

## D9 — PMetal implementation

Implement fast-weight apply, update, decay, snapshot, rollback, reset, and low-rank delta matmul.

Support batched sessions, deterministic ordering, BF16 stability, and checkpointing.

**Exit gate:** PMetal state transitions and gradients match the reference.

## D10 — Evaluation

Measure few-shot adaptation, examples and time required, retention, rule switching, interference, rollback/reset fidelity, latency, memory overhead, general-quality degradation, and malicious-update resistance.

**Exit gate:** HZ-0D beats prompting, memory-only, and static-adapter baselines while respecting bounds.

## Completion definition

HZ-0D is complete when:

1. Fast-weight and lifecycle semantics are explicit.
2. Isolation, update, rollback, and reset tests pass.
3. It beats fair adaptation baselines.
4. HZ-0B and HZ-0C behavior is preserved.
5. PMetal matches the reference.
6. Update budgets and overhead are documented.
7. Permanent pretrained weights remain unchanged during use.

## First milestone

> A standalone low-rank fast-weight simulator that learns a temporary mapping, snapshots it, survives interference, rolls back exactly, and resets to baseline.
