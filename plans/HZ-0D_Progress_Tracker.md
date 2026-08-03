# HZ-0D Progress Tracker

Updated: July 27, 2026

## Mission

Add bounded session-local low-rank fast weights to a completed HZ-0C model, with strict isolation, snapshot, rollback, reset, and serialization semantics. Permanent pretrained weights must never change during ordinary use.

## Current Status

- Overall phase: `D0 not started`
- Dependency status: isolated fast-weight work may begin; full integration is blocked on a frozen HZ-0C and stable HZ-0B memory/trigger behavior
- Last verified HZ-0D evidence: none
- Current stopping rule: do not modify the recurrent core or integrate before the contract and simulator pass

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| D0 | history audit and recovered requirements | Not started | Create `docs/restart/hz0d_history_audit.md` and `hz0d_recovered_requirements.md` |
| D1 | fast-weight contract | Not started | Layers, ranks, budgets, clipping, decay, lifecycle, gradients, memory |
| D2 | isolated simulator | Not started | Temporary mappings, rule changes, noise, malicious updates, rollback/reset |
| D3 | update-mechanism selection | Not started | Compare Hebbian, learned gradient-like, delta prediction, error-conditioned updates |
| D4 | fair adaptation baselines | Not started | Prompting, context, memory, HZ-0C, retrieval, static and permanent adapters |
| D5 | HZ-0C dependency gate | Blocked | Requires frozen recurrence, memory, surprise attention, PMetal, checkpoint, baselines |
| D6 | frozen-backbone integration | Blocked | Start only in narrow upper MLP/controller/output locations |
| D7 | state ordering | Not started | One memory write and at most one fast update per token |
| D8 | curriculum | Not started | Supervised updates through natural schemas and adversarial rollback tasks |
| D9 | PMetal implementation | Blocked | Apply/update/decay/snapshot/rollback/reset and batched deterministic sessions |
| D10 | evaluation | Not started | Adaptation, retention, interference, fidelity, overhead, degradation, resistance |

## Required Artifacts

- [ ] `docs/restart/hz0d_history_audit.md`
- [ ] `docs/restart/hz0d_recovered_requirements.md`
- [ ] Fast-state tensor/lifecycle contract
- [ ] Standalone low-rank simulator and exact lifecycle tests
- [ ] Update-mechanism comparison report
- [ ] Fair adaptation baseline report
- [ ] Frozen-backbone integration and state-ordering tests
- [ ] PMetal parity and gradient report
- [ ] Full adaptation, overhead, and safety report

## Contract Checklist

- [ ] Define `W_effective = W_base + A_fast @ B_fast` placement and rank.
- [ ] Define update frequency, decay, clipping, normalization, and per-session budget.
- [ ] Prove permanent weights remain unchanged.
- [ ] Prove snapshot/rollback and reset fidelity exactly.
- [ ] Define serialization, batching, deterministic ordering, and gradient flow.
- [ ] Bound maximum fast-state memory and update cost.

## Exit Gates

- D0: mechanism is specified independently of archived code.
- D1: every state tensor and lifecycle operation is documented.
- D2: temporary mappings work and prior state is restored exactly.
- D3: one bounded update method clearly beats simple alternatives.
- D5/D6: inactive fast weights reproduce HZ-0C behavior; active weights improve adaptation.
- D7/D9: state transitions and PMetal gradients match the reference.
- D10: fast adaptation beats fair baselines while respecting all bounds.

## First Milestone Checklist

- [ ] Implement a standalone low-rank fast-weight simulator.
- [ ] Learn a temporary mapping from a few examples.
- [ ] Snapshot, introduce interference, and roll back exactly.
- [ ] Reset to the baseline state exactly.
- [ ] Report adaptation speed, state norm, interference, rollback, and reset metrics.
