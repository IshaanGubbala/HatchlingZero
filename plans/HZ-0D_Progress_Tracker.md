# HZ-0D Progress Tracker

Updated: August 4, 2026 (D0 complete: real prior work found and audited; a critical defect in the old "gradient-based" update mechanism was traced in full)

## Mission

Add bounded session-local low-rank fast weights to a completed HZ-0C model, with strict isolation, snapshot, rollback, reset, and serialization semantics. Permanent pretrained weights must never change during ordinary use.

## Current Status

- Overall phase: `D0 complete; D1 (contract) is next`
- Dependency status: HZ-0C is now COMPLETE (`plans/HZ-0C_Progress_Tracker.md`) -- D5's dependency gate is satisfiable whenever integration work reaches it; the isolated simulator (D1-D4) may proceed regardless, per the plan's own text
- Last verified HZ-0D evidence: `docs/restart/hz0d_history_audit.md`, `docs/restart/hz0d_recovered_requirements.md` (2026-08-04) -- real prior "Phase 16" fast-weight work found (6 commits, `archive/src/hz0/fast_weights/`, read in full); its snapshot/rollback/session design is reusable as a PATTERN, but its adaptation mechanism was never real gradient descent (unbiased random perturbation mislabeled as a two-point gradient estimate, the measured perturbation effect discarded and never used) -- a critical, disclosed finding shaping D1/D3's requirements
- Current stopping rule: do not modify the recurrent core or integrate before the contract and simulator pass; any update mechanism chosen at D3 must pass an explicit correctness check (finite-difference or synthetic-signal sanity test) before being trusted on a real task

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| D0 | history audit and recovered requirements | **Complete** | `docs/restart/hz0d_history_audit.md`, `hz0d_recovered_requirements.md` -- real `git log --all` sweep found substantial prior work (the old "Phase 16" HZ-0C-named fast weights, 6 commits, archived at `archive/src/hz0/fast_weights/`). Snapshot/rollback/session-management design is real and reusable as a pattern. The "gradient-based" adaptation mechanism was NOT real gradient descent -- verified by reading the actual update code, which discards the one signal (`loss_pert`) that would make it a valid finite-difference estimator and instead applies unbiased random noise. This was self-admitted in the phase's own commit message ("insufficient for learning") but the following phase declared "production-ready" anyway; both facts are disclosed. |
| D1 | fast-weight contract | Not started | Layers, ranks, budgets, clipping, decay, lifecycle, gradients, memory -- informed by D0's requirement that the update mechanism must be independently verifiable, not assumed correct |
| D2 | isolated simulator | Not started | Temporary mappings, rule changes, noise, malicious updates, rollback/reset |
| D3 | update-mechanism selection | Not started | Compare Hebbian, learned gradient-like, delta prediction, error-conditioned updates -- each MUST pass a correctness sanity check before being trusted, per D0's finding |
| D4 | fair adaptation baselines | Not started | Prompting, context, memory, HZ-0C, retrieval, static and permanent adapters |
| D5 | HZ-0C dependency gate | **Satisfiable** | HZ-0C is now complete (all phases, `plans/HZ-0C_Progress_Tracker.md`); this gate is no longer a real blocker whenever integration work (D6+) is reached |
| D6 | frozen-backbone integration | Not started | Start only in narrow upper MLP/controller/output locations -- convergent with the old (archived) implementation's own choice of attention QKV projections |
| D7 | state ordering | Not started | One memory write and at most one fast update per token |
| D8 | curriculum | Not started | Supervised updates through natural schemas and adversarial rollback tasks |
| D9 | PMetal implementation | Not started | Apply/update/decay/snapshot/rollback/reset and batched deterministic sessions |
| D10 | evaluation | Not started | Adaptation, retention, interference, fidelity, overhead, degradation, resistance |

## Required Artifacts

- [x] `docs/restart/hz0d_history_audit.md`
- [x] `docs/restart/hz0d_recovered_requirements.md`
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

- [x] D0: mechanism is specified independently of archived code. -- `docs/restart/hz0d_history_audit.md`, `hz0d_recovered_requirements.md`
- [ ] D1: every state tensor and lifecycle operation is documented.
- [ ] D2: temporary mappings work and prior state is restored exactly.
- [ ] D3: one bounded update method clearly beats simple alternatives.
- [ ] D5/D6: inactive fast weights reproduce HZ-0C behavior; active weights improve adaptation.
- [ ] D7/D9: state transitions and PMetal gradients match the reference.
- [ ] D10: fast adaptation beats fair baselines while respecting all bounds.

## First Milestone Checklist

- [ ] Implement a standalone low-rank fast-weight simulator.
- [ ] Learn a temporary mapping from a few examples.
- [ ] Snapshot, introduce interference, and roll back exactly.
- [ ] Reset to the baseline state exactly.
- [ ] Report adaptation speed, state norm, interference, rollback, and reset metrics.
