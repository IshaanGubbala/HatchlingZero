# HZ-0D Progress Tracker

Updated: August 4, 2026 (D0, D1, AND D2 complete: real prior work audited, the fast-weight contract specified and implemented with real tests, and the isolated simulator demonstrates real few-shot mapping learning with exact snapshot/rollback/reset under real interference -- first milestone met)

## Mission

Add bounded session-local low-rank fast weights to a completed HZ-0C model, with strict isolation, snapshot, rollback, reset, and serialization semantics. Permanent pretrained weights must never change during ordinary use.

## Current Status

- Overall phase: `D0, D1, D2 complete; D3 (update-mechanism selection) is next`
- Dependency status: HZ-0C is now COMPLETE (`plans/HZ-0C_Progress_Tracker.md`) -- D5's dependency gate is satisfiable whenever integration work reaches it; D3/D4 (still isolated) may proceed regardless, per the plan's own text
- Last verified HZ-0D evidence: `docs/restart/hz0d_d2_isolated_simulator_results.md` (2026-08-04) -- real few-shot symbol-remapping generalization (73.7% mean held-out loss reduction across 8 seeds, always positive), a real calibration finding disclosed (an initial dim=32/rank=4/k_train=4 config gave near-pure memorization, fixed by right-sizing), contradictory-rule-change interference measured directly, decay monotonic in both norm and task loss, snapshot/rollback and reset verified both bit-identical AND behaviorally exact under real interference, noisy updates stay bounded, malicious updates are clipped and fully recoverable. First Milestone Checklist fully met.
- Current stopping rule: do not modify the recurrent core or integrate before the contract and simulator pass (met) -- D3's update-mechanism comparison must also pass a correctness check per D0's own lesson before any method is trusted

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| D0 | history audit and recovered requirements | **Complete** | `docs/restart/hz0d_history_audit.md`, `hz0d_recovered_requirements.md` -- real `git log --all` sweep found substantial prior work (the old "Phase 16" HZ-0C-named fast weights, 6 commits, archived at `archive/src/hz0/fast_weights/`). Snapshot/rollback/session-management design is real and reusable as a pattern. The "gradient-based" adaptation mechanism was NOT real gradient descent -- verified by reading the actual update code, which discards the one signal (`loss_pert`) that would make it a valid finite-difference estimator and instead applies unbiased random noise. This was self-admitted in the phase's own commit message ("insufficient for learning") but the following phase declared "production-ready" anyway; both facts are disclosed. |
| D1 | fast-weight contract | **Complete** | `docs/restart/hz0d_d1_contract.md`, `reference/hz0d_fast_weights.py`, `tests/reference/test_hz0d_fast_weights.py` (15 tests, all real code not stubs). Layers: anchor-attention output projection at the 6 existing HZ-0C `ATTENTION_INDICES`. Rank 16 at dim 768 (147,456 params / 576 KiB per session, audited exactly, not estimated). Update: real `mx.grad`-based gradient descent, verified against finite differences to `<1e-2` and shown to strictly reduce loss on both a single step and a 200-step toy-mapping convergence test (>50% loss reduction). Clipping: realized-delta Frobenius norm bounded regardless of input gradient magnitude. Decay: multiplicative, `1.0` exact no-op, tested. Snapshot/rollback/reset: bit-identical (`mx.array_equal`), not approximate. Serialization: exact round-trip. A real bug (symmetric zero-init gives exactly-zero gradients for both factors, a dead saddle point) was found BY the test suite and fixed with the standard asymmetric LoRA init, documented in the contract doc as a real example of D0's lesson applied. |
| D2 | isolated simulator | **Complete** | `reference/hz0d_isolated_simulator.py`, `tests/reference/test_hz0d_isolated_simulator.py` (8 tests), `docs/restart/hz0d_d2_isolated_simulator_results.md`. Real few-shot symbol-remapping task (fixed low-rank rule, train/held-out symbol split); a real calibration finding (dim=32/rank=4/k_train=4 gave pure memorization, 5.9% held-out improvement -- fixed by right-sizing to dim=8/rank=2/k_train=6, verified across 10 seeds: 30-98% held-out loss reduction, always positive) is documented, not hidden. Adaptation speed measured across step checkpoints. Contradictory-rule-change interference measured directly (rule 1's held-out loss rises from 0.244 to 0.816 after switching to rule 2). Decay shown monotonic in both state norm and task loss. Snapshot/rollback and reset verified BOTH bit-identical (`mx.array_equal`) AND behaviorally exact (held-out task loss equal, not just tensors) under real interference. Noisy updates stay finite/bounded through 400 steps; a malicious 1e6-magnitude gradient is clipped exactly and the state remains fully recoverable via rollback/reset afterward. |
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
- [x] Fast-state tensor/lifecycle contract (`docs/restart/hz0d_d1_contract.md`, `reference/hz0d_fast_weights.py`)
- [x] Standalone low-rank simulator and exact lifecycle tests (`reference/hz0d_isolated_simulator.py`, `tests/reference/test_hz0d_isolated_simulator.py`)
- [ ] Update-mechanism comparison report
- [ ] Fair adaptation baseline report
- [ ] Frozen-backbone integration and state-ordering tests
- [ ] PMetal parity and gradient report
- [ ] Full adaptation, overhead, and safety report

## Contract Checklist

- [x] Define `W_effective = W_base + A_fast @ B_fast` placement and rank. -- anchor-attention output projection, 6 layers, rank 16
- [x] Define update frequency, decay, clipping, normalization, and per-session budget. -- `max_updates_per_session=50` (policy field), multiplicative decay, delta-norm clipping, no normalization beyond clipping (documented as a deliberate simplicity choice)
- [x] Prove permanent weights remain unchanged. -- `base_weight`/`base_bias` are plain arrays with no gradient path from `(a_fast, b_fast)`, tested directly (`test_permanent_weight_never_appears_in_fast_weight_gradient`)
- [x] Prove snapshot/rollback and reset fidelity exactly. -- `mx.array_equal`, not approximate closeness
- [x] Define serialization, batching, deterministic ordering, and gradient flow. -- exact round-trip serialization; deterministic seeded init; gradient flow verified via finite differences
- [x] Bound maximum fast-state memory and update cost. -- `fast_state_memory_bytes` computed from real shapes, tested against the hand-computed default (589,824 bytes)

## Exit Gates

- [x] D0: mechanism is specified independently of archived code. -- `docs/restart/hz0d_history_audit.md`, `hz0d_recovered_requirements.md`
- [x] D1: every state tensor and lifecycle operation is documented. -- `docs/restart/hz0d_d1_contract.md` (prose) + `reference/hz0d_fast_weights.py` (field-level docstrings) + `tests/reference/test_hz0d_fast_weights.py` (15 tests, every operation independently verified, not just documented)
- [x] D2: temporary mappings work and prior state is restored exactly. -- `docs/restart/hz0d_d2_isolated_simulator_results.md`: 73.7% mean held-out loss reduction across 8 seeds (always positive), snapshot/rollback and reset verified both bit-identical and behaviorally exact under real interference
- [ ] D3: one bounded update method clearly beats simple alternatives.
- [ ] D5/D6: inactive fast weights reproduce HZ-0C behavior; active weights improve adaptation.
- [ ] D7/D9: state transitions and PMetal gradients match the reference.
- [ ] D10: fast adaptation beats fair baselines while respecting all bounds.

## First Milestone Checklist

- [x] Implement a standalone low-rank fast-weight simulator. -- `reference/hz0d_isolated_simulator.py`
- [x] Learn a temporary mapping from a few examples. -- 73.7% mean held-out loss reduction across 8 seeds, `k_train=6`
- [x] Snapshot, introduce interference, and roll back exactly. -- bit-identical AND behaviorally exact (held-out loss equal) after real interference
- [x] Reset to the baseline state exactly. -- bit-identical to fresh init, behaviorally exact task loss
- [x] Report adaptation speed, state norm, interference, rollback, and reset metrics. -- `docs/restart/hz0d_d2_isolated_simulator_results.md`

**First milestone met.**
