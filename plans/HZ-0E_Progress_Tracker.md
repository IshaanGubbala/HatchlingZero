# HZ-0E Progress Tracker

Updated: July 27, 2026

## Mission

Add a conservative four-expert, top-1 micro-MoE FFN with shared dense fallback to a completed HZ-0D model, measuring specialization and active compute without destabilizing recurrence, memory, surprise-triggered attention, or fast weights.

## Current Status

- Overall phase: `E0 not started`
- Dependency status: router design and simulation may begin; full integration is blocked on a frozen HZ-0D with stable HZ-0B/C behavior and lifecycle semantics
- Last verified HZ-0E evidence: none
- Current stopping rule: keep experts out of recurrent-state and memory-write internals initially

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| E0 | history audit and recovered requirements | Not started | Create `docs/restart/hz0e_history_audit.md` and `hz0e_recovered_requirements.md` |
| E1 | expert contract | Not started | Four experts, top-1, capacity, fallback, counts, deterministic inference |
| E2 | isolated router simulator | Not started | Mixed domains, imbalance, shifts, noise, utilization, overflow, collapse |
| E3 | routing objectives | Not started | Task loss, load balance, z-loss, overflow, diversity, warm starts |
| E4 | fair baselines | Not started | Matched active/total dense, wider dense, adapters, static/shared experts |
| E5 | HZ-0D dependency gate | Blocked | Requires frozen recurrence, memory, trigger, fast-state, PMetal, checkpoint, baselines |
| E6 | frozen-backbone integration | Blocked | Replace selected upper MLPs only |
| E7 | interaction rules | Not started | No uncontrolled routing/surprise/memory/fast-weight feedback loop |
| E8 | specialization curriculum | Not started | Balanced domains through mixed and adversarially imbalanced sequences |
| E9 | PMetal implementation | Blocked | Dispatch, capacity, fallback, grouped tokens, Apple-Silicon residency |
| E10 | evaluation | Not started | Quality, utilization, active/total params, overhead, tail latency, interactions |

## Required Artifacts

- [ ] `docs/restart/hz0e_history_audit.md`
- [ ] `docs/restart/hz0e_recovered_requirements.md`
- [ ] Exact expert and active-parameter contract
- [ ] Deterministic router simulator and utilization report
- [ ] Routing-objective comparison
- [ ] Matched-parameter and matched-active-compute baseline report
- [ ] Frozen-backbone interaction and stability tests
- [ ] PMetal dispatch/capacity parity and overhead report
- [ ] Specialization and end-to-end evaluation report

## Contract Checklist

- [ ] Record total parameters and active parameters separately.
- [ ] Define expert placement, size, top-k, capacity factor, overflow behavior, and shared fallback.
- [ ] Keep routing deterministic at inference.
- [ ] Keep experts out of GDN-2, HZ-0B writes, HZ-0C surprise, and HZ-0D update controllers initially.
- [ ] Bound dispatch overhead, overflow, memory, and tail latency.

## Exit Gates

- E0: routing and expert design is independent of archived code.
- E1: exact total and active parameter counts are known.
- E2: multiple experts remain active without collapse.
- E3: balancing does not overwhelm task learning.
- E5/E6: MoE preserves stateful HZ-0B/C/D behavior.
- E7/E9: interaction rules are deterministic and PMetal matches the reference.
- E8/E10: specialization is measurable and MoE beats fair dense baselines at matched active compute or quality.

## First Milestone Checklist

- [ ] Build a four-expert top-1 router with shared fallback.
- [ ] Test balanced synthetic prose, code, math, JSON, and tool domains.
- [ ] Measure utilization, entropy, overflow, collapse, and routing stability.
- [ ] Compare against a dense baseline after routing overhead.
- [ ] Report total and active parameters separately.
