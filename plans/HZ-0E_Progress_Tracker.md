# HZ-0E Progress Tracker

Updated: August 5, 2026 (E0 complete: real `git log --all` sweep found zero prior MoE/routing work in this project's own code -- a genuine clean slate, unlike HZ-0D's D0 which recovered real reusable prior work under a relocated name. The one real, concrete finding: HZ-0A's existing 31-block architecture gives every layer, attention or recurrent, an identical dense SwiGLU FFN (`gate`/`up`: 768->2304, `down`: 2304->768, 5,313,792 params/block, 164,727,552 total), measured directly from the real frozen checkpoint -- the concrete substrate E1's expert contract will specify against.)

## Mission

Add a conservative four-expert, top-1 micro-MoE FFN with shared dense fallback to a completed HZ-0D model, measuring specialization and active compute without destabilizing recurrence, memory, surprise-triggered attention, or fast weights.

## Current Status

- Overall phase: `E0 complete; E1 (expert contract) is next`
- Dependency status: **HZ-0D is complete and its dependency gate is satisfied** (`plans/HZ-0D_Progress_Tracker.md`, `docs/restart/hz0d_d10_evaluation_results.md`), further confirmed by the real joint HZ-0A/B/C/D evaluation (`docs/restart/hz0abcd_joint_evaluation_results.md`); E1-E4 (MoE contract/simulator/objectives/baselines) may proceed, with E2 (isolated router simulator) allowed to start even before E1 fully closes, matching the plan's own text and HZ-0D's own D2-before-D5 precedent.
- Last verified HZ-0E evidence: `docs/restart/hz0e_history_audit.md`, `docs/restart/hz0e_recovered_requirements.md` (2026-08-05) -- no prior implementation to recover; real substrate found and measured instead (see above)
- Current stopping rule: keep experts out of recurrent-state and memory-write internals initially

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| E0 | history audit and recovered requirements | **Complete** | `docs/restart/hz0e_history_audit.md`, `docs/restart/hz0e_recovered_requirements.md`. Full `git log --all` sweep (grep + filename scan across the project's own source trees, not installed packages) found zero prior MoE/expert/router work -- a genuine clean slate. Real substrate found instead: HZ-0A's 31-block model (6 attention layers at indices 4/9/14/19/24/29, 25 GDN-2 layers) gives every block an identical dense SwiGLU FFN (`gate`/`up`: 768->2304, `down`: 2304->768), measured directly from the real checkpoint at 5,313,792 params/block, 164,727,552 total -- the concrete "upper MLP blocks" substrate E1 will specify its expert-candidate layer set against. |
| E1 | expert contract | Not started | Four experts, top-1, capacity, fallback, counts, deterministic inference |
| E2 | isolated router simulator | Not started | Mixed domains, imbalance, shifts, noise, utilization, overflow, collapse |
| E3 | routing objectives | Not started | Task loss, load balance, z-loss, overflow, diversity, warm starts |
| E4 | fair baselines | Not started | Matched active/total dense, wider dense, adapters, static/shared experts |
| E5 | HZ-0D dependency gate | Ready after E0-E4 | HZ-0D recurrence, memory, trigger, fast-state, PMetal, checkpoint, and baselines are complete; E0-E4 must define the MoE contract before integration. |
| E6 | frozen-backbone integration | Pending E0-E5 | Replace selected upper MLPs only after router and interaction rules are frozen. |
| E7 | interaction rules | Not started | No uncontrolled routing/surprise/memory/fast-weight feedback loop |
| E8 | specialization curriculum | Not started | Balanced domains through mixed and adversarially imbalanced sequences |
| E9 | PMetal implementation | Pending E0-E8 | Dispatch, capacity, fallback, grouped tokens, Apple-Silicon residency |
| E10 | evaluation | Not started | Quality, utilization, active/total params, overhead, tail latency, interactions |

## Required Artifacts

- [x] `docs/restart/hz0e_history_audit.md`
- [x] `docs/restart/hz0e_recovered_requirements.md`
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

- [x] E0: routing and expert design is independent of archived code. -- `docs/restart/hz0e_history_audit.md`: no archived MoE code exists to depend on; the one real prior-art dependency (HZ-0A's own live, current per-block dense FFN) is current infrastructure, not archived material
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
