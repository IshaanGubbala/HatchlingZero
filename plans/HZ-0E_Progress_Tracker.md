# HZ-0E Progress Tracker

Updated: August 5, 2026 (E0-E2 complete. E1: real, implemented, tested 4-expert top-1 MoE contract -- 3 upper GDN-2 layers (27/28/30, deliberately disjoint from HZ-0D's fast-weight layers), 576-wide experts, a full-size unscaled shared-dense-fallback for overflow, capacity_factor=1.5. Exact computed numbers: total model params +5.30% (317,135,628), typical-case ACTIVE params -3.97% (289,233,036) versus the 301,178,112 dense baseline. E2: a 300-configuration real-domain sweep (20 seeds x 5 real domains x 3 target layers) finds zero dead experts and a bounded worst-case raw routing share (57.4%, comfortably short of collapse) -- the untrained router's mechanism is stable across code/prose/math/JSON/tools, mixed domains, heavy imbalance, within-sequence domain shifts, and noise up to 100x real activation scale. A real bug was found and fixed before any E2 number was trusted: the real-activation collector initially skipped a target block's own mixer before its FFN-input norm, caught only by an exact manual-replay comparison, not by output statistics.)

## Mission

Add a conservative four-expert, top-1 micro-MoE FFN with shared dense fallback to a completed HZ-0D model, measuring specialization and active compute without destabilizing recurrence, memory, surprise-triggered attention, or fast weights.

## Current Status

- Overall phase: `E0-E2 complete; E3 (routing objectives) is next`
- Dependency status: **HZ-0D is complete and its dependency gate is satisfied** (`plans/HZ-0D_Progress_Tracker.md`, `docs/restart/hz0d_d10_evaluation_results.md`), further confirmed by the real joint HZ-0A/B/C/D evaluation (`docs/restart/hz0abcd_joint_evaluation_results.md`); E3-E4 (routing objectives/fair baselines) may proceed against E1's real contract and E2's confirmed mechanism stability.
- Last verified HZ-0E evidence: `docs/restart/hz0e_e2_router_simulator_results.md` (2026-08-05) -- real router-mechanism stability check against the ACTUAL frozen checkpoint and REAL corpus text across all 5 plan-named domains (prose/code/math/JSON/tools, tools matched to `terminal_and_debugging_validation.jsonl` as the closest real available domain, disclosed as a substitution). 300-configuration sweep (20 seeds x 5 domains x 3 target layers): zero dead experts, worst-case raw routing share 57.37% (vs. a 75% regression-test bound), entropy 91-93% of theoretical max on clean data. Mixed-domain batches, heavy imbalance (15:1 real sequence ratio), and a literal within-sequence domain shift (code half + math half per row) all stay collapse-free. Noise robustness up to 100x real activation scale reveals a real, disclosed divergence: per-token routing ENTROPY collapses toward zero under extreme noise (decisions get sharper) while AGGREGATE UTILIZATION stays balanced/converges toward uniform (no collapse) -- different signals moving in opposite directions, both checked. Router is untrained at this stage by design (E8's job, not E2's) -- E2 tests mechanism stability, not semantic specialization. A real bug (the real-activation collector skipping a target block's own mixer) was found via exact manual-replay comparison, not output statistics, and fixed before any reported number was trusted. Prior verified evidence: `docs/restart/hz0e_e1_contract.md`, `docs/restart/hz0e_history_audit.md`, `docs/restart/hz0e_recovered_requirements.md`
- Current stopping rule: keep experts out of recurrent-state and memory-write internals initially -- met by construction (E1's placement choice touches zero layers HZ-0B/C/D's own mechanisms use)

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| E0 | history audit and recovered requirements | **Complete** | `docs/restart/hz0e_history_audit.md`, `docs/restart/hz0e_recovered_requirements.md`. Full `git log --all` sweep (grep + filename scan across the project's own source trees, not installed packages) found zero prior MoE/expert/router work -- a genuine clean slate. Real substrate found instead: HZ-0A's 31-block model (6 attention layers at indices 4/9/14/19/24/29, 25 GDN-2 layers) gives every block an identical dense SwiGLU FFN (`gate`/`up`: 768->2304, `down`: 2304->768), measured directly from the real checkpoint at 5,313,792 params/block, 164,727,552 total -- the concrete "upper MLP blocks" substrate E1 will specify its expert-candidate layer set against. |
| E1 | expert contract | **Complete** | `reference/hz0e_moe_contract.py`, `tests/reference/test_hz0e_moe_contract.py` (13 tests), `docs/restart/hz0e_e1_contract.md`. 4 experts (576-wide each), top-1 deterministic routing, capacity_factor=1.5, full-size unscaled shared-dense fallback for overflow tokens, placement at layers 27/28/30 (deliberately disjoint from HZ-0D's 6 fast-weight layers). Exact params: layer total 10,632,964 (vs. 5,313,792 dense baseline); layer active (no overflow) 1,332,100. Whole-model: 317,135,628 total (+5.30%), 289,233,036 active typical-case (-3.97%). HZ-0D fast weights explicitly do NOT modify expert/router/fallback weights at this stage. |
| E2 | isolated router simulator | **Complete** | `reference/hz0e_e2_router_simulator.py`, `tests/reference/test_hz0e_e2_router_simulator.py` (8 tests), `docs/restart/hz0e_e2_router_simulator_results.md`. Real corpus data for all 5 plan-named domains, real frozen-checkpoint activations at all 3 E1 target layers. 300-config sweep: 0 dead experts, worst-case raw share 57.37%. Mixed domains, 15:1 imbalance, within-sequence domain shift, and noise to 100x all stay collapse-free. Real disclosed finding: entropy collapses under extreme noise while utilization stays balanced -- different signals. Real bug (collector skipped a block's own mixer) found and fixed via manual-replay comparison before trusting any number. |
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
- [x] Exact expert and active-parameter contract (`docs/restart/hz0e_e1_contract.md`)
- [x] Deterministic router simulator and utilization report (`docs/restart/hz0e_e2_router_simulator_results.md`)
- [ ] Routing-objective comparison
- [ ] Matched-parameter and matched-active-compute baseline report
- [ ] Frozen-backbone interaction and stability tests
- [ ] PMetal dispatch/capacity parity and overhead report
- [ ] Specialization and end-to-end evaluation report

## Contract Checklist

- [x] Record total parameters and active parameters separately. -- `moe_layer_param_counts` (`reference/hz0e_moe_contract.py`), whole-model figures in `docs/restart/hz0e_e1_contract.md` section 6
- [x] Define expert placement, size, top-k, capacity factor, overflow behavior, and shared fallback. -- `docs/restart/hz0e_e1_contract.md` sections 1-4
- [x] Keep routing deterministic at inference. -- `argmax` routing + fixed-token-order capacity ranking, verified by `test_forward_is_deterministic_given_identical_inputs`
- [x] Keep experts out of GDN-2, HZ-0B writes, HZ-0C surprise, and HZ-0D update controllers initially. -- E1's placement (layers 27/28/30) touches none of HZ-0B/C/D's own mechanisms; HZ-0D fast weights explicitly excluded from modifying expert/router/fallback weights (contract doc section 7)
- [ ] Bound dispatch overhead, overflow, memory, and tail latency. -- overflow is bounded by construction (capacity-based, verified never exceeded); dispatch overhead/tail latency measurement is E9/E10's job, not E1's static contract

## Exit Gates

- [x] E0: routing and expert design is independent of archived code. -- `docs/restart/hz0e_history_audit.md`: no archived MoE code exists to depend on; the one real prior-art dependency (HZ-0A's own live, current per-block dense FFN) is current infrastructure, not archived material
- [x] E1: exact total and active parameter counts are known. -- `docs/restart/hz0e_e1_contract.md` section 6: 317,135,628 total (+5.30%), 289,233,036 typical-case active (-3.97%), both computed from real shapes and cross-checked two independent ways
- [x] E2: multiple experts remain active without collapse. -- `docs/restart/hz0e_e2_router_simulator_results.md`: 0 dead experts across a 300-config real-domain sweep, worst-case raw share 57.37%, stable under mixed domains/imbalance/domain shift/noise to 100x
- E3: balancing does not overwhelm task learning.
- E5/E6: MoE preserves stateful HZ-0B/C/D behavior.
- E7/E9: interaction rules are deterministic and PMetal matches the reference.
- E8/E10: specialization is measurable and MoE beats fair dense baselines at matched active compute or quality.

## First Milestone Checklist

- [x] Build a four-expert top-1 router with shared fallback. -- `reference/hz0e_moe_contract.py` (E1)
- [x] Test balanced prose, code, math, JSON, and tool domains -- real corpus content, not synthetic. -- `reference/hz0e_e2_router_simulator.py`, `docs/restart/hz0e_e2_router_simulator_results.md` (E2)
- [x] Measure utilization, entropy, overflow, collapse, and routing stability. -- same (E2)
- [ ] Compare against a dense baseline after routing overhead. -- E4's job, not yet done
- [x] Report total and active parameters separately. -- `docs/restart/hz0e_e1_contract.md` section 6 (E1)
