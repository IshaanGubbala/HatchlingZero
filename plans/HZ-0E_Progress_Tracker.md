# HZ-0E Progress Tracker

Updated: August 5, 2026 (E0-E1 complete. E1: real, implemented, tested 4-expert top-1 MoE contract -- 3 upper GDN-2 layers (27/28/30, deliberately disjoint from HZ-0D's fast-weight layers), 576-wide experts, a full-size unscaled shared-dense-fallback for overflow, capacity_factor=1.5. Exact computed numbers: total model params +5.30% (317,135,628), typical-case ACTIVE params -3.97% (289,233,036) versus the 301,178,112 dense baseline -- capacity grows while active compute per token actually shrinks, the real version of the plan's own framing question.)

## Mission

Add a conservative four-expert, top-1 micro-MoE FFN with shared dense fallback to a completed HZ-0D model, measuring specialization and active compute without destabilizing recurrence, memory, surprise-triggered attention, or fast weights.

## Current Status

- Overall phase: `E0-E1 complete; E2 (isolated router simulator) is next`
- Dependency status: **HZ-0D is complete and its dependency gate is satisfied** (`plans/HZ-0D_Progress_Tracker.md`, `docs/restart/hz0d_d10_evaluation_results.md`), further confirmed by the real joint HZ-0A/B/C/D evaluation (`docs/restart/hz0abcd_joint_evaluation_results.md`); E2-E4 (router simulator/objectives/baselines) may proceed against E1's real, tested contract.
- Last verified HZ-0E evidence: `docs/restart/hz0e_e1_contract.md` (2026-08-05) -- real, implemented (`reference/hz0e_moe_contract.py`), tested (`tests/reference/test_hz0e_moe_contract.py`, 13 tests) 4-expert top-1 contract. Placement: layers 27/28/30 (deliberately excludes layer 29 -- an attention layer already carrying a HZ-0D fast-weight-augmented output projection -- keeping HZ-0E's first integration structurally disjoint from HZ-0D's touch points). Experts: `dim=768 -> expert_d_ff=576 -> dim=768`, sized so 4 experts' combined width equals the original dense FFN's 2304. Shared fallback: full dense-FFN size, UNSCALED, engaged only for capacity overflow (a real, disclosed design choice among two plausible readings of "shared dense fallback," resolved explicitly rather than left ambiguous). `capacity_factor=1.5`. Exact numbers, computed from real shapes and cross-checked by independent hand arithmetic and an independent test computation: total model params 301,178,112 -> 317,135,628 (+5.30%); typical-case (no-overflow) ACTIVE params 301,178,112 -> 289,233,036 (-3.97%) -- capacity grows, active compute per token actually shrinks. A real MLX numerical finding disclosed along the way: matmul takes a measurably different path for batch-of-N vs batch-of-1 inputs (~6e-4 absolute difference), fixed by comparing routing-correctness tests against the same batched reference computation, not a separately re-run single-row one. Prior verified evidence: `docs/restart/hz0e_history_audit.md`, `docs/restart/hz0e_recovered_requirements.md`
- Current stopping rule: keep experts out of recurrent-state and memory-write internals initially -- met by construction (E1's placement choice touches zero layers HZ-0B/C/D's own mechanisms use)

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| E0 | history audit and recovered requirements | **Complete** | `docs/restart/hz0e_history_audit.md`, `docs/restart/hz0e_recovered_requirements.md`. Full `git log --all` sweep (grep + filename scan across the project's own source trees, not installed packages) found zero prior MoE/expert/router work -- a genuine clean slate. Real substrate found instead: HZ-0A's 31-block model (6 attention layers at indices 4/9/14/19/24/29, 25 GDN-2 layers) gives every block an identical dense SwiGLU FFN (`gate`/`up`: 768->2304, `down`: 2304->768), measured directly from the real checkpoint at 5,313,792 params/block, 164,727,552 total -- the concrete "upper MLP blocks" substrate E1 will specify its expert-candidate layer set against. |
| E1 | expert contract | **Complete** | `reference/hz0e_moe_contract.py`, `tests/reference/test_hz0e_moe_contract.py` (13 tests), `docs/restart/hz0e_e1_contract.md`. 4 experts (576-wide each), top-1 deterministic routing, capacity_factor=1.5, full-size unscaled shared-dense fallback for overflow tokens, placement at layers 27/28/30 (deliberately disjoint from HZ-0D's 6 fast-weight layers). Exact params: layer total 10,632,964 (vs. 5,313,792 dense baseline); layer active (no overflow) 1,332,100. Whole-model: 317,135,628 total (+5.30%), 289,233,036 active typical-case (-3.97%). HZ-0D fast weights explicitly do NOT modify expert/router/fallback weights at this stage. |
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
- [x] Exact expert and active-parameter contract (`docs/restart/hz0e_e1_contract.md`)
- [ ] Deterministic router simulator and utilization report
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
