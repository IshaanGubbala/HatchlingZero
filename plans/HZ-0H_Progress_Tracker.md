# HZ-0H Progress Tracker

Updated: August 8, 2026

## Current status

- Overall phase: **H0 in progress** (real work done, not yet complete -- see below).
- H0-H2 may be isolated; H3-H8 wait for the HZ-0G canonical-backbone decision.
- T0-T2 may proceed as a separate ternary-training lane, but cannot replace the
  required full-precision BDH/GDN-2/Transformer baselines.
- Canonical HZ backbone: unchanged. No BDH code or mechanism is promoted.
- Current decision: **UNRESOLVED** until the faithful oracle, streaming-state,
  and full-precision comparison gates pass.

## Phase checklist

| Phase | Status | Exit evidence |
| --- | --- | --- |
| H0 provenance/component map | In progress -- paper (arXiv 2509.26507) and official `bdh.py` source both fetched and read directly; two paper-summary claims (no-RoPE, no-weight-tying) checked against real code and corrected (both were wrong -- RoPE is present, depth weights ARE shared). `train.py`/`BDHConfig` defaults and the Section 4.2 scaling-law table not yet extracted. | Every claim labeled by source/evidence type |
| H1 Torch/MLX BDH-GPU oracle | Not started | Forward/gradient parity and deterministic resume |
| H2 parallel/streaming `rho` equivalence | Not started | Agreement across lengths and chunk boundaries |
| T0 ternary training design memo | Not started | Quantization contract and success metrics documented |
| T1 ternary sandbox on simple baselines | Not started | Stable training recipe on same-architecture controls |
| T2 same-architecture FP vs ternary study | Blocked on T0-T1 | Convergence, memory, throughput, and resume comparison |
| H3 matched BDH/GDN-2/Transformer study | Blocked on HZ-0G G1-G5 | Curves plus quality, compute, state, latency, memory |
| T3 post-H3 ternary replay of surviving arms | Blocked on H3 and T2 | Whether ternary preserves ranking or changes deployment frontier |
| H4 component ablations | Blocked on H3 | Component wins/losses with controls |
| H5 synaptic memory comparison | Blocked on H1/H2 | Recall, reversal, interference, reset, strengthening |
| H6 effective graph tests | Blocked on H1 | Topology ablation and justified sparse execution |
| H7 maximum four graft candidates | Blocked on H4-H6 | Promotion decision per candidate |
| T4 ternary graft qualification | Blocked on H7 and T3 | Surviving grafts retain value under ternary |
| H8 causal interpretability | Blocked on H1/H6 | Stable selectivity plus causal ablation |

## Required artifacts

- [x] `docs/restart/hz0h_bdh_history_audit.md` -- real, sourced (paper + raw official code read directly), not complete (Section 4.2 table, train.py/BDHConfig defaults still open)
- [x] `docs/restart/hz0h_bdh_component_map.md` -- structured, labeled per H0's own taxonomy, cites the audit doc's corrections
- [ ] `docs/restart/hz0h_ternary_training_design.md`
- [ ] `reference/hz0h_bdh_torch.py`
- [ ] `reference/hz0h_bdh_mlx.py`
- [ ] `tests/reference/test_hz0h_bdh_parity.py`
- [ ] Paper-regime reproduction report
- [ ] HZ-regime matched comparison report
- [ ] T1 ternary sandbox report
- [ ] T2 same-architecture FP vs ternary report
- [ ] T3 surviving-arm ternary replay report
- [ ] H4 component ablation report
- [ ] H5 memory report
- [ ] H6 graph-structure report
- [ ] H7 selective-graft report
- [ ] T4 ternary graft qualification report
- [ ] H8 causal-interpretability report
- [ ] Final KEEP / REJECT / UNRESOLVED decision

## Decision log

| Date | Decision | Consequence |
| --- | --- | --- |
| 2026-08-07 | HZ-0H is reconciliation, not BDH integration | No canonical HZ changes before evidence |
| 2026-08-07 | H0-H2 may be isolated; H3-H8 depend on HZ-0G | No second ungoverned backbone |
| 2026-08-07 | BDH-GPU and BDH-GPU' are separate variants | Vanilla BDH is first; gated/merged is labeled separately |
| 2026-08-08 | Ternary work is a side lane, not a substitute for BDH reconciliation baselines | H3 remains full precision; ternary starts with same-architecture controls |
| 2026-08-08 | H0: confirmed directly against raw `bdh.py` that RoPE is present and depth weights are shared/tied across all layers -- both contradict what an initial paper-summary pass suggested | H1's faithful port must implement shared depth weights (not per-layer params) and RoPE; H4's "shared vs untied" ablation treats shared as BDH's real default, not a variant |

## Promotion rule

A BDH mechanism may be proposed for HZ-1 only after passing relevant
oracle/parity tests, beating or materially complementing existing HZ on a
predeclared metric under a matched control, avoiding unacceptable quality/
memory/latency/state regressions, and producing a reproducible report that
records negative and mixed results.

## Ternary guardrail

No result from ternary/1.58-bit training changes the HZ-0H architecture
conclusion unless the corresponding full-precision control is already known.
Treat ternary as an efficiency qualification layer on top of established
architecture evidence, not as the evidence itself.
