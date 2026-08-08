# HZ-0H Progress Tracker

Updated: August 7, 2026

## Current status

- Overall phase: **Planned, not started**.
- H0-H2 may be isolated; H3-H8 wait for the HZ-0G canonical-backbone decision.
- Canonical HZ backbone: unchanged. No BDH code or mechanism is promoted.
- Current decision: **UNRESOLVED** until the faithful oracle and streaming-state gates pass.

## Phase checklist

| Phase | Status | Exit evidence |
| --- | --- | --- |
| H0 provenance/component map | Not started | Every claim labeled by source/evidence type |
| H1 Torch/MLX BDH-GPU oracle | Not started | Forward/gradient parity and deterministic resume |
| H2 parallel/streaming `rho` equivalence | Not started | Agreement across lengths and chunk boundaries |
| H3 matched BDH/GDN-2/Transformer study | Blocked on HZ-0G G1-G5 | Curves plus quality, compute, state, latency, memory |
| H4 component ablations | Blocked on H3 | Component wins/losses with controls |
| H5 synaptic memory comparison | Blocked on H1/H2 | Recall, reversal, interference, reset, strengthening |
| H6 effective graph tests | Blocked on H1 | Topology ablation and justified sparse execution |
| H7 maximum four graft candidates | Blocked on H4-H6 | Promotion decision per candidate |
| H8 causal interpretability | Blocked on H1/H6 | Stable selectivity plus causal ablation |

## Required artifacts

- [ ] `docs/restart/hz0h_bdh_history_audit.md`
- [ ] `docs/restart/hz0h_bdh_component_map.md`
- [ ] `reference/hz0h_bdh_torch.py`
- [ ] `reference/hz0h_bdh_mlx.py`
- [ ] `tests/reference/test_hz0h_bdh_parity.py`
- [ ] Paper-regime reproduction report
- [ ] HZ-regime matched comparison report
- [ ] H4 component ablation report
- [ ] H5 memory report
- [ ] H6 graph-structure report
- [ ] H7 selective-graft report
- [ ] H8 causal-interpretability report
- [ ] Final KEEP / REJECT / UNRESOLVED decision

## Decision log

| Date | Decision | Consequence |
| --- | --- | --- |
| 2026-08-07 | HZ-0H is reconciliation, not BDH integration | No canonical HZ changes before evidence |
| 2026-08-07 | H0-H2 may be isolated; H3-H8 depend on HZ-0G | No second ungoverned backbone |
| 2026-08-07 | BDH-GPU and BDH-GPU' are separate variants | Vanilla BDH is first; gated/merged is labeled separately |

## Promotion rule

A BDH mechanism may be proposed for HZ-1 only after passing relevant
oracle/parity tests, beating or materially complementing existing HZ on a
predeclared metric under a matched control, avoiding unacceptable quality/
memory/latency/state regressions, and producing a reproducible report that
records negative and mixed results.
