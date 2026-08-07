# HZ-0E E8: Router Specialization Warm-Start

Date: 2026-08-05. The existing supervised router warm-start now composes with
E6's pretrained expert initialization through `start_params`, and validates
that every domain label is in the four-expert range.

The five available real domains cannot map one-to-one onto four experts, so
the curriculum uses the explicit mapping `prose->0`, `code->1`, `math->2`,
`json->3`, `tools->0`. This is a deliberate shared-expert assignment, not an
invalid fifth class.

After 40 deterministic router-only steps at `lr=1e-3`, real frozen-checkpoint
FFN activations achieved:

| Domain | Target expert | Routing accuracy |
|---|---:|---:|
| prose | 0 | 98.29% |
| code | 1 | 95.75% |
| math | 2 | 97.80% |
| JSON | 3 | 98.93% |
| tools | 0 | 98.49% |

Only router weights were updated; expert and fallback weights remain from the
E6 warm start. The earlier invalid-label probe (`tools->4`) was rejected and
is no longer possible because `supervised_warm_start` raises on labels outside
`[0, num_experts)`.

This fixes the router specialization prerequisite, not the complete E8 exit
gate: expert task-loss training and cross-domain quality still require the
balanced curriculum evaluation.

Full HZ-0E regression after this change: **43 passed in 37.08s** across E1,
E2, E3, E4, E6, and E7. The invalid-label failure found during the first
probe is therefore fixed without regressing the existing routing-objective or
fair-baseline behavior.

The invalid-label contract now has a dedicated regression test,
`test_hz0e_label_contract.py`; the focused label/E6/E7 guard suite is **8
passed**.
