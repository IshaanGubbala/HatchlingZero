# HATCHLING-ZERO Master Progress Tracker

Updated: August 7, 2026

## Purpose

This tracker translates the master development plan into restart-era execution status. It is a governance file for the full HATCHLING-ZERO program, not a source of implementation truth -- see each stage's own `plans/HZ-0X_Progress_Tracker.md` and `docs/restart/` evidence docs for real detail.

## Overall Status

- Program state: HZ-0A through HZ-0F complete. HZ-0G (architecture freeze + integration) is active, and HZ-0H (BDH reconciliation) is planned but not started; see `plans/HZ-0G_Integration_Plan.md` and `plans/HZ-0H_BDH_Reconciliation_Plan.md`.
- Active focus: G1 -- validating the corrected exact-GDN-2 backbone at real scale (100M -> 500M -> 2B -> 6B token continuation ladder), gated on whether it beats a matched Transformer at 500M-2B tokens.
- Blocking rule carried into HZ-0G: no new mechanism until G0-G5 close, per `plans/HZ-0G_Integration_Plan.md`'s explicit out-of-scope list.

## Stage Summary

| Stage | Goal | Status | Notes |
| --- | --- | --- | --- |
| HZ-0A | Recurrent-hybrid base, exact GDN-2 correction | Complete (architecture); G1 scale validation open | Corrected exact-GDN-2 implementation done; only real training evidence at 301M scale is a 10M-token run -- G1's job |
| HZ-0B | Session-local associative memory | Complete | Real evidence in `docs/restart/hz0b_*` |
| HZ-0C | Surprise-triggered anchor attention | Complete | Real evidence in `docs/restart/hz0c_*` |
| HZ-0D | Bounded fast-weight updates | Complete | Real evidence in `docs/restart/hz0d_*` |
| HZ-0E | Micro-MoE specialization | Complete -- mixed, disclosed both ways | `docs/restart/hz0e_e10_evaluation_results.md`: beats fair dense on per-domain quality at matched active compute (6/6 trials), loses general/OOD quality (real, structural), PMetal never beat plain MLX across 5 iterations |
| HZ-0F | MoE generalization + technique investigations | Complete | `docs/restart/hz0e_f_investigation_summary.md`: fallback/routing tradeoff confirmed via 3 independent mechanisms; `mx.gather_mm` adopted; AttnRes rejected at tiny scale (unresolved at 100M+); counterfactual routing not adopted |
| HZ-0G | Architecture freeze + integration | G0 done (this tracker + the plan doc); G1 starting | `plans/HZ-0G_Integration_Plan.md` |
| HZ-0H | BDH reconciliation + selective integration | Planned; H0-H2 may be isolated, H3-H8 gated on HZ-0G | `plans/HZ-0H_BDH_Reconciliation_Plan.md` |

## Program Rules

- No legacy implementation is trusted without re-derivation or reproduction.
- Historical metrics are evidence, not claims.
- Every stage needs:
  - history audit
  - recovered requirements/spec
  - tiny simulator/reference
  - tests
  - baseline comparisons
  - integration only after isolated validation
- Carried into HZ-0G: real evidence disclosed both ways, every finding (positive, negative, or unresolved) reported honestly -- this discipline is what HZ-0E/F ran on throughout and is not being relaxed for the integration phase.

## Current Evidence Base

- HZ-0A restart docs:
  - `docs/restart/hz0a_history_audit.md`, `docs/restart/hz0a_recovered_spec.md`
  - `plans/GDN-2_Fix.md` -- the exact-GDN-2 correction itself
- HZ-0B/C/D/E/F: see each stage's own `plans/HZ-0X_Progress_Tracker.md` and `docs/restart/hz0x_*` evidence docs -- too many real, individually-cited findings to duplicate here without drifting out of sync.
- The repo has been intentionally reduced, with legacy material moved under `archive/`.

## Immediate Next Milestones

1. G1: run the exact-GDN-2 continuation ladder (100M -> 500M -> 2B -> 6B tokens) on the real 301M backbone, evaluating at every gate against held-out CE, the pre-correction HZ recurrence baseline, and a matched Transformer.
2. G2-G4: revalidate B/C/D against whichever checkpoint G1 produces, once it exists.
3. G5: real Dense vs. MoE vs. domain-adapter decision on the fully integrated checkpoint.
4. Prepare HZ-0H H0-H2 provenance and oracle work without changing the canonical HZ backbone.

## Risks

- Legacy docs contain conflicting confidence levels; overclaims must not leak into the restart.
- The old "110M" versus "292M" confusion can invalidate comparisons if reused carelessly.
- G1's own real risk, stated in `plans/HZ-0G_Integration_Plan.md`: the exact-GDN-2 backbone's only real evidence at 301M scale is a 10M-token run. If its early advantage does not survive to 500M-2B tokens against a matched Transformer, that must be reported as a real negative result, not smoothed over because the rest of the program is built on top of it.

## Stop / Go Gate

- `GO` for HZ-0G G1 (exact-GDN-2 scaling ladder)
- `STOP` for promoting any new architecture mechanism, including BDH components, until HZ-0G's G0-G5 gates close, per `plans/HZ-0G_Integration_Plan.md`. Isolated HZ-0H provenance/oracle work is allowed under `plans/HZ-0H_BDH_Reconciliation_Plan.md`.
