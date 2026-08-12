# HATCHLING-ZERO Master Progress Tracker

Updated: August 7, 2026

## Purpose

This tracker translates the master development plan into restart-era execution status. It is a governance file for the full HATCHLING-ZERO program, not a source of implementation truth -- see each stage's own `plans/HZ-0X_Progress_Tracker.md` and `docs/restart/` evidence docs for real detail.

## Overall Status

- Program state: HZ-0A through HZ-0F complete. HZ-0G's G0-G5 have all run real work at the 100M-token checkpoint (see Stage Summary); HZ-0H is underway (H0/H1 done). See `plans/HZ-0G_Integration_Plan.md` and `plans/HZ-0H_BDH_Reconciliation_Plan.md`.
- Active focus / real open item: G1's own **critical gate** (500M-2B tokens vs. a matched Transformer) is NOT yet run -- only the 100M checkpoint exists. Everything G2-G5 measured is real but at that 100M scale; the plan's own bar for a credible backbone verdict is 500M-2B tokens, still open. Do not read "G0-G5 all have results" as "HZ-0G is closed."
- A same-corpus matched-Transformer control is training separately (RTX 3060) toward making a real comparison possible once both sides exist -- see `docs/rtx3060_g1_matched_transformer.md`.
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
| HZ-0G | Architecture freeze + integration | G0-G5 all have real results at the 100M checkpoint; G1's own critical gate (500M-2B tokens vs. matched Transformer) still open | `plans/HZ-0G_Integration_Plan.md`; see Immediate Next Milestones below for per-gate detail |
| HZ-0H | BDH reconciliation + selective integration | H0 (provenance/component map) and H1 (Torch/MLX BDH-GPU oracle, 5/5 parity tests) done; H2 (streaming equivalence) not started; H3+ gated on HZ-0G | `plans/HZ-0H_BDH_Reconciliation_Plan.md` |

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

1. G1: 100M-token gate reached (real, clean exit, held-out CE 2.4330 nats on 528/529 sequences). Continuation toward 500M -> 2B -> 6B not yet started; a same-corpus matched-Transformer control is running separately (RTX 3060, see `docs/rtx3060_g1_matched_transformer.md`) to make that comparison valid once both exist.
2. G2 (revalidate B): done against the 100M checkpoint -- real, mixed result, not uniform in either direction. See `docs/restart/hz0g_g2_b_revalidation_results.md`.
3. G3 (revalidate C): done against the 100M checkpoint -- triggered attention still beats naive baselines at matched cost, real downstream loss cost of missing triggers is small (+0.009 nats) but scenario-dependent, latency shows no meaningful full-vs-triggered difference at this scale. See `docs/restart/hz0g_g3_c_revalidation_results.md`.
4. G4 (revalidate D): done against the 100M checkpoint -- clean transfer, no reversals (unlike G2/G3): all 5 exit gates pass with wide margins, GD divergence now 8/8 seeds (stronger than the original finding). Real gap remains: rollback/rule-change only tested at pure-mechanism level, not against a real checkpoint. See `docs/restart/hz0g_g4_d_revalidation_results.md`.
5. G5: done against the 100M checkpoint -- HZ-MoE beats Dense+domain-adapter on both general-prose (2.9733 vs 3.0185) and per-domain specialization (2.7355 vs 2.7885), the OPPOSITE of E4's isolated-scale "adapter beats MoE" finding; HZ-Dense (no E) still wins general-prose robustness over both trained arms, confirming E10's specialization-costs-generality finding holds in the full integration too. Single seed, real next-step caveats disclosed. See `docs/restart/hz0g_g5_dense_moe_adapter_results.md`.
6. Prepare HZ-0H H0-H2 provenance and oracle work without changing the canonical HZ backbone.

## Risks

- Legacy docs contain conflicting confidence levels; overclaims must not leak into the restart.
- The old "110M" versus "292M" confusion can invalidate comparisons if reused carelessly.
- G1's own real risk, stated in `plans/HZ-0G_Integration_Plan.md`: the exact-GDN-2 backbone's only real evidence at 301M scale is a 10M-token run. If its early advantage does not survive to 500M-2B tokens against a matched Transformer, that must be reported as a real negative result, not smoothed over because the rest of the program is built on top of it.

## Stop / Go Gate

- `GO` for HZ-0G G1 (exact-GDN-2 scaling ladder)
- `STOP` for promoting any new architecture mechanism, including BDH components, until HZ-0G's G0-G5 gates close, per `plans/HZ-0G_Integration_Plan.md`. Isolated HZ-0H provenance/oracle work is allowed under `plans/HZ-0H_BDH_Reconciliation_Plan.md`.
