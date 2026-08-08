# HZ-0G G4: revalidating HZ-0D's fast-weight mechanism against the corrected `gdn2_fix` backbone

Date: 2026-08-07. Real run of G4's compact diagnostic subset against `outputs/hz0g_g1_gdn2_fix_301m/native_metal_checkpoint_best_full_holdout`. Two parts: `tests/reference/test_hz0d_d10_evaluation.py` (5 real pass/fail exit gates, already checkpoint-parameterized, no fix needed) and `scripts/hz0d_g4_real_checkpoint_report.py` (new -- same logic, prints real magnitudes, since the test file only asserts).

## Result: clean transfer, no reversals

Unlike G2 (B) and G3 (C), which both showed real, mixed, task-specific reversals on the corrected backbone, HZ-0D's mechanism transfers cleanly:

| Check | Result | Gate |
| --- | --- | --- |
| Inactive fast-weights preserves B/C behavior exactly | PASS (bit-identical) | -- |
| Benign adaptation, unrelated-text degradation | relative_delta = **0.0000** | <0.05 |
| Adversarial clipped state (rule_scale=1000x), unrelated-text harm | relative_delta = **0.0000** | <0.05 |
| Delta prediction vs. no_adaptation (6.720) | 0.883 -- beats by 2x | beat by 2x |
| Delta prediction vs. static_random_adapter (4.742) | 0.883 -- beats by 2x | beat by 2x |
| Delta prediction vs. in_context_attention (4.697) | 0.883 -- beats by 2x | beat by 2x |
| Delta prediction vs. longer_context (5.462) | 0.883 -- beats by 2x | beat by 2x |
| Delta prediction vs. knn_retrieval (42.970) | 0.883 -- beats by 2x | beat by 2x |
| Delta prediction stability (max loss across 8 seeds) | 1.1148 | <2.0 |
| Gradient-descent divergence rate at matched, re-tuned lr | **8/8 seeds diverged** (losses 114-2428) | >=1 diverged |

All 5 pytest exit gates pass. The numeric detail confirms the pass/fail isn't marginal -- every gate clears by a wide margin, not a close call.

## One real note: GD divergence is total, not partial

The original D10 finding (pre-correction checkpoint) required only "at least 1 of N seeds diverges" to establish gradient descent's real instability disadvantage vs. delta prediction. On the corrected `gdn2_fix` checkpoint, **all 8 of 8 seeds diverged**, with loss magnitudes (114-2428) far beyond the >10.0 divergence threshold. This is at least as strong evidence for D3's mechanism selection as before, possibly stronger -- not a regression, a sharper version of the same real finding.

## Real, known gap: rollback and rule-change lack real-checkpoint coverage

G4's named diagnostic list includes rollback and rule-change. Both are only tested at the pure-mechanism level (`tests/reference/test_hz0d_fast_weights.py`'s `test_snapshot_and_rollback_are_bit_identical` etc. -- no `load_frozen_model`, no real checkpoint anywhere in that file), the same state B2's memory simulator was in before B6-B11 built real integration. Not built this session -- flagged as open scope, not assumed sufficient.

## What G4 establishes

- HZ-0D's bounded fast-weight mechanism -- benign-adaptation isolation, adversarial containment, baseline-beating, and GD-instability comparison -- all transfer cleanly to the corrected backbone with wide margins, unlike B and C.
- No retraining of permanent weights was needed or done, per the plan's explicit constraint.

## What G4 does not establish

- Rollback and rule-change semantics against a real checkpoint -- open, real follow-up work if pursued.
