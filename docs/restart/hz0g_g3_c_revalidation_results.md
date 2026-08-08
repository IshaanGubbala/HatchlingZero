# HZ-0G G3: revalidating HZ-0C's surprise-triggered attention against the corrected `gdn2_fix` backbone

Date: 2026-08-07. Real run of G3's compact diagnostic subset (5 named comparisons per `plans/HZ-0G_Integration_Plan.md`: fixed-periodic / random / state-novelty / the current learned controller / a true-benefit controller, at identical attention rate) against `outputs/hz0g_g1_gdn2_fix_301m/native_metal_checkpoint_best_full_holdout`.

## Fair baselines (`hz0c_c4_fair_baselines.py`), mean recall across 8 real scenarios, matched rate=0.150

| Policy | Mean recall | Rate |
| --- | --- | --- |
| no_anchor | 0.000 | 0.000 |
| fixed_periodic | 0.099 | 0.150 |
| random_matched | 0.164 | 0.150 |
| **state_novelty** (real, causal) | **0.229** | 0.150 |
| **equal_compute_transformer** (trained) | **0.270** | 0.150 |
| token_loss (offline teacher, not deployable) | 0.766 | 0.150 |
| oracle | 1.000 | 0.025-0.075 |
| full_attention | 1.000 | 1.000 (max cost) |

Triggered attention beats naive fixed/random baselines at matched cost (0.229/0.270 vs 0.099/0.164) -- the mechanism earns its keep over the null baselines. Real, honest gap to the offline oracle-ish teacher (0.766): the deployable causal signal only captures ~30% of what's visible if you could see the true future loss. Per-scenario variance is large (individual scenario recalls for state_novelty range 0.062-0.812) -- the aggregate is not representative of any single scenario.

## Real primary-metric result (`hz0c_c9_end_to_end_report.py`) -- delta-loss cost, not just recall

The plan explicitly warns recall alone isn't the right metric ("the C7 lesson: an interesting-looking anchor is not necessarily a useful one"). This report measures the real thing: downstream held-out LM-loss cost of missing a should-trigger position.

- **Mean missed-trigger loss cost across 8 scenarios: +0.00903 nats.** Real but small on average; per-scenario range -0.019 to +0.052 nats (2 of 8 scenarios show ~zero/negative cost, likely noise; one scenario shows a real, larger +0.052 nat cost).
- **Latency: no meaningful difference.** no_anchor 0.0547ms/token, fixed-15%-trigger 0.0512ms/token, full_attention 0.0515ms/token -- essentially identical at this model/hardware scale. Attention computation isn't the bottleneck (backbone forward dominates), so triggering fewer positions doesn't show up as a measured wall-clock win here. This is a real, honest finding that undercuts part of triggered-attention's cost-savings motivation, at least at 301M-param scale on this hardware -- it may matter more at larger scale or different hardware, not verified either way.
- Quality (recall at matched rate=0.15): ranges 0.31-0.72 across scenarios, consistent with the fair-baselines run's per-scenario picture.

## State-novelty score's real discrimination vs. calibration gap (`hz0c_c2_natural_novelty_validation.py`)

The underlying signal reliably discriminates real novel positions on real data: normalized delta +1.46 to +1.50 nats between novelty and steady-state positions, 96.9% of examples correctly ranked above the steady-state mean. But the actual deployed rate-bounded threshold only captured 53-56% of true novelty positions and, at seq=32, undershot its own 15% target rate (achieved 12.5%) -- see the bug fix below.

## Learned controllers

- **True-benefit controller** (`hz0c_c7_true_benefit_controller.py`): recall 0.373 against its own consistent ground truth (the fair comparison). 0.221 against the old hand-labeled ground truth definition -- explicitly not comparable, different target, reported only for context.
- **RL trigger controller** (`hz0c_c7_rl_trigger_controller.py`): recall 0.497 at rate=0.150, vs. its own offline teacher's 0.766 at the same rate -- retains ~65% of the teacher's recall in a real, causal, deployable form.

## Real bug found and fixed: `rate_bounded_threshold` systematically undershoots its target

`reference/hz0c_surprise_trigger.py::rate_bounded_threshold` computed its quantile index via `int((1-target_rate) * seq)` (floor truncation), which systematically under-shoots the requested trigger rate -- confirmed directly from this session's run: at seq=32 (the natural-novelty validation's sequence length), target_rate=0.15, it achieved exactly 4/32=0.125, a real ~17% relative miss, not noise (the function's own docstring promises "close to target_rate BY CONSTRUCTION"). Worse at shorter sequences by construction.

Fixed to directly target the desired triggered COUNT (`round(target_rate * seq)`) rather than floor-truncating the quantile index -- the mathematically correct way to hit a rate target on a discrete set. Verified: same seq=32/target=0.15 case now achieves 0.15625 (vs. the true 0.15, within one discrete position). All 4 real callers (`random_trigger`, `hz0c_c2_surprise_validation.py`, `hz0c_c3_trigger_simulator.py`, `hz0c_c6_conditional_attention_eval.py`) inherit the fix automatically; existing tests use loose rate tolerances (not exact-value assertions), so nothing broke. New regression test added (`test_rate_bounded_threshold_hits_target_at_short_sequence_lengths`).

## What G3 establishes

- The corrected backbone's triggered-attention mechanism still beats naive baselines at matched cost -- the core mechanism transfers.
- The real downstream loss cost of missing triggers is small on average (+0.009 nats) but genuinely scenario-dependent, not uniform.
- Latency doesn't meaningfully differ between full attention and 15%-triggered attention at this scale/hardware -- a real caveat on the cost-savings case, not evaluated at larger scale.
- A real, previously-undiagnosed rate-calibration bug in the shared trigger-thresholding utility, found via direct measurement and fixed with verification, not assumed correct.

## What G3 does not establish

- Whether the state-novelty/RL-controller gap to the offline teacher (30-65% retention) is acceptable for HZ-0G's integration decision -- that's a downstream call, not resolved here.
- Cost/latency behavior at larger model scale or different hardware -- only measured on this Mac at 301M params.
