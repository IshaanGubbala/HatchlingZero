# HZ-0G G5: real Dense vs. MoE vs. domain-adapter decision, on the full A+B+C+D+(E) integration

Date: 2026-08-08. Real result on `outputs/hz0g_g1_gdn2_fix_301m/native_metal_checkpoint_best_full_holdout` (301M, `gdn2_fix`, 100M tokens, full-holdout val loss 2.4330). Matches `reference/hz0e_e8_curriculum.py`'s own established step counts (balanced=50, mixed=50, imbalanced=50, warm_start=40) -- comparable rigor, not a shortcut.

## Prerequisite (already built, see prior commits)

`reference/hz0g_g5_full_integration.py` composes A+B+C+D+E for the first time -- every prior HZ-0E MoE result (E4, E6, E8, and this session's earlier work) was measured on A+E alone, backbone plus MoE, with B/C/D never in the loop. `reference/hz0g_g5_curriculum.py` adapts E8's real curriculum (`run_joint_multilayer_curriculum`, `run_joint_multilayer_dense_baseline`) to route every forward pass through the full integration instead. B/C/D are held fixed (fresh untrained memory controller, standard 15%-rate trigger, inactive fast weights) across all three arms -- E is the only real variable, matching the plan's own "no retraining permanent weights" constraint.

## Result

| Arm | General-prose val (lower=better) | Per-domain mean (lower=better) |
| --- | --- | --- |
| HZ-Dense (no E at all) | **2.892651** | -- |
| HZ-MoE (trained curriculum) | 2.973255 | **2.735514** |
| Dense + domain adapter (trained curriculum) | 3.018475 | 2.788533 |

Per-domain detail (held-out, disjoint from training):

| Domain | HZ-MoE | Dense+adapter |
| --- | --- | --- |
| prose | 3.258127 | 3.352096 |
| code | 3.127036 | 3.190647 |
| math | 4.237869 | 4.213054 |
| json | 0.529748 | 0.611284 |
| tools | 2.524792 | 2.575583 |

## Two real findings, both against expectation set by the isolated (A+E-only) prior work

1. **HZ-MoE beats Dense+adapter on both axes here** (general-prose 2.9733 vs 3.0185; per-domain mean 2.7355 vs 2.7885) -- the opposite of E4's own established finding ("a small trained adapter can beat MoE outright at a fraction of the parameter cost"), which is exactly the "dangerous baseline" the plan named as the thing G5 needs to check. In the full B+C+D-integrated pipeline, that reversal does not hold at this checkpoint/config -- a real, measured result, not assumed to transfer from the isolated case.

2. **HZ-Dense (no trained FFN change at all) still wins general-prose robustness over both trained arms** (2.8927 vs 2.9733 and 3.0185) -- consistent with E10's original "specialization costs generality" finding, now confirmed to hold inside the full integrated pipeline too, not just the isolated A+E case.

## Honest scope and caveats

- Single seed (0) for B/C/D's fixed setup and the training curriculum itself -- G2-G4's own multi-seed findings (real seed-to-seed variance, including outright collapse in some HZ-0B configurations) make single-seed G5 numbers a real but not yet seed-robust result. A real next step, not done here.
- B/C/D are fixed at a fresh, untrained state for this comparison -- this isolates E's contribution cleanly (matching the plan's design) but does not test what happens if B/C/D were ALSO actively engaged with real content during MoE training (e.g., real memory writes happening concurrently). That's a different, harder question the plan doesn't ask G5 to answer.
- Matches E8's step counts for comparability, not because these are known-optimal for the integrated pipeline specifically -- untested whether more steps changes either ranking.

## Verdict, per the plan's own promotion rule

The plan states: "If MoE does not beat the adapter baseline by enough to justify its real routing/latency/parameter complexity... E stays a research/deployment OPTION, not the default architecture." Here, MoE *does* beat the adapter baseline, on both axes, in the full integration -- a real point in favor of promoting E, though single-seed and not yet checked against the "enough to justify complexity" bar explicitly (that's a judgment call for whoever makes the final HZ-1 architecture decision, not resolved unilaterally here). HZ-Dense's own win on general-prose robustness over both trained arms is the honest complication: adding E (or the adapter) buys real per-domain specialization at a real, measured cost to general quality -- the same tradeoff E10 already established, still present here.
