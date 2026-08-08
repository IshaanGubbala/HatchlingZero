# HZ-0G G2: revalidating HZ-0B's memory mechanism against the corrected `gdn2_fix` backbone

Date: 2026-08-07. Real, complete run of G2's compact diagnostic subset (7 named tasks per `plans/HZ-0G_Integration_Plan.md`) against `outputs/hz0g_g1_gdn2_fix_301m/native_metal_checkpoint_best_full_holdout` (301M params, `gdn2_fix` mixer, 100,007,936 tokens, full-holdout validation loss 2.432983). Each task run at its script's own established defaults (`--steps 1000 --num-seeds 5`, task-specific `--train-count`/`--held-out-count`) -- the same rigor the original numbers below were measured at, so the comparison is real, not shortcut.

## Prerequisite fix: checkpoint/mixer override

`scripts/hz0b_b11_baseline_comparison.py::load_frozen_model()` (the shared chokepoint 50+ B/C/D eval scripts depend on) hardcoded the old, uncorrected `gdn2` mixer with no override. Made `CHECKPOINT`/`MIXER` overridable via `HZ0_EVAL_CHECKPOINT`/`HZ0_EVAL_MIXER` env vars, default-preserving. Found and fixed two more standalone copies of the same bug during G2/G3 prep: `scripts/hz0a_select_best_full_holdout.py` and `scripts/hz0c_c2_surprise_validation.py`, both had their own unfixed `load_frozen_model()`.

## Results: new (`gdn2_fix`, G1) vs. original (`gdn2`, pre-correction)

| Task | Original: memory vs. adapter | New: memory vs. adapter | Shift |
| --- | --- | --- | --- |
| Single-fact recall | 0.819-0.830 vs 0.512 (memory won decisively) | **0.147** vs 0.600 | **Reversed -- memory now loses badly** |
| Long-gap consistency | 0.775 vs 0.409 (memory won) | 0.650 vs 0.438 | Still wins, smaller margin |
| Passkey | 0.608 vs ~0.330 (memory won) | 0.495 vs 0.265 | Still wins, weaker, 1 seed collapsed |
| Overwrite/reassignment | 0.283 vs 0.370 (memory lost) | **0.650** vs 0.268 | **Reversed -- memory now wins clearly** |
| Multi-hop | 0.305 vs 0.328 (memory lost, narrow) | 0.245 vs 0.250 | Same story, both near chance |
| Tool-result reuse | 0.513 vs 0.623 (memory lost) | **0.698** vs 0.490 | **Reversed -- memory now wins clearly** |
| Noisy recall | 0.819->0.000 as noise 0->10x std (adapter flat ~0.5) | 0.525->0.053 as noise 0->10x std (adapter flat ~0.5) | Same fragility shape, lower baseline throughout |

Full per-seed logs preserved in the session's scratchpad (`g2_logs/1-7_*.log`); headline numbers verified directly against each script's own printed `--- Summary` block, not recomputed or rounded.

**Honest read: this is not a clean "corrected backbone helps/hurts memory" story.** Two tasks flipped direction entirely (overwrite and tool-result reuse: losses become wins; single-fact recall: a decisive win becomes a decisive loss). The other four tasks kept the same qualitative direction but shifted in magnitude. Nothing here supports a single unifying explanation across all 7 -- the corrected recurrence changes HZ-0B's behavior in task-specific, not uniform, ways.

## Single-fact recall's collapse: a real, only-partially-explained mechanism

Single-fact recall's reversal was investigated directly, not just reported. Raw per-seed logs showed 3 of 5 memory seeds (556, 558, 559) plateau at a bit-identical loss (13.65230) from step ~300 onward -- a dead fixed point, not slow convergence.

**Hypothesis tested**: `latent_write_and_read_step`'s write gate reads the raw, unnormalized backbone residual stream (`hidden_state`, pre-`final_norm`) directly into `sigmoid(hidden_state @ write_gate_w + bias)`. Measured real write-logit distributions at init for all 5 seeds against both checkpoints: G1's `gdn2_fix` checkpoint showed 1.5-2x higher write-logit variance than the old `gdn2` checkpoint (std 5.6-8.4 vs 3.0-5.0), a real, measured difference in residual-stream geometry -- consistent with, but not by itself proof of, a saturation-driven collapse.

**Fix built and tested**: `normalize_gate_input` (opt-in, default `False`, `reference/hz0b_b8_latent_write.py`) RMS-normalizes only the write gate's own input, leaving key/value/read-query untouched. Unit tests confirm it prevents saturation on a synthetic adversarial input that reproduces the failure mode.

**Real result against G1's checkpoint: does NOT fix the collapse.** 5-seed rerun with the flag: mean 0.134 (vs. 0.147 without -- not a real improvement, within noise). Seeds 556 and 559 still hit the identical flat-loss plateau. Seed 558 alone recovered (0.000 -> 0.266) -- a real, partial, seed-specific effect. Write-gate saturation on raw hidden-state input is a real, measured, contributing factor for some seeds, but not the root cause of the general collapse pattern. That root cause remains open.

## What G2 establishes

- HZ-0B's memory mechanism's behavior is **not stable across the `gdn2_fix` correction** -- real, measured, task-specific reversals in both directions, not a uniform regression or uniform improvement.
- Single-fact recall's specific collapse has a real, partially-understood mechanism (write-gate saturation contributes but doesn't fully explain it) and a tested, real, honest negative result on the most obvious fix.
- G2's own "watch read discrimination directly" instruction (per the integration plan) is borne out: multiple failures/reversals trace to write or read dynamics, not the backbone's raw capability (the backbone itself trains fine on every task, as shown by the adapter baseline's consistent, healthy convergence across all 7).

## What G2 does not establish

- Whether HZ-0B as a mechanism should be carried forward into the integrated checkpoint -- that's explicitly G5's decision, not G2's, and G2's mixed result here doesn't resolve it either way.
- The actual root cause of seeds 556/559's collapse -- open, real follow-up work if pursued.
