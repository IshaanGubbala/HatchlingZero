# HZ-0B B9 Stage 1: Unfreeze Last HZ-0A Block -- Results

Date: 2026-07-30. First B9 work: per the plan's own staging ("1. unfreeze
only memory-adjacent projections"), unfroze HZ-0A's LAST block (index 30
of 31, 9,449,472 params -- the block immediately upstream of where B6/B7's
memory injection happens) and fine-tuned it jointly with the B7 write
controller, against the real, previously-frozen checkpoint. Same task as
B7 (`scripts/hz0b_b9_stage1_finetune_probe.py`, reusing
`scripts/hz0b_b7_real_integration_probe.py`'s exact prompts/labels),
`confidence_scaled=True`.

## Correction (2026-07-30, same date): the original conclusion below was wrong

The first pass of this doc compared this experiment's train loss (22.04
at step 999, joint from step 0) against a number remembered as "B7's own
comparable confidence-scaled run, ~9.4 at step 2499" and concluded the
unfrozen block was making optimization *harder*. That comparison number
was wrong -- 9.4 was from a **different task** (B6's simpler read-only
oracle-recall probe), not B7's write+read task. B7's own write+read task,
at the same hyperparameters (controller_lr=0.4, confidence_scaled=True),
reaches **23.08** at step 2499, not 9.4 -- verified directly by re-running
the controller-only warmup phase and confirming it reproduces B7's own
trajectory bit-for-bit (loss 23.07860 at step 2499, matching to 5 decimal
places). The "moving target" framing was built on an apples-to-oranges
comparison. Retracted below; a real, controlled comparison replaces it.

## The real, controlled comparison

Added `--warmup-steps` (train the controller alone first, block frozen --
identical to B7's own recipe -- for N steps, THEN unfreeze the block and
continue for M more). Ran two arms at the **same total step count**
(3500), differing only in whether the last 1000 steps had the block
frozen or not:

| Arm | Total steps | Block unfrozen for | Final train loss | Held-out rank (write-then-read) |
| --- | --- | --- | --- | --- |
| Control (frozen throughout) | 3500 | 0 steps | 22.37608 | 1763.0 / 24576 |
| **Curriculum (warmup then joint)** | 3500 | last 1000 steps | **22.18753** | **1375.6 / 24576** |

Lower loss, better (lower) rank for the arm that unfroze the block --
**a real, modest, correctly-directioned benefit from unfreezing**, the
opposite of the original (wrong-baseline) conclusion. General quality
stayed preserved in both arms (delta ~0.00% either way), and memory
interference stayed exactly 0 (the confidence-scaling fix holds
regardless).

## An honest complication: this doesn't fully resolve either

The very first attempt (joint-from-step-0, no warmup, only 1000 TOTAL
steps) reached loss 22.04042 and rank 1184.5 -- better than BOTH 3500-step
arms above. That is not an apples-to-apples comparison (different total
step count, different optimization path), so it does not mean "no
warmup is better" -- but it does mean the picture is noisier than a single
controlled pair can settle. All of these are **single runs, no repeated
seeds** -- exactly the discipline this session's GDN-3 investigation
learned the hard way is necessary before trusting any one comparison
(`docs/restart/hz0a_gdn3_associative_recall_results.md`'s multi-seed
section). These numbers are reported as real, single data points, not as
a settled ranking of curriculum vs. no-curriculum vs. step count -- that
would need multiple seeds per configuration, not done here.

## Honest read against B9's exit gate

B9's exit gate: *"Memory improvements survive limited fine-tuning without
destroying HZ-0A quality."*

- **"Without destroying HZ-0A quality"**: satisfied cleanly across every
  configuration tried (control, curriculum, and the original joint-from-
  scratch run) -- general val loss never moved by more than ~0.05%, and
  memory interference stayed exactly 0 throughout.
- **"Memory improvements survive"**: the controlled pair shows a real,
  positive direction -- unfreezing the last block, after the controller
  has had a chance to partially converge first, measurably improves the
  memory-task rank over continuing frozen-only training for the same
  steps. Not yet a strong result (rank 1375.6 is still far worse than
  B7's own best frozen-backbone number of 179.4-325.0 at full
  convergence), and not yet confirmed across multiple seeds.

## Real next steps (not run in this pass)

- **Multiple seeds** per configuration -- the single-biggest gap given
  this session's own GDN-3 lesson about trusting single runs.
  Or seed the memory-key/prompt-generation identically, in each run.
- **More total steps** -- none of these runs have converged the way B7's
  own longer (4000-step) frozen-only run did; B9 needs at least that much
  budget before a real ceiling comparison is possible.
- **A cleaner ablation**: warmup-length sweep (is 2500 the right amount of
  controller-only training before unfreezing, or would more/less change
  the result), and separately, whether block_lr itself (currently 1e-5,
  never tuned) is well-chosen.
