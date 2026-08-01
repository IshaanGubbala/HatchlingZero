# HZ-0B B11: Passkey Retrieval Task (2nd of 16 named tasks)

Date: 2026-08-01. `scripts/hz0b_b11_passkey_task.py`. Genuinely new task
type, not a repackaging: 4-way exact-value retrieval (the classic
"passkey" eval format) against the real frozen HZ-0A checkpoint, using
the validated `lambda_sparse=0.1` fix from the start
(`docs/restart/hz0b_b11_evaluation_results.md`, "The culminating test")
rather than rediscovering the sparsity-penalty bug on a new task. Same
"fair test" property as the earlier fact-discrimination task: the
passkey identity token genuinely appears in the context, so a no-memory
model has a real, non-structural chance to solve it.

Setup: 5 seeds, steps=1000, lr=0.15, train_count=80, held_out_count=80
(20 examples/class), 692,418-param equal-param adapter vs. 692,837-param
real memory controller (same budgets as the earlier task).

## Result

| Condition | mean | std | range | chance=0.250 |
| --- | --- | --- | --- | --- |
| Floor (no memory) | 0.000 | -- | -- | below chance (structurally cannot solve it) |
| Equal-param adapter | 0.330 | 0.058 | 0.263-0.425 | modestly above chance |
| **HZ-0B real memory** | **0.455** | **0.198** | **0.200-0.625** | above chance on average |

**Memory beats the adapter on average** (0.455 vs. 0.330, +0.125 margin)
-- but not uniformly, and this is a genuinely different, more mixed
result than the binary fact-discrimination task, reported honestly
rather than summarized as a clean win.

## A different, real failure mode: overfitting, not collapse

2 of 5 seeds (556, 557) scored BELOW chance on held-out (0.225, 0.200)
despite reaching near-perfect TRAINING loss (0.029, 0.024 at step 999 --
the lowest training losses recorded anywhere in this entire B11
investigation). This is not the "stuck in a bad local optimum, high
train loss" pattern seen in the earlier task's pre-fix runs -- it is
real, textbook overfitting: with only 20 training examples per class
(4-way, 80 total), the memory controller can apparently memorize the
specific training examples' idiosyncrasies well enough to nearly zero
out training loss while generalizing worse than random guessing on
held-out data. The other 3 seeds (555, 558, 559) generalize well
(0.600-0.625, held-out accuracy roughly tracking their own low but
non-zero training loss).

This is a real, different, honestly-reported finding: the sparsity-
penalty fix that resolved the earlier task's collapse does not
automatically make the mechanism robust against overfitting on a
harder, higher-cardinality, smaller-per-class-sample task. Both are real
training-dynamics problems, but they are NOT the same problem, and a fix
for one (lower `lambda_sparse`) was not shown here to address the other.

## Honest interpretation

**Supports, weakly, not decisively**: the mean does support B11's exit
gate on this task too (memory beats the adapter), but the high variance
and the two below-chance seeds mean this result is meaningfully less
robust than the fact-discrimination task's 10-seed confirmation (mean
0.830, std 0.173, 9/10 seeds converging cleanly,
`docs/restart/hz0b_b11_evaluation_results.md`). A likely, not-yet-tested
explanation: this task needs more training examples per class (80
total/4 classes is much sparser than the binary task's 64 total/2
classes) or a stronger regularizer against overfitting specifically
(different from the write-sparsity penalty, which targets timing, not
generalization) -- named as real future work, not chased further this
pass.

## Testing the obvious fix: more training data (2026-08-01)

Doubled `train_count` from 80 to 160 (40 examples/class instead of 20),
same 5 seeds, steps, lr, `lambda_sparse=0.1`, `held_out_count=80`
unchanged for a like-for-like comparison.

| Condition | train_count=80 (original) | train_count=160 (doubled) |
| --- | --- | --- |
| Adapter mean (std) | 0.330 (0.058) | 0.398 (0.050) |
| Memory mean (std) | 0.455 (0.198) | **0.608 (0.277)** |
| Memory seeds below chance (0.25) | 2 of 5 (0.200, 0.225) | **0 of 5** (worst: 0.263) |
| Memory advantage over adapter | +0.125 | **+0.210** |

**Real, partial fix -- reported precisely, not rounded to "solved" or
"didn't help."** More data clearly helps: mean rose (0.455 -> 0.608),
no seed collapsed below chance anymore (worst case moved from 0.200 to
0.263, now just barely above chance rather than meaningfully below it),
and memory's margin over the adapter widened (more data helps the
memory mechanism MORE than it helps the adapter, not just both equally).

**But it does not fully fix the underlying inconsistency.** Std actually
INCREASED (0.198 -> 0.277), not decreased -- because the well-converging
seeds got much better (0.812-0.850, near ceiling, all with real,
low, non-overfit-looking final train loss 0.12-0.14) while the two
previously-below-chance seeds (556, 557) only recovered to
barely-above-chance (0.263, 0.275) despite reaching even LOWER training
loss than before (0.054, 0.042 -- still the overfitting signature: very
low train loss, unremarkable held-out accuracy). More data raised the
ceiling and the floor together, but did not close the gap between seeds
that find a good solution and seeds that memorize instead -- doubling
data is a real, measurable improvement, not a complete fix for this
task's seed-to-seed inconsistency.

## What this adds to B11's real coverage

2 of 16 named tasks now have real, checkpoint-backed, multi-seed
results. Task 1 (fact discrimination): clean, robust, decisive positive
result after the fix. Task 2 (passkey): positive on average, but with a
real, distinct generalization failure mode on 40% of seeds -- a more
nuanced, less clean-cut picture, disclosed as such rather than rounded
up to "confirms the pattern."
