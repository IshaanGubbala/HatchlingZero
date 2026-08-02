# HZ-0B B11: Code-Symbol Tracking

Date: 2026-08-01. One more of B11's 16 named eval tasks. Distinct
SHAPE from the earlier fact-recall tasks (baseline_comparison,
passkey): the same symbol is reassigned 3 times (3 sequential writes
to one key, each separated by 8 tokens of padding so proximity-to-
read-trigger can't trivially solve it), and the correct answer is
whichever value was assigned LAST -- a real test of "overwrite
tracking" (discard stale writes to the same key), not "recall a
single written fact". Mirrors B8 Stage 4's "variable reassignment"
item type, real-model this time.

`scripts/hz0b_b11_code_symbol_tracking.py`. Real frozen HZ-0A
checkpoint, `precomputed_hidden` caching, the validated
`lambda_sparse=0.1` + `target_write_rate=0.1` config from
`docs/restart/hz0b_b11_evaluation_results.md` (not rediscovering the
sparsity bug on a new task). 5 seeds, steps=1000, lr=0.15,
train_count=80, held_out_count=80, 4-way final value (chance=0.250).

## Result: a real, disclosed negative -- memory UNDERPERFORMS the adapter here

| Configuration | mean | std | range |
| --- | --- | --- | --- |
| Floor (chance) | 0.000 | -- | -- |
| Equal-param adapter | **0.370** | 0.056 | 0.325-0.450 |
| HZ-0B memory (lambda_sparse=0.1, target_write_rate=0.1) | **0.283** | 0.038 | 0.225-0.325 |

Reported plainly, not rounded up: on this task, the memory mechanism
is WORSE than the equal-parameter no-memory adapter (-0.087 mean),
and memory's own mean (0.283) is barely above the 0.250 chance floor
while the adapter's (0.370) is more clearly above it. This does not
match the clean wins on the single-fact tasks
(`hz0b_b11_evaluation_results.md`: mean 0.819-0.830; passkey task
after doubling data: mean 0.608).

**A clear overfitting signature, not a training failure**: every
memory seed reaches a dramatically lower final training loss
(0.148-0.277) than every adapter seed (1.096-1.273) -- the memory
mechanism fits the training set far better, yet generalizes worse.
This is the same overfitting pattern already seen once before on this
session's harder/sparser 4-way passkey task (2 of 5 seeds below
chance before doubling training data), now showing up MORE severely
on a task that additionally requires overwrite/update semantics
rather than a single clean write. A plausible mechanism (not verified
here, named as a real open question): the controller may be writing
all 3 reassignments into memory (or blending across them) rather than
cleanly overwriting/discarding the first 2, and then relying on
position/recency artifacts specific to the 80 training examples to
guess the right one, rather than learning a true "most recent write
wins" rule that generalizes to unseen reassignment orderings.

## Tested the passkey task's own fix (2026-08-01, same day) -- it does NOT transfer, and makes this task WORSE

The passkey task's overfitting was fixed by doubling training data
(80->160). Tried the identical fix here:

| Configuration (train_count=160) | mean | std | range |
| --- | --- | --- | --- |
| Equal-param adapter | 0.360 | 0.053 | 0.263-0.412 |
| HZ-0B memory | **0.233** | 0.022 | 0.200-0.263 |

**This is not an improvement -- it is a further regression.** Memory's
mean dropped from 0.283 (train_count=80) to 0.233 (train_count=160),
now solidly BELOW the 0.250 chance floor on average, with every seed
in a tight 0.200-0.263 band. The adapter is essentially unchanged
(0.370 -> 0.360). Training loss for memory also changed character:
with more data it converges much more slowly and to a HIGHER final
loss (mostly 1.0-1.5, vs. 0.148-0.277 at train_count=80) -- the
opposite of overfitting-with-low-train-loss seen before; this looks
more like genuine difficulty fitting the harder, larger training set
at all within 1000 steps, not a data-starvation problem the passkey
fix addressed.

**Honest conclusion**: the "more data" fix is task-specific, not a
general remedy for memory underperforming on B11 tasks. It helped the
passkey task (single clean write, sparse 4-way retrieval) and hurt
this one (3 sequential overwrites to the same key). This is real
evidence the overwrite/reassignment-tracking failure mode is
mechanistically different from the passkey task's sparse-retrieval
overfitting, not just "needs more examples" -- consistent with the
architectural hypothesis named above (repeated writes to the same key
may not be cleanly overwriting). Not yet tried: an explicit
reassignment-clearing mechanism change, or more training steps at the
same data size (ruling out simple undertraining at the larger set).

## What this adds to B11's real coverage

One more of the 16 named tasks moves from 0% to done, with an honest
negative result: the exit gate ("cannot be explained only by more
parameters") is NOT supported on this task at this scale/config --
the opposite of the single-fact tasks' finding. This is exactly the
kind of task-dependent result this investigation has repeatedly found
(the passkey task's original mixed result before the data-doubling
fix is the closest precedent) and is reported as such, not
generalized from the single-fact tasks' success. Not yet tried: more
training data (the fix that helped the passkey task), a target-write-
rate sweep specific to this task, or an explicit "clear on
reassignment" architectural change (unverified, real future work, not
attempted this pass). Remaining scope: 3 more named tasks (multi-hop,
long-conversation consistency, tool-result reuse), and real-model
versions of 3 more Stage 5 scenarios (contradictory info,
near-identical keys, capacity pressure).
