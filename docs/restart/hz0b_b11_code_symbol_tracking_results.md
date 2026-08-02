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
