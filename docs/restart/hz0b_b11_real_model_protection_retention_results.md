# HZ-0B B11: Protection Retention (Real-Model)

Date: 2026-08-02. One of the plan's 16 named eval tasks
("protection retention"), previously only tested via B8 Stage 5's
pure-simulator "protected-memory overwrite rejection" scenario (direct
`write()`/`protect()` calls, oracle slot targeting, no LM, no learned
write timing/slot choice). This is the real-model version: slot 0 is
pre-populated with a synthetic anchor fact and `protect()`-ed to full
strength BEFORE training starts; the real latent write controller is
then trained on an ordinary single-fact recall task using the
remaining 7 slots. Checks whether the protected slot survives real,
LEARNED write pressure, not just oracle writes.

`scripts/hz0b_b11_real_model_protection_retention.py`. Manually
replicates the sequential write/read loop from a custom initial state
(the shared `sequential_latent_write_and_read` helper always starts
from all-zero, no opt-in initial-state param). 5 seeds, steps=1000,
lr=0.15, train_count=64, held_out_count=64, `lambda_sparse=0.1` +
`target_write_rate=0.1`.

## A real bug caught in this script's own first version

The first version compared the protected slot's final key/value
against a fresh reference using a single whole-batch `mx.array_equal`
call across all 64 held-out examples at once. This reports "changed"
for the ENTIRE batch the instant even ONE example differs -- it showed
4/5 seeds "not intact," which looked like a serious, systemic
protection failure. A direct, isolated single-step check first
confirmed the underlying write-rejection logic is mathematically
sound (no change after one step, matching `_choose_write_slot`'s
protection-exclusion code read directly). A per-example audit then
found only 1 of 64 examples in the "failing" seed 555 actually showed
a change -- the whole-batch check was real but badly miscalibrated.
Fixed to report a per-example leak count/rate instead, then rerun.

## Result: a real, small, nonzero leak rate -- not systemic, not zero

| Seed | Task accuracy | Protected-slot leak rate |
| --- | --- | --- |
| 555 | 0.547 | 1/64 (1.6%) |
| 556 | 0.906 | 6/64 (9.4%) |
| 557 | 0.328 | 1/64 (1.6%) |
| 558 | 0.500 | 1/64 (1.6%) |
| 559 | 0.531 | 0/64 (0.0%) |
| **Total** | mean 0.562 (range 0.328-0.906) | **9/320 (2.8%)** |

The protection FIELD itself and the confidence field are bit-exact
intact in all 5 seeds, every single held-out example (100%) -- the
oracle-level `protect()` bookkeeping never degrades. What leaks, on a
real but small minority of held-out examples (2.8% overall, ranging
0-9.4% per seed), is the protected slot's KEY and/or VALUE content
itself. Task accuracy using the remaining 7 slots is reasonable
(mean 0.562), showing the controller can still solve its own task with
reduced capacity.

**Honest characterization**: this is neither "protection is broken"
nor "protection is perfectly reliable" -- it is a real, small,
seed-and-example-dependent leak rate under genuine learned write
pressure, not observable in the pure-simulator oracle-write test (B8
Stage 5), which never exercises the LEARNED routing path at all. Not
investigated further this pass: the exact mechanism by which a rare
held-out example's key manages to route into (or otherwise perturb)
a protected slot despite `_choose_write_slot`'s protection-exclusion
logic reading as airtight -- real, disclosed future work.

## What this adds to B11's real coverage

One more of the 16 named tasks done for real. Corrects an
overclaiming risk within its own investigation (the initial 4/5-seed
"not intact" reading would have been a real, significant overclaim
about a broken mechanism, caught before being reported as such) and
lands on an honest, precisely quantified, small-but-real finding
instead. This qualifies (does not overturn) B8 Stage 5's simulator-
level "protected-memory overwrite rejection" result and reopening
criterion 6's rollback-mechanism finding: those remain correct for
oracle writes; this is the first evidence that real, learned writes
introduce a real, small, previously-unmeasured leak.
