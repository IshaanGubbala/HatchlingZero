# HZ-0B B11: Real-Model Distractor Immunity

Date: 2026-08-01. Closes an honest gap named repeatedly today
(`docs/restart/hz0b_b11_stage5_baseline_results.md`,
`docs/restart/hz0b_b8_stage5_results.md`): B8 Stage 5's own scenarios
are pure-simulator (oracle key/value written directly into `MemoryState`,
no LM, no learned write timing). This tests the REAL, LEARNED write
mechanism (the fixed `lambda_sparse=0.1` controller) against 4
in-context distractors -- `DISTRACTOR_MARKER` + random-value pairs
actually appearing as tokens, competing for the same limited slots the
real fact needs, not injected as raw array writes.

`scripts/hz0b_b11_real_model_distractor_immunity.py`. Same 2-way
fact-discrimination task and scale as
`docs/restart/hz0b_b11_evaluation_results.md`'s culminating test (5
seeds, steps=1000, lr=0.15, train_count=64, held_out_count=64,
`lambda_sparse=0.1`), plus 4 distractor marker+value pairs scattered
through the middle section of every example.

## Result

| Configuration | mean | std | range |
| --- | --- | --- | --- |
| No distractors (reference, `docs/restart/hz0b_b11_evaluation_results.md`) | 0.830 | 0.173 | 0.328-0.953 |
| **With 4 in-context distractors** | **0.769** | 0.193 | 0.391-0.922 |

**Distractor-immunity holds up well.** A real but modest cost
(-0.061 mean), well within the noise this investigation has already
seen from seed-to-seed variance alone. 4 of 5 seeds converge cleanly
(0.812-0.922, train loss 0.10-0.17 at step 999) -- close to their
no-distractor counterparts.

## Seed 557 fails again -- a third independent confirmation

Seed 557 scored 0.391, train loss plateaued at 1.71 (never approaching
the ~0.10-0.17 the other seeds reach). This is the SAME seed that failed
in both the original 5-seed run (0.328) and the 10-seed extension
(0.328 again, identically) of the no-distractor culminating test
(`docs/restart/hz0b_b11_evaluation_results.md`). Failing a THIRD time,
on a meaningfully different task variant (with distractors now
competing for slots), is strong evidence this is a genuine,
reproducible property of seed 557's specific random initialization for
this controller architecture -- not task-specific noise, not
distractor-specific. Real, identified, not yet root-caused; a natural
next target for the `target_write_rate` fix attempt
(`docs/restart/hz0b_b11_evaluation_results.md`'s newest section, if
run) or further investigation.

## What this adds to B11's real coverage

The 15th named-task gap list shrinks by one real-model version of a
Stage 5-adjacent scenario (distractor immunity), now tested against the
actual learned mechanism instead of only the pure simulator. Still
open: real-model versions of the other Stage 5 scenarios (contradictory
info, near-identical keys, capacity pressure), which would each need
their own task construction the same way this one did.
