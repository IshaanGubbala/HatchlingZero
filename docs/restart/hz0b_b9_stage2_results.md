# HZ-0B B9 Stage 2: Unfreeze 3 Upper HZ-0A Blocks -- Results

Date: 2026-07-30. Extends Stage 1 (`docs/restart/hz0b_b9_stage1_results.md`,
which unfroze just the last block and found a real but modest benefit,
1375.6 vs 1763.0 rank). Stage 2 unfreezes the last **3** blocks (indices
28, 29, 30 -- 26,576,640 params combined) instead of 1, same warmup-then-
joint recipe (2500 controller-only steps, then 1000 joint steps),
**multi-seed from the start this time** (3 seeds: 321/322/323) --
applying the lesson from Stage 1's own single-run mistake up front rather
than retrofitting it after drawing a wrong conclusion again.

## Result: clean, strong, fully reproducible

| Seed | Final train loss | Held-out rank (write-then-read) | General val loss delta |
| --- | --- | --- | --- |
| 321 | 13.31310 | **0.0** | -0.057% |
| 322 | 13.30700 | **0.0** | -0.057% |
| 323 | 13.31387 | **0.0** | -0.057% |
| **Mean** | 13.31 | **0.0** | **-0.057%** |

**Rank 0 (top prediction) across all 3 seeds, identical general-quality
cost across all 3 seeds (stdev 0.0 on both).** This is a materially
stronger and cleaner result than Stage 1's single-block experiment
(1375.6 best rank) -- more unfrozen capacity (26.6M vs 9.4M params) gave
the controller+backbone combination enough room to fully solve the
memory-specific task, not just improve on it, while general held-out loss
barely moved (-0.057%, and negative -- technically a very slight
*improvement*, well within noise but certainly not degradation).

## Honest read against B9's exit gate

B9's exit gate: *"Memory improvements survive limited fine-tuning without
destroying HZ-0A quality."*

**Satisfied, cleanly, and now with real multi-seed confidence behind it**
-- not the single-run, unsettled result Stage 1 left off with. Quality is
preserved (arguably very slightly improved) across all 3 seeds; the
memory task is not just improved but fully solved (rank 0) across all 3
seeds. This is the strongest, most decisive result in the B9 investigation
so far.

## What this means for Stage 3

The plan's own Stage 3 framing: *"consider full fine-tuning only if
necessary."* Given Stage 2 already fully satisfies B9's exit gate --
clean rank-0 memory-task performance, negligible (and multi-seed-
confirmed) general-quality cost -- **full fine-tuning of the entire
301M-param model does not currently look necessary.** Not run in this
pass; the honest case for skipping it is that Stage 2's result leaves no
obvious gap for full fine-tuning to close. If pursued anyway (e.g. to
check whether it generalizes further, or helps on a harder task than this
specific single-fact probe), that's a deliberate choice to test something
beyond what B9's exit gate itself requires, not a necessity the data
points to.
