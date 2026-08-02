# HZ-0B B11: Long-Conversation Consistency

Date: 2026-08-01. One more of B11's 16 named eval tasks. Same 2-way
fact-discrimination task as `scripts/hz0b_b11_baseline_comparison.py`
(the already-validated culminating test, mean 0.819-0.830 at
MIDDLE_LEN=24), but with the gap between the written fact and the
read trigger increased 8x, from 24 to 200 tokens (PROMPT_LEN=210) --
still comfortably inside the 256-token sequence length the HZ-0A
backbone was actually trained at, so out-of-distribution extrapolation
is not a confound here.

`scripts/hz0b_b11_long_conversation_consistency.py`. Real frozen
HZ-0A checkpoint, `precomputed_hidden` caching, the validated
`lambda_sparse=0.1` + `target_write_rate=0.1` config. 5 seeds,
steps=1000, lr=0.15, train_count=64, held_out_count=64.

## Result: the advantage HOLDS and WIDENS over a much longer gap

| Configuration | mean | std | range |
| --- | --- | --- | --- |
| Floor | 0.000 | -- | -- |
| Equal-param adapter | 0.409 | 0.006 | 0.406-0.422 |
| HZ-0B memory | **0.775** | 0.077 | 0.625-0.828 |

Memory beats the adapter by +0.366 -- a wider margin than the
short-gap baseline's own +0.309-ish advantage (adapter ~0.51, memory
~0.819-0.830 at MIDDLE_LEN=24). The adapter's own accuracy actually
DROPS as the gap grows (from ~0.51 at 24 tokens to 0.409 at 200
tokens) -- consistent with a no-memory model increasingly struggling
to carry a single fact across more intervening context using only the
frozen backbone's own attention. Memory's accuracy also drops
somewhat (0.819 -> 0.775) but far less, and stays well clear of the
adapter and the chance floor at every seed (worst case 0.625, still
+0.216 over the adapter's best seed).

Seed 557 (the seed that failed catastrophically, 0.328, on the
original short-gap task before the `target_write_rate` fix) scores
0.625 here -- the relative laggard among these 5 seeds again, but not
a collapse, and clearly above both the adapter and chance.

## What this adds to B11's real coverage

One more of the 16 named tasks moves from 0% to done, with a clean
positive result: the exit gate ("cannot be explained only by more
parameters or more context") holds up, and if anything strengthens,
over an 8x longer real conversation gap -- a genuinely different
finding from code-symbol tracking's negative result on the same day,
underscoring that B11's exit-gate support is real but task-dependent,
not a blanket "memory always wins." Remaining scope: 2 more named
tasks (multi-hop, tool-result reuse), and real-model versions of 3
more Stage 5 scenarios (contradictory info, near-identical keys,
capacity pressure).
