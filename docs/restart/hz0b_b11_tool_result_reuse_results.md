# HZ-0B B11: Tool-Result Reuse

Date: 2026-08-01. The LAST of B11's 16 named eval tasks -- all 16 now
covered. Deliberately not a re-skin of the recall tasks already
tested: a "tool call" returns an ordinal result (`TOOL_MARKER,
result_i`, 4 possible levels); later a threshold is presented
(`COMPARE_MARKER, threshold_j`); the correct answer is `GREATER` or
`NOT_GREATER` depending on comparing the STORED result against the
threshold. This requires actually reusing the stored value in a
downstream decision, not just retrieving it verbatim.

`scripts/hz0b_b11_tool_result_reuse.py`. Real frozen HZ-0A checkpoint,
`precomputed_hidden` caching, validated `lambda_sparse=0.1` +
`target_write_rate=0.1` config. 5 seeds, steps=1000, lr=0.15,
train_count=80, held_out_count=80. Binary target; held-out
`is_greater` base rate 0.438 (majority-class baseline 0.562).

**A real bug caught and fixed before trusting any result**: the first
attempt used token ids 25000-25051 for the task's special markers/
targets, but `VOCAB_SIZE=24576` -- those ids exceed the model's output
vocabulary entirely, making the two target classes literally
unreachable as predictions (the logits array only has 24,576 output
classes). This produced a flat, meaningless 0.000 across the floor,
adapter, AND memory conditions -- not a real research finding, an
out-of-range bug. Caught by noticing the adapter's training loss was
decreasing normally (0.37-1.2, real learning) while held-out accuracy
was impossibly exactly 0/80, and confirmed by checking the constants
against `VOCAB_SIZE`. Fixed by moving the markers to the
20000-20051 range (with an assert added: `NOT_GREATER_TARGET <
VOCAB_SIZE`) and rerun.

## Result: adapter clearly wins, memory near chance -- a third confirmation of the "reuse" failure pattern

| Configuration | mean | std | range |
| --- | --- | --- | --- |
| Floor | 0.000 | -- | -- |
| Equal-param adapter | **0.623** | 0.051 | 0.562-0.688 |
| HZ-0B memory | 0.513 | 0.042 | 0.463-0.588 |

The adapter clearly beats the majority-class baseline (0.562) and
memory (0.623 vs 0.513, a real -0.110 gap). Memory's mean (0.513) sits
almost exactly at chance/majority-baseline level despite most seeds
reaching a near-zero final training loss (0.008-0.11, one seed as low
as 0.008) -- the same overfitting signature already documented on
code-symbol tracking and multi-hop retrieval: the memory mechanism
fits the training set extremely well but does not generalize the
underlying comparison rule to held-out examples.

## What this adds to B11's real coverage

**All 16 of B11's named eval tasks are now covered** (single-fact
recall, cross-baseline Stage 5 scenarios, passkey retrieval,
throughput/cost, reinforcement/forgetting/serialization,
long-conversation consistency, code-symbol tracking, multi-hop
retrieval, tool-result reuse, plus the earlier B8/B6/B7-derived
groundwork). The exit-gate finding across the tested named tasks is
real but genuinely mixed: clear wins on single-clean-fact tasks
(recall, long-conversation, passkey-after-fix, distractor immunity)
and no advantage or a real negative on tasks requiring comparison,
discrimination-under-distraction, or overwrite tracking (code-symbol
tracking, multi-hop retrieval, tool-result reuse) -- now a THIRD
independent confirmation of the same pattern, strengthening it from
"emerging hypothesis" to a real, repeated finding across 3 of 3 tested
non-recall task shapes. Remaining scope: real-model versions of 3
Stage 5 scenarios (contradictory info, near-identical keys, capacity
pressure) -- currently only tested against the pure B2 simulator, not
the real learned mechanism.
