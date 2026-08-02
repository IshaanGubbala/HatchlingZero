# HZ-0B B11: Multi-Hop Retrieval

Date: 2026-08-01. One more of B11's 16 named eval tasks. Genuinely
different shape from the earlier single-key retrieval tasks: a 2-hop
chain. Hop 1 (`ENTITY_MARKER, ENTITY_ID, POINTER_MARKER, pointer_i`)
is a within-window association the frozen backbone's own attention can
plausibly resolve unaided. Hop 2 (`VALUE_MARKER, pointer_i, value_j`,
scattered later in the sequence, padded, alongside 2 DISTRACTOR value
triples with different pointer tokens) requires matching the specific
pointer from hop 1 against the correct triple, not just grabbing the
nearest `VALUE_MARKER` pair.

`scripts/hz0b_b11_multi_hop_retrieval.py`. Real frozen HZ-0A
checkpoint, `precomputed_hidden` caching, the validated
`lambda_sparse=0.1` + `target_write_rate=0.1` config. 5 seeds,
steps=1000, lr=0.15, train_count=80, held_out_count=80, 4-way final
value (chance=0.250).

## Result: another honest negative -- no clear memory advantage

| Configuration | mean | std | range |
| --- | --- | --- | --- |
| Floor | 0.000 | -- | -- |
| Equal-param adapter | 0.328 | 0.070 | 0.237-0.450 |
| HZ-0B memory | 0.305 | 0.050 | 0.237-0.375 |

Reported plainly: memory does NOT show an advantage here -- its mean
(0.305) is slightly BELOW the adapter's (0.328), and both are only
modestly above the 0.250 chance floor. 4 of 5 memory seeds again show
the overfitting signature already seen on code-symbol tracking (final
train loss 0.02-0.19, far below the adapter's 1.2-1.7) without a
corresponding held-out win; the 5th (seed 557) stays undertrained
(loss 0.93) and is also unremarkable on held-out accuracy (0.338).

## An emerging, real pattern across this session's B11 tasks

Stated as a genuine hypothesis, not yet verified by a dedicated
ablation: memory's advantage appears concentrated on tasks with a
SINGLE clean fact to store and recall (`hz0b_b11_evaluation_results.md`:
mean 0.819-0.830; long-conversation consistency: mean 0.775, an even
WIDER margin over the adapter at 8x the gap) or exact-match retrieval
under mild noise once given enough data (passkey task after doubling
data: mean 0.608; distractor immunity: mean 0.769). It does NOT show
an advantage -- and in code-symbol tracking's case is actively WORSE
-- on tasks requiring the mechanism to discriminate among MULTIPLE
structurally similar entries under distraction (multi-hop's 2
distractor triples here) or to correctly overwrite/update a
previously-written key (code-symbol tracking's 3 reassignments). Both
of the negative-result tasks share overfitting as their failure
signature: very low training loss, held-out accuracy no better than
(or worse than) the parameter-matched no-memory adapter.

## What this adds to B11's real coverage

One more of the 16 named tasks moves from 0% to done, with an honest
negative result. B11's exit-gate support is now demonstrated to be
real on 4 of 6 tested named tasks (single-fact recall, long-
conversation consistency, passkey after the data fix, distractor
immunity) and NOT supported on 2 (code-symbol tracking, multi-hop
retrieval) -- a genuinely mixed, task-dependent finding, reported as
such rather than rounded toward either a blanket "works" or "doesn't
work" claim. Remaining scope: 1 more named task (tool-result reuse),
and real-model versions of 3 more Stage 5 scenarios (contradictory
info, near-identical keys, capacity pressure).
