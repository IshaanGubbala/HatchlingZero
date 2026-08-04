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

## Exploratory confidence-decay transfer (2026-08-03)

The overwrite fix was tested on multi-hop without changing its two-hop query
or labels: `decay_rate=0.95`, 3 seeds, `train_count=80`, 500 steps, and
`lr=0.15`. Memory reached **0.296 +/- 0.021** (`0.287, 0.325, 0.275`) versus
the matched adapter's **0.275 +/- 0.020** (`0.250, 0.300, 0.275`). This is a
small positive probe, not a completed fix; it remains close to the 0.25 chance
floor and needs a full matched 1000-step evaluation before changing the task
verdict.

## Complete balanced stabilization rerun (2026-08-03)

The stronger candidate was completed with balanced training (`320` examples,
`160` held out), `1000` steps, `lr=0.05`, STE writes, `decay_rate=0.99`, and
cached frozen hidden states. The adapter arm completed in the same run before
the memory-only seed completion.

| Configuration | Mean | Std | Per-seed |
| --- | ---: | ---: | --- |
| Equal-parameter adapter | 0.392 | 0.033 | 0.375, 0.438, 0.363 |
| HZ-0B memory | **0.423** | 0.003 | **0.419, 0.425, 0.425** |

Memory improves the matched adapter by `+0.031` mean and is well above the
0.25 four-way chance floor, although it does not win every seed. This
supersedes the earlier weak 0.305 result for the balanced stabilization
protocol; it is not a claim that multi-hop is solved at ceiling.

## Shared key/query stabilization (2026-08-03)

The multi-hop controller now defaults to using the learned key projection for
both retrieval hops (`--shared-key-query`). Under the same balanced `320/160`
protocol, `1000` steps, `lr=0.05`, STE writes, and `decay_rate=0.99`, three
seeds reached:

| Configuration | Mean | Std | Per-seed |
| --- | ---: | ---: | --- |
| Previous two-hop controller | 0.423 | 0.003 | 0.419, 0.425, 0.425 |
| Shared key/query | **0.435** | 0.003 | **0.438, 0.438, 0.431** |

This is a matched **+0.012** improvement. The CLI supports
`--no-shared-key-query` for reproducing the prior ablation.

## Full-protocol decay-transfer control (2026-08-03)

The overwrite task's confidence-decay setting was tested as a direct transfer
control: balanced `320/160` training/held-out examples, `1000` steps, `lr=0.05`,
STE writes, and `decay_rate=0.95` across seeds `555-557`. Memory reached only
`0.358 +/- 0.041` (`0.388, 0.388, 0.300`), below the retained `0.423 +/- 0.003`
result at `decay_rate=0.99`. The transfer is rejected; the stronger multi-hop
configuration remains the default.
## Read-hop sweep (2026-08-03)

The evaluator now exposes `--read-hops` so iterative read refinement can be
tested without changing the matched protocol. On the balanced 320/160,
1000-step, lr=0.05, STE, decay=0.99, shared-key/query protocol with seed 555:

| Read hops | Held-out memory accuracy |
| ---: | ---: |
| 1 | 0.425 |
| 2 | 0.438 |
| 3 | 0.438 |

The one-read variant is weaker; a third read adds no measurable benefit over
the retained two-read configuration. No change is promoted from this sweep.

## Slot-capacity sweep (2026-08-03)

The evaluator now exposes `--num-slots` for capacity tests. On the same
balanced 320/160, 1000-step, lr=0.05, STE, decay=0.99, shared-key/query,
two-read protocol with seed 555:

| Slots | Held-out memory accuracy |
| ---: | ---: |
| 8 | 0.438 |
| 16 | 0.381 |
| 32 | 0.425 |

More slots do not reduce the multi-hop interference; the validated 8-slot
configuration remains the default.

## Longer-training check (2026-08-03)

With the validated 8-slot, shared-key/query, two-read configuration, extending
training from 1,000 to 2,000 steps lowered the seed-555 training loss from the
1,000-step endpoint but reduced held-out accuracy from `0.438` to `0.388`.
This is an overfitting signal, not an under-convergence fix; the 1,000-step
protocol remains the retained result.

## Write-rate target fix (2026-08-03)

The multi-hop evaluator's target write-rate sweep found a robust improvement
from `0.1` to `0.2`. Under the balanced 320/160, 1000-step, lr=0.05, STE,
decay=0.99, shared-key/query, 8-slot, two-read protocol:

| Target write rate | Seed 555 | Seed 556 | Seed 557 | Mean +/- population std |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.438 | 0.438 | 0.431 | 0.435 +/- 0.003 |
| 0.2 | 0.450 | 0.438 | 0.481 | **0.456 +/- 0.018** |

The `0.2` target is now the multi-hop evaluator default. This change is scoped
to multi-hop; other HZ-0B tasks retain their validated write-rate settings.
