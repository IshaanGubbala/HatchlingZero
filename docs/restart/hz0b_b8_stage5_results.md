# HZ-0B B8, Stage 5 (Adversarial Memory): Results

Date: 2026-07-30. All 7 scenarios the plan names verbatim, tested
directly against B2's memory simulator (`reference/hz0b_b8_stage5_adversarial.py`,
`tests/reference/test_hz0b_b8_stage5_adversarial.py`) -- no LM, no
training run. This tests properties of the memory MECHANISM itself,
independent of what drives it; B6/B7/B8's earlier integration work
already covers real-model behavior on top of this same mechanism.

## Results: 5 confirmed-correct, 2 real disclosed gaps

| Scenario | Result |
| --- | --- |
| Contradictory later information | **Correct** -- newer fact wins cleanly, no blending |
| Distractors | **Correct** -- unrelated writes don't disturb the real fact |
| Malicious overwrite attempt (protected memory) | **Correct** -- explicitly rejected, protection holds |
| Near-identical keys | **Real vulnerability** -- see below |
| Stale memories (decay) | **Correct** -- confidence genuinely decays (>5x reduction over 20 steps at decay_rate=0.9) |
| Stale memories (retrieval strength) | **Real gap** -- see below |
| Capacity pressure | **Correct** -- protected memory survives 11 competing writes, no overflow/crash |
| Reset boundaries | **Correct** -- reset wipes everything, including protected entries, no leakage |

## Finding 1: near-identical keys are silently conflated (a real adversarial surface)

Two keys at cosine similarity 0.995 (above B1/B2's fixed 0.95 match
threshold), representing two **genuinely different** facts: both writes
land in the same slot. The second write is treated as an update to the
first fact's existing entry, not a new, distinct memory -- confirmed
directly (both writes route to slot 0), not inferred. Querying either key
afterward returns only the second fact; the first is gone, silently, with
no protection check ever triggered (this is different from the
`malicious_overwrite_attempt` scenario, which correctly blocks an
explicit overwrite of a *protected* slot -- this conflation happens to an
*unprotected* slot purely from key geometry, a path protection doesn't
guard at all).

This is B1 decision 8's fixed 0.95 threshold working exactly as designed
-- not a bug in the simulator or this test. But it is a genuine
adversarial surface: anything (an attacker, or just unlucky embedding
collisions in a real trained system) that can produce a key close enough
to an existing one can silently overwrite it, because the write path
treats "close enough" as "the same fact, please update," not as "a
competing new fact that should go through eviction/protection logic."
Not fixed here -- documented as a real, disclosed limitation for whoever
works on B1's threshold design next.

## Finding 2: staleness (low confidence) does not weaken retrieval at all

Decayed a memory's confidence to under 1% of its original value (50 steps
at decay_rate=0.9), then read it with a hard (top-1) query that still
matches its key well: **the readout is unaffected** -- retrieved at full
strength (cosine >0.99 to the original value), despite the memory being
almost entirely "forgotten" by its own confidence score. Confirmed
directly, not assumed: B2's `read()` does not weight by confidence at
all, only by key similarity. Confidence currently only matters for
*write-slot eviction scoring* (deciding what to overwrite when capacity
is under pressure) -- it has no effect on what a read actually returns.
"Staleness" in the current design does not, on its own, make a stale
memory less trusted when queried; it only makes it more likely to
eventually be evicted by a competing write. A real, disclosed gap between
what "stale" intuitively implies (should be less trusted) and what the
mechanism actually does (no effect until eviction).

## Honest scope

Both findings are properties of B2's simulator as it exists today, not
new bugs introduced by this Stage 5 work -- Stage 5's job was to look for
exactly this kind of gap under adversarial framing, and it found two real
ones rather than confirming everything works. Neither is fixed in this
pass; both are candidates for a future B1/B2 revision (e.g. a
confidence-weighted or two-tier similarity threshold for the conflation
issue, a confidence-gated read for the staleness issue) if this project
continues past B9.
