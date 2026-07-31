# HZ-0B B8, Stage 5 (Adversarial Memory): Results

**Update (2026-07-30, same date): both findings below are now fixed** in
`reference/hz0b_memory_simulator.py` -- see "Both findings fixed" at the
end of this document. The findings themselves are kept below exactly as
originally written, since they're still an accurate description of what
was wrong and why -- fixing something doesn't retroactively make finding
it not have been real.

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

## Honest scope (at the time this was first written)

Both findings are properties of B2's simulator as it exists today, not
new bugs introduced by this Stage 5 work -- Stage 5's job was to look for
exactly this kind of gap under adversarial framing, and it found two real
ones rather than confirming everything works.

## Both findings fixed (2026-07-30, same date)

**Finding 1 (key conflation)**: `_choose_write_slot`'s match threshold
raised from 0.95 to 0.999 -- only near-EXACT key identity now takes the
in-place-update path; anything less goes through the normal eviction-
scored "new fact" path instead. Re-ran the exact same adversarial
scenario (cosine 0.995, two different facts): now routes to two distinct
slots, both facts independently retrievable (cosine >0.99 to their own
value each). `test_overwrite_existing_fact` (the legitimate same-key
update case) is unaffected -- it reuses the literal same key, similarity
exactly 1.0, still above the new threshold.

**Finding 2 (confidence-blind reads)**: `read()` gained
`confidence_weighted: bool = True` (new default) -- adds `log(confidence
+ eps)` to each slot's similarity score before ranking. A new scenario
(`scenario_stale_vs_fresh_competition`: two slots holding the identical
key, one heavily decayed, one fresh -- a genuine tie in raw similarity)
confirms the fix directly: confidence-weighted read now correctly prefers
the fresh memory, while the unweighted path (kept available via
`confidence_weighted=False`) still resolves the tie arbitrarily
(lower-index slot wins), demonstrating exactly what the fix changes.
Does not break the "empty memory behaves like no memory" guarantee B6-B9
depend on: a uniform additive shift (all-empty-slots case) is invariant
under softmax/argmax, and an empty slot's value is exactly zero
regardless of its read weight.

Both fixes verified against the full test suite: 186/186 pass, zero
regressions across B2-B9's own tests (which use either exact-duplicate
keys or clearly-orthogonal ones, never near-threshold values the
tightened match threshold would affect).
