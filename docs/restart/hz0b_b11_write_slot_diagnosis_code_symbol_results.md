# HZ-0B B11: Root-Cause Diagnosis for Code-Symbol Tracking's Failure

Date: 2026-08-01. `docs/restart/hz0b_b11_code_symbol_tracking_results.md`
named an unverified hypothesis for that task's negative result (memory
mean 0.283 vs. adapter 0.370): "the controller may be writing/blending
all 3 reassignments rather than cleanly overwriting, relying on
training-set-specific artifacts instead of a generalizing 'most recent
write wins' rule." This directly tests it.

`scripts/hz0b_b11_write_slot_diagnosis_code_symbol.py`. Trains ONE
real instance (seed 555, identical config to the negative-result run)
and then manually steps through all 8 held-out examples
position-by-position, calling `_choose_write_slot`
(`reference/hz0b_memory_simulator.py`) at each of the 3
`ASSIGN_MARKER` value positions to see which slot each reassignment's
write actually targets, plus the write_gate strength and pairwise key
cosine similarity between the 3 reassignments.

## Result: the hypothesis is REFUTED -- writes cleanly overwrite the same slot every time

| Held-out examples where all 3 reassignments hit the SAME slot | 8/8 |
| --- | --- |
| Held-out examples split across DIFFERENT slots | 0/8 |
| Mean write_gate at reassignment positions | 1.000 (range 1.000-1.000) |

Every single held-out example, all 3 reassignments to the symbol
route to the SAME memory slot (slot 0), with write_gate essentially
saturated at 1.000 every time -- a full-strength commit, not a
partial blend. Per `write()`'s mechanics
(`reference/hz0b_memory_simulator.py`), a write_gate this close to
1.0 means `_blend_state_by_row` leaves almost none of the previous
state behind: each reassignment genuinely, cleanly overwrites slot
0's key AND value, in place, on held-out data -- not just on the
training set. Pairwise key cosine similarity BETWEEN the 3
reassignments' own computed keys ranges 0.60-0.99 (well below the
0.999 in-place-update-match threshold), meaning this consistent
same-slot routing is NOT happening via `_choose_write_slot`'s
"matching key" path -- it is landing on slot 0 through the general
eviction-scored competition path every time, a real (if not fully
traced to the exact numeric reason here) but CONSISTENT and CORRECT
routing behavior, not the named "split across slots" failure mode.

## What this means: the failure is downstream of write/slot-routing, not in it

The originally named hypothesis is wrong, stated plainly rather than
left standing. Since the write mechanism DOES correctly and
consistently overwrite the same slot with the final value on held-out
examples themselves, the code-symbol tracking task's 0.283 accuracy
(vs. adapter's 0.370, vs. chance 0.250) cannot be explained by writes
being scattered across multiple competing slots. The real cause must
be one of: (a) the READ step at the final `READ_TRIGGER` position not
correctly matching/retrieving slot 0's content despite it holding the
right value, (b) the `key_proj`/`value_proj`/read-query weights
overfitting a training-set-specific mapping that doesn't generalize
to held-out inputs even though the WRITE ROUTING generalizes fine, or
(c) some other read-side or output-decoding issue not yet isolated.
This narrows future diagnosis meaningfully: it should look at the
READ mechanism and the final logits decode, not at write/slot
selection, which this diagnostic shows is already working correctly.

## What this adds to B11's real coverage

A real correction to a previously-stated, plausible-sounding but
untested hypothesis -- caught and disclosed rather than left as an
unverified guess in the record. This is exactly the standing
discipline this investigation has applied throughout (verify before
trusting, disclose reversals openly): the original code-symbol
tracking doc's hypothesis is now marked REFUTED by direct evidence,
not merely "unverified". Follow-up work (not attempted this pass):
inspect the read-query construction / final logits at the
`READ_TRIGGER` position specifically, to isolate whether the read
step or the value/query projections' generalization is the real
remaining cause.
