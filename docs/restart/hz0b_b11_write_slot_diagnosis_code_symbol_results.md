# HZ-0B B11: Root-Cause Diagnosis for Code-Symbol Tracking's Failure

Date: 2026-08-01. `docs/restart/hz0b_b11_code_symbol_tracking_results.md`
named an unverified hypothesis for that task's negative result (memory
mean 0.283 vs. adapter 0.370): "the controller may be writing/blending
all 3 reassignments rather than cleanly overwriting, relying on
training-set-specific artifacts instead of a generalizing 'most recent
write wins' rule." This directly tests it, in two stages.

`scripts/hz0b_b11_write_slot_diagnosis_code_symbol.py`. Trains ONE
real instance (seed 555, identical config to the negative-result run)
and then manually steps through all 8 held-out examples
position-by-position, calling `_choose_write_slot`
(`reference/hz0b_memory_simulator.py`) at each of the 3
`ASSIGN_MARKER` value positions to see which slot each reassignment's
write actually targets, plus the write_gate strength, pairwise key
cosine similarity between the 3 reassignments, and (stage 2) the
`gated_memory_read` weights at the final `READ_TRIGGER` position.

## Stage 1 result: the "split writes" hypothesis is REFUTED -- writes cleanly overwrite the same slot every time

| Held-out examples where all 3 reassignments hit the SAME slot | 8/8 |
| --- | --- |
| Held-out examples split across DIFFERENT slots | 0/8 |
| Mean write_gate at reassignment positions | 1.000 (range 1.000-1.000) |

Every single held-out example, all 3 reassignments to the symbol
route to the SAME memory slot (slot 0), with write_gate essentially
saturated at 1.000 every time -- a full-strength commit, not a
partial blend. The originally named hypothesis is wrong, stated
plainly rather than left standing.

## Stage 2 (same day): the READ step is the real culprit -- confirmed directly

Extended the same trained instance to also inspect `gated_memory_read`
(`reference/hz0b_readonly_integration.py`) at the final `READ_TRIGGER`
position -- does the read query correctly concentrate on slot 0 (which
holds the correct, freshly-overwritten value)?

| Examples where READ correctly argmax-focused on the correct slot | 2/8 |
| --- | --- |
| Mean read_weight placed on the correct slot | 0.1512 (range 0.0748-0.2075) |

**Only 2 of 8 held-out examples correctly retrieve slot 0.** The
other 6 focus (via softmax argmax) on a DIFFERENT slot entirely (1,
2, or 4 across different examples), and even the "correct" 2 examples
only place modest weight (0.204-0.208) on slot 0 -- close to the
0.125 uniform-across-8-slots baseline, not a confident, selective
retrieval. This is the real, confirmed root cause: **the write
mechanism stores the right information in the right place, but the
read mechanism cannot reliably find it.**

**A real, disclosed limitation of this diagnostic**: read weight this
close to uniform, spread across NOMINALLY-empty slots, is surprising
under `read()`'s own confidence-weighted math (an empty slot's
`log(confidence+eps)` term is ~-13.8 nats, which should be
overwhelmed by slot 0's near-1.0 confidence in almost any case) --
UNLESS slots other than 0 are not actually empty. Since `sequential_
latent_write_and_read` gives every position a chance to write, not
just the 3 marker positions, the ~30+ filler/padding positions between
markers may also be triggering small nonzero write_gate values that
softly populate OTHER slots with filler-derived noise, diluting the
read's ability to concentrate on slot 0. This was not directly
re-measured in this pass (filler-position write_gate values were not
logged), but is strongly consistent with the ALREADY-established,
independently-found B8 Stage 3 finding (`docs/restart/hz0b_b8_stage3_results.md`
section 3, and the tracker's own summary of "5 independent failed fix
attempts for write-position selectivity"): write_gate is not cleanly
selective to the semantically meaningful positions, a real, prior,
non-shallow local optimum this whole investigation has repeatedly
encountered. Named as the most likely explanation, not re-verified to
certainty this pass.

## What this adds to B11's real coverage

A complete, honest root-cause chain for code-symbol tracking's
negative result: (1) an initially plausible hypothesis (split/blended
writes) was named, then (2) directly tested and REFUTED with real
evidence (writes cleanly overwrite the correct slot), then (3) the
real cause was isolated one level deeper (the READ step fails to
reliably retrieve the correctly-written slot), and (4) the most likely
explanation for THAT was connected to an already-established,
independently-discovered systemic issue (write_gate's lack of
positional selectivity) rather than treated as a new, unrelated
mystery. Each step disclosed plainly, including what was NOT
re-verified. Not attempted this pass: directly logging filler-position
write_gate values to confirm the dilution hypothesis, or testing
whether `ste=True` (hard/discrete writes, which B8's own investigation
found gives a real but partial improvement to selectivity) also
improves code-symbol tracking's read-focus specifically.
