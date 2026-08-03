# HZ-0B B11: Reinforcement / Forgetting / Serialization Accuracy

Date: 2026-08-01. Closes one more of B11's 16 named eval tasks. Unlike
the existing unit tests for `reinforce`/`forget_or_decay`/`serialize`/
`restore` (`tests/reference/test_hz0b_memory_simulator.py` etc, which
check each op's local behavior in isolation), this measures real
RETRIEVAL ACCURACY across an extended session: do reinforced facts
stay retrievable while unreinforced facts decay into
unretrievability, and does serialize/restore preserve that behavior
exactly. Pure B2 simulator, no LM needed (a property of the mechanism
itself, same reasoning as B8 Stage 5 / the reopening-criteria scripts).

`scripts/hz0b_b11_reinforcement_forgetting_serialization.py`. 6 facts
written into 8 slots (oracle slot assignment, no eviction confound).
3 facts ("reinforced") get `reinforce()` called every 10 steps
(simulating periodic re-access); 3 ("unreinforced") only decay via
`forget_or_decay(decay_rate=0.85)` every step, for up to 150 steps.

## Result: a real, gradual-then-saturated forgetting curve

| Steps | Reinforced accuracy | Unreinforced accuracy | Unreinforced confidence |
| --- | --- | --- | --- |
| 10 | 1.000 | **0.667** (2/3 still correct) | 0.197 |
| 20 | 1.000 | 0.000 | 0.039 |
| 30 | 1.000 | 0.000 | 0.008 |
| 50-150 | 1.000 | 0.000 | ~0.000 |

Not a floor/ceiling artifact of one arbitrary parameter choice: at
step 10, unreinforced facts are still mostly retrievable (2 of 3),
confirming the transition is a genuine gradual decay, not an
instantaneous cliff. By step 20 unreinforced accuracy saturates to
0.000 and stays there through step 150. Reinforced facts hold 1.000
throughout every tested horizon.

**Update (2026-08-03)**: `reference/hz0b_memory_simulator.py`'s
`SOFT_READ_SCORE_SCALE=4.0` (added 2026-08-02 for overwrite/multi-hop
discrimination elsewhere) legitimately shifts the exact saturation
point from step 20 to step 30. This was verified only after also
fixing two real regressions the same date, both in `read()`'s
confidence-weighted scoring: an unpopulated-slot score floor of `0.0`
that let empty slots beat a real, sole, heavily-decayed candidate
(broke `test_stale_memory_alone_still_retrieves_fine_no_competition_to_lose_to`
and this doc's own saturation test), and `SOFT_READ_SCORE_SCALE` being
applied a SECOND time to the whole combined score at softmax time (an
unintentional double-scaling bug that oversharpened reads to the
point of losing query-relevance discrimination and gradient signal --
broke `test_unrelated_memory_produces_smaller_change_than_matching_memory`
and `test_gated_memory_read_is_differentiable`). The qualitative
finding here (gradual decay, real saturation, not a floor artifact) is
unchanged, only the specific step number, reconfirmed under the fully
corrected formula. Regression test updated accordingly
(`tests/reference/test_hz0b_b11_reinforcement_forgetting_serialization.py`).

**A real, non-obvious mechanism finding, not just "forgetting works":**
at full decay (confidence -> ~0), an unreinforced fact's OWN slot
still has the highest raw key-similarity (cosine ~1.0 against its
exact key) but LOSES the read argmax to a reinforced DECOY slot with
an unrelated key -- because `read()`'s `confidence_weighted` scoring
adds `log(confidence)` to the similarity score, and a ~13-nat
confidence penalty (from `log(1e-6)` at near-zero confidence) swamps
the ~1.0-2.0 range of cosine-similarity differences. In other words,
the read mechanism doesn't just "fail to find" a forgotten fact -- it
actively prefers a confident wrong answer over a decayed right one,
which is arguably the correct behavior for a confidence-aware memory
but is worth having stated explicitly rather than assumed.

## Serialize/restore accuracy: exact, at every step count tested

At every step count in the sweep (10, 20, 30, 50, 80, 150):
- all 8 `MemoryState` fields (`keys`, `values`, `confidence`, `age`,
  `protection`, `write_count`, `last_write_step`, `write_source`) are
  bit-exact after a `serialize()` -> `restore()` roundtrip
  (`mx.array_equal`, not an approximate comparison)
- retrieval accuracy (both reinforced and unreinforced) is identical
  before and after restore
- `read()` readout VALUES are bit-exact before/after restore, not
  just the accuracy label

No precision loss, no silent state drift across the roundtrip at any
point in the forgetting curve tested.

## What this adds to B11's real coverage

One more of the 16 named tasks moves from 0% to done, with a real
accuracy curve (not a single point) and a genuine mechanism-level
finding about how confidence-weighting and key-similarity interact
under decay. Remaining scope: multi-hop, long-conversation
consistency, tool-result reuse, code-symbol tracking (4 more named
tasks), and real-model versions of 3 more Stage 5 scenarios
(contradictory info, near-identical keys, capacity pressure).
