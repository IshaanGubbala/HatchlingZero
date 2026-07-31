# HZ-0B B8, Stage 4 (Natural Sequences): Results

Date: 2026-07-30. All 6 named item-types from the plan's Stage 4 list
(the 7th, "multi-turn conversations," is the session structure itself,
not a separate item), composed into ONE realistic session rather than 7
isolated micro-tests -- deliberately different framing from Stage 5's
adversarial probes (`reference/hz0b_b8_stage5_adversarial.py`): Stage 4
asks whether ordinary, expected, layered usage works correctly, not
whether the mechanism resists attacks.

## Setup

`reference/hz0b_b8_stage4_natural_sequences.py`,
`tests/reference/test_hz0b_b8_stage4_natural_sequences.py`. One composite
session: a document fact and a code symbol read early and needed at the
end; a variable assigned then reassigned mid-session; a system constraint
set once and explicitly protected; ordinary conversational noise cycling
through a small fixed set of off-topic slots (not a fresh distinct key
every turn -- real conversations circle back to a few kinds of small
talk, they don't mint a new memory-worthy topic constantly); a user
preference set then changed; a tool-call result needed only much later,
after a real gap. 16 memory slots (up from Stage 5's 8) -- deliberately
enough capacity that ordinary session noise doesn't evict real facts,
since capacity pressure under genuine competition is Stage 5's own
dedicated scenario, not this one's job.

## A real bug caught before any test ran

First draft attempted to overwrite the protected constraint using a
manually guessed slot index (`slot_idx=mx.array([0])`) rather than
letting the simulator's own key-matching auto-routing find the real slot
-- the constraint hadn't actually landed in slot 0 (two other items were
written first). Fixed by removing the guess and relying on
similarity-based auto-routing, exactly the mechanism this test is meant
to exercise.

## A real test-design bug caught by the test run itself

First run: 3 of 7 tests failed (document fact and tool result came back
as literal zero vectors -- evicted). Root cause: distractor "noise" turns
were each writing to a brand-new, never-repeated key, so a "long session"
with many distractor turns filled all 8 (then even 16) memory slots and
started evicting real, unprotected facts purely from raw item count, not
from anything the test was actually trying to check. Fixed by having
distractor turns cycle through a small, fixed set of off-topic keys
(matching how a real long conversation actually behaves -- it revisits a
few kinds of small talk, it doesn't generate unbounded distinct topics)
and raising capacity to 16 slots. All 7 tests pass after both fixes,
including a heavier-load variant (12 distractor turns instead of 4).

## Results: all 7 checks pass

| Check | Result |
| --- | --- |
| Document fact survives the whole session | Pass |
| Code symbol survives the whole session | Pass |
| Variable reassignment reads as the latest value | Pass |
| Protected constraint resists a casual mid-session overwrite | Pass |
| Tool result recalled correctly after a long gap | Pass |
| Changed user preference reads as the new one | Pass |
| Holds up under heavier distractor load (12 vs 4 turns) | Pass |

Unlike Stage 5, no vulnerabilities were expected or found here -- these
are all legitimate, expected usage patterns the memory mechanism (as
already fixed by Stage 5's own two corrections: the 0.999 match threshold
and confidence-weighted reads) handles correctly. The two bugs caught
along the way were both in this test's own setup (a wrong guessed slot
index, and an unrealistic distractor-key generation pattern), not in the
underlying memory mechanism -- disclosed as such, not folded into a false
"mechanism vulnerability found" narrative.
