# HZ-0C C2: Surprise Signal Validation

Date: 2026-08-02. C2's own exit gate: "surprise correlates with
controlled novelty or difficulty." Real evidence against the frozen
HZ-0A checkpoint (same "real model, controlled synthetic construction"
pattern as every HZ-0B B11 real-model task), not a unit-test sanity
check -- `surprise_score()`'s hidden-state-delta-norm signal
(`reference/hz0c_surprise_trigger.py`), chosen in C1 for needing zero
extra learned parameters and working at real inference time.

`scripts/hz0c_c2_surprise_validation.py`. Two scenarios against the
real, frozen checkpoint.

## Scenario 1: novelty point -- FAILS, in the wrong direction

32 sequences, each a repeated 4-token cyclic pattern (8 repetitions,
32 tokens total) with ONE unexpected token injected at a known,
randomized position (avoiding the first 2 repetitions' startup
transient and the sequence's tail).

| Metric | Value |
| --- | --- |
| Mean normalized surprise AT the injected novelty position | **-0.459** |
| Mean normalized surprise immediately AFTER novelty | -0.392 |
| Mean normalized surprise at steady-state (repeated) positions | **0.363** |
| Delta (novelty - steady-state) | **-0.822** |
| Fraction of examples where novelty scores ABOVE steady-state mean | **0.094** (should be ~1.0 for a working signal, ~0.5 for a useless one) |
| Fraction of novelty positions triggered at a 15%-target-rate threshold | **0.000** |

**This is a real, reproducible, wrong-direction result, not noise or a
confound.** Re-verified after excluding the pattern's first 2
repetitions from the "steady-state" baseline (in case an early
startup transient was inflating it) -- the negative result got
STRONGER, not weaker (fraction dropped 0.219 -> 0.094). The injected
anomaly scores LOWER than ordinary repeated-pattern positions, and at
a rate-bounded 15% trigger threshold, ZERO of the 32 injected
novelties actually trigger. Hidden-state delta norm, at least measured
after the full block's residual+MLP pathway, does NOT reliably spike
at a single unexpected token embedded in an otherwise-predictable
sequence -- the opposite of what a novelty-point detector needs.

**A real, disclosed mechanistic hypothesis (not verified this pass)**:
within a genuinely repeating pattern, the model's recurrent state may
still be doing real, nontrivial work each cycle (tracking position/
periodicity), producing meaningfully large deltas at ordinary
positions, while a single anomalous token's effect on the hidden state
may be dampened by the recurrent gates or smoothed by the residual/MLP
pathway rather than amplified -- the opposite of the naive
"unexpected input -> big state change" intuition this signal assumed.

## Scenario 2: difficulty proxy -- works, real and strong

| Configuration | Mean raw surprise |
| --- | --- |
| Random (high-entropy) tokens | 124.27 |
| Constant (low-entropy) token repeated | 17.53 |
| **Ratio** | **7.09x** |

A real, strongly correctly-directioned signal: genuinely unpredictable
token sequences produce far larger hidden-state deltas than a
trivially predictable constant stream. This IS what "surprise
correlates with difficulty" should look like.

## Honest verdict on C2's exit gate

The exit gate says "correlates with controlled novelty OR
difficulty" -- satisfied via the difficulty half (Scenario 2), NOT
via the novelty-point half (Scenario 1), which is the more
architecturally relevant use case for anchor-triggering ("spend
quadratic attention when the recurrent state encounters something
UNEXPECTED" -- a point-in-time event, not a stream-level entropy
property). Reported precisely rather than rounding "met via OR" up to
"the signal works": hidden-state delta norm is a reasonable general
difficulty/entropy proxy but a poor novelty-POINT detector, and should
not be the sole signal carried into C3's isolated trigger simulator
without either fixing this or trying one of C2's other named
candidates (token-loss proxy, recurrent/attention disagreement, state
novelty, HZ-0B memory-read uncertainty) specifically for the
novelty-point case.

## Follow-up (2026-08-02, same day): a second candidate signal also fails -- likely a task-construction issue, not just a bad signal choice

Built `state_novelty_score()` (cosine distance to a causal windowed
mean of recent hidden states, not just the immediately preceding
position) specifically to fix the failure above -- verified on a toy
synthetic pattern-break example that it correctly detects an anomaly
there (`test_state_novelty_score_high_for_pattern_break`). Re-ran the
SAME real-checkpoint Scenario 1 construction with it:

| Signal | Delta (novelty - steady-state) | Fraction novelty above steady-state | Novelty positions triggered at 15% rate |
| --- | --- | --- | --- |
| delta-norm (`surprise_score`) | -0.822 | 0.094 | 0/32 |
| state-novelty, window=4 | -0.322 | 0.031 | 0/32 |
| state-novelty, window=8 | -0.245 | 0.156 | 0/32 |

**Both signals fail, in the same direction, on the same construction**
-- state-novelty is not clearly better (window=4 is actually worse
than delta-norm; window=8 is marginally less bad but still fails).
This pattern -- two mechanistically different signals both failing
the same way -- is itself evidence worth taking seriously: rather than
"the signal choice was wrong twice," a more likely explanation is that
**the task construction itself may not be a fair test**. The
"pattern" here is a cycle of RANDOM, semantically meaningless token
IDs -- a real language-trained model has no reason to have built a
strong sequence-level expectation for an arbitrary numeric cycle the
way it would for a genuinely repeated natural-language phrase or
topic. If the model never treats "random token ID repeats every 4
positions" as something worth predicting in the first place, there is
no real expectation for an anomaly to violate, and NEITHER signal
should be expected to fire -- consistent with what was actually
observed.

**Real next step, not attempted this pass**: rebuild Scenario 1 using
a construction closer to what a language model actually models well
(e.g., a repeated short real-token n-gram drawn from the tokenizer's
actual vocabulary in a way that forms a genuine local pattern, or a
task built directly from natural corpus data with a real topic-shift
or contradiction inserted, matching the plan's own C3 task list:
"topic shifts, long-range key reappearance, changed variable bindings,
... contradictions"). This is now the priority before either signal
is judged as failed for real.

## What this adds to HZ-0C's real progress

A real, checked answer to C2's exit gate, with the negative half
disclosed as prominently as the positive half. This is exactly the
kind of result this project's own discipline exists to catch before
it becomes a hidden assumption baked into C3's simulator or C6's
integration -- shipping `surprise_score()` as-is into C3 without this
finding would have meant building and evaluating a trigger simulator
around a signal already known not to catch its main intended case.
Real next step, not attempted this pass: implement and test the
token-loss-proxy or recurrent/attention-disagreement candidate
specifically on Scenario 1's novelty-point construction, to find a
signal that actually satisfies the exit gate's novelty half before C3.
