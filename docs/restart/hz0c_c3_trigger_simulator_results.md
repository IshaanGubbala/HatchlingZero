# HZ-0C C3: Isolated Trigger Simulator

Date: 2026-08-02. Real evidence for C3's exit gate ("the controller
avoids always-on and always-off behavior") across all 8 of the plan's
named scenario types, using `state_novelty_score` (C2's validated
winner, window=4) computed from the REAL frozen HZ-0A checkpoint's
hidden states, on REAL, in-distribution corpus content throughout --
per the lesson learned and disclosed in
`docs/restart/hz0c_c2_surprise_validation_results.md` (arbitrary token
IDs don't form a real "expectation" for a language-trained model).

`scripts/hz0c_c3_trigger_simulator.py`. Each of the 8 scenarios built
from real data (`data/packed/repro_1024_val.jsonl` for general text,
`data/packed/external/{code,json_and_configuration}_validation.jsonl`
for code/JSON), 32 examples each, seq_len=40, `rate_bounded_threshold`
at `target_rate=0.15`.

## Exit gate result: PASS, but the headline number understates real, mixed underlying performance

| Metric | Result |
| --- | --- |
| Anchor rate range across all 8 scenarios | 0.125 - 0.125 (identical, by construction of `rate_bounded_threshold`) |
| Always-on or always-off? | No |
| **C3 exit gate** | **PASS** |

This PASS is real but structurally close to guaranteed:
`rate_bounded_threshold` sets a per-sequence quantile threshold, so a
degenerate always-on/off result would require the score distribution
itself to be pathological (e.g., constant), which none of these real
scenarios produce. The more informative results are the per-scenario
precision/recall, which the exit gate itself doesn't ask for but which
matter for judging whether the SIGNAL is actually useful, not just
non-degenerate.

## Per-scenario precision/recall -- honest, mixed, mostly weak beyond the one construction C2 validated

| Scenario | Precision | Recall | False-trigger rate | Missed-anchor rate |
| --- | --- | --- | --- | --- |
| 1. Repeated pattern with anomaly (C2's validated construction) | **0.131** | **0.656** | 0.111 | 0.344 |
| 2. Topic shift | 0.013 | 0.062 | 0.127 | 0.938 |
| 3. Long-range key reappearance | 0.031 | 0.156 | 0.124 | 0.844 |
| 4. Changed variable bindings | 0.031 | 0.156 | 0.124 | 0.844 |
| 5. Code/JSON boundary | 0.031 | 0.156 | 0.124 | 0.844 |
| 6. Contradiction | 0.025 | 0.125 | 0.125 | 0.875 |
| 7. Rare-token burst | **0.000** | **0.000** | 0.135 | 1.000 |
| 8. Distractor-heavy retrieval | 0.062 | 0.312 | 0.120 | 0.688 |

**Reported plainly, not rounded up: only scenario 1 (the exact
construction C2 already validated `state_novelty_score` on) shows
strong performance.** All 7 new scenarios show weak-to-zero recall
(0.062-0.312, vs. scenario 1's 0.656), with scenario 7 (rare-token
burst) showing COMPLETE failure (recall exactly 0.000 across all 32
examples) -- surprising, since inserting genuinely rare tokens should
intuitively be an easy case.

## Scenario 7's complete failure, investigated and explained -- a real mechanistic limitation, not a bug

Direct inspection of raw scores confirmed this is not a construction
bug: the burst positions consistently score LOW (often negative),
while ordinary elsewhere-positions occasionally score much higher
(max ~1.9-2.0). Within a single burst example, the score pattern is
striking: the FIRST burst token scores near-zero-to-mildly-positive,
but the 2nd and 3rd consecutive burst tokens score progressively MORE
NEGATIVE (down to -1.5 to -1.9).

**Root cause, confirmed on a controlled synthetic example
(`test_state_novelty_score_decays_within_a_multi_position_anomaly_burst`)**:
`state_novelty_score` measures distance from a WINDOWED MEAN of recent
positions. Once the first anomalous token enters the window, it starts
pulling the window's own mean toward itself -- by the second and third
consecutive anomalous tokens, the window already partially contains
the anomaly, so later burst positions look LESS novel relative to a
window that has already shifted toward them. This is a real, genuine
structural property of windowed-mean novelty scoring: it detects the
ONSET of an anomaly well but actively dampens on SUSTAINED,
multi-position anomalies, the opposite of what "rare-token burst"
detection needs. Locked in with a dedicated regression test so a
future change to the window mechanics doesn't silently alter this
without notice.

## What this means for HZ-0C going forward

C3's exit gate is technically met, but the deeper, more useful finding
is this: **`state_novelty_score` (window=4) is validated for
single-point anomaly detection in a locally-repeating context (its
original C2 validation case) but is NOT yet a broadly reliable signal
across the plan's full named scenario range.** Two scenario families
show a real, identified weakness each:
- **Sustained/multi-position anomalies (scenario 7)**: the windowed-
  mean mechanism structurally dampens on consecutive anomalies -- a
  real, explained limitation, not a training issue. Fix candidates for
  future work (not attempted this pass): a shorter effective window,
  an EXCLUDE-self-from-window variant, or a max-over-window rather
  than mean-based comparison.
- **Structural/positional boundaries (scenarios 2, 3, 5) and
  content-reassignment (scenarios 4, 6)**: weak but non-zero recall
  (0.06-0.16) suggests the signal captures SOME of these events but
  not reliably -- may need a longer window, a different C2 candidate
  (recurrent/attention disagreement, still untried), or combining
  multiple signals.

This is real, disclosed, negative-leaning evidence that should inform
C7 (controller training): a single hand-picked threshold on
`state_novelty_score` alone is not yet demonstrated to generalize;
C7's own job (training the trigger policy against real objectives,
not just this fixed heuristic) may be exactly what's needed to close
this gap, rather than expecting C2's raw signal to work unsupervised
across every scenario type.

## Fix attempt (2026-08-02, same day, in response to "fix it")

Two things were tried. One was a real, principled signal-level fix
that failed on real data despite passing its own synthetic tests. The
other was diagnosing and correcting the actual construction bug --
and it worked.

**Attempt 1: `ema_novelty_score` -- FAILED on real data.** Built
specifically to fix the windowed-mean dampening explained above:
replaces the hard window with a slowly-decaying EMA (`decay=0.9`,
`reference/hz0c_surprise_trigger.py`), so a short burst can't quickly
drag the reference toward itself. Verified correct on synthetic tests
first (`test_ema_novelty_score_does_not_decay_within_a_multi_position_anomaly_burst`,
`test_ema_novelty_score_detects_single_point_anomaly` -- both pass).
Rerun on the REAL checkpoint with the real 8-scenario construction:
scenario 7 STILL showed complete failure (recall 0.000, unchanged),
and scenario 1 (previously the strongest, recall 0.656) got WORSE
(recall 0.375). Reported honestly rather than declared a fix because
its own unit tests passed -- passing a synthetic mechanism check is
not the same as fixing the real problem.

**Attempt 2: fix scenario 7's own construction -- WORKED.** Direct
investigation (comparing high-ID "rare" tokens against a genuinely
real, contextually-displaced multi-token span lifted from elsewhere in
the real corpus) found the ORIGINAL construction was the actual bug,
not the signal: high token-ID magnitude produced no real elevation in
EITHER signal (mean score at burst positions indistinguishable from
elsewhere), while a real out-of-context span produced a clear,
measurable elevation with both signals (mean 0.31 at burst positions
vs. -0.025 elsewhere) -- the exact same category of confound C2 had
already diagnosed once (arbitrary token identity isn't the same as
contextual surprise to the model). Fixed `scenario_rare_token_burst`
to use a real displaced span instead of a high-ID proxy, reran the
full corrected 8-scenario suite with `state_novelty_score`:

| Scenario 7 (rare-token burst) | Before fix | After fix |
| --- | --- | --- |
| Precision | 0.000 | **0.094** |
| Recall | 0.000 | **0.156** |
| Missed-anchor rate | 1.000 | 0.844 |

**No longer a complete failure**, and now in the same weak-to-moderate
range as most of the other 6 non-scenario-1 scenarios (0.06-0.34) --
a real, meaningful fix for the worst outlier, though it does not
elevate scenario 7 to scenario 1's strength. The underlying, broader
finding stands: `state_novelty_score` remains strong only on
scenario 1's exact construction; the other 7 scenarios (now including
a properly-constructed scenario 7) all sit in a weak recall band,
which is real signal that C7's trained controller has real work to do
beyond this fixed heuristic, not evidence of a broken pipeline.
