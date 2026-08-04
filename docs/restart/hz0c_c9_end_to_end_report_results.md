# HZ-0C C9: End-to-End Quality, Cost, Latency, and Adversarial-Failure Report

Date: 2026-08-04. Closes the last open C9 Required Artifact and the
plan's own completion-definition item 8 ("Trigger cost, latency, and
failure modes are documented",
`plans/HZ-0C_Surprise_Anchors_Total_Restart_Plan.md`).

`scripts/hz0c_c9_end_to_end_report.py`. Trains the same causal distilled
controller used throughout C9 (train seeds 555+556, eval seed 557, exact
15% rate) and adds three measurements prior C4/C9 reports did not have:
real per-call latency, the actual LM-loss cost of the controller's
missed triggers (not just their count), and one new adversarial scenario.

Reproduce: `PYTHONPATH=. .venv/bin/python scripts/hz0c_c9_end_to_end_report.py --examples 32 --adversarial-examples 32 --latency-repeats 15`

## 1. Quality (consistency check against prior C9 numbers)

Mean recall across the 8 real C3 scenarios: **0.5182** (per-scenario:
0.4375, 0.71875, 0.34375, 0.375, 0.5625, 0.53125, 0.4896, 0.6875) --
matches `docs/restart/hz0c_c9_matched_cost_report_results.md`'s recorded
`0.5013`/`0.5182` for the same controller-training recipe almost exactly,
confirming this report's controller-training path is a faithful replay,
not a different pipeline producing a coincidentally similar number.

## 2. Missed-trigger cost -- not just how often, but what it costs

Across all 8 scenarios (32 examples each), the deployed controller missed
**156** of the ground-truth positions at the exact 15% rate. For each
missed position, adding just that one position back in (single-position
ablation, isolated to its own example row) and measuring the real LM-loss
delta gives a mean cost of **+0.0164** nats -- small but real and
positive on average: missing a ground-truth position typically does cost
something.

**Real nuance, not smoothed over**: 7 of the 8 scenarios' mean costs were
positive (ranging `0.0038` to `0.0879`), but scenario 7's mean cost was
**NEGATIVE (-0.0083)** -- adding its missed anchors back in made loss
WORSE on average, not better. Per-example variance within scenarios is
also real and sometimes large (std up to `0.111` on scenario 5, driven by
a few high-cost outlier examples inside an otherwise modest-cost
scenario). Not every "ground truth" position is unconditionally
beneficial to attend to; some anchors are net-neutral or even mildly
counterproductive for this specific downstream token, consistent with
attention sometimes adding noise rather than signal at a given position.
This is disclosed as a real property of the mechanism, not rounded
toward "missing anchors is always costly."

## 3. Latency -- the real reference implementation has NO speed benefit from sparsity yet

| Policy | Mean seconds/call | Mean ms/token |
| --- | ---: | ---: |
| No anchors (0% rate) | 0.01917 | 0.0599 |
| Fixed periodic (15% rate) | 0.01854 | 0.0579 |
| Full attention (100% rate) | 0.01858 | 0.0581 |

**All three are statistically indistinguishable** (differences smaller
than the run-to-run std). This is a real, important, and honestly
disclosed limitation: `conditional_forward`
(`scripts/hz0c_c6_conditional_attention_eval.py`) implements triggering
via ADDITIVE MASKING over a full O(seq^2) attention computation -- exactly
matching what C8's own kernel contract requires for correctness, but it
means the REFERENCE Python/MLX path does the same amount of work
regardless of how sparse the trigger is. Any real latency win from
triggering is gated entirely on C8's still-open "model-level
integration" item: dispatching the actual PMetal sparse kernel (which
DOES skip non-triggered computation, per
`docs/restart/hz0c_c8_pmetal_attention_results.md`) from this forward
path, not yet done. Reporting this now, rather than only after that
integration exists, so the gap is visible rather than assumed away.

## 4. Adversarial scenario: gradual drift with no sharp onset

`scenario_gradual_drift` (new, in this report's own script): a real topic
shift from one real corpus source to another, but instead of
`scenario_topic_shift`'s single abrupt boundary, each position's
probability of being drawn from the second source rises LINEARLY from 0
to ~1 across the sequence. No individual token is locally anomalous (both
sources are real, in-distribution content, per this project's own
task-construction discipline); only the CUMULATIVE drift is real. Ground
truth is the midpoint where the mixture crosses 50% -- purpose-built to
stress-test the mechanism's own documented weakness (only the FIRST token
of a multi-position anomaly is strongly surprising,
`reference/hz0c_surprise_trigger.py`'s module docstring) in its most
extreme form: a "anomaly" with no first token at all.

**Result: recall drops to 0.281**, versus **0.518** mean recall on the 8
original in-distribution scenarios -- close to (though somewhat above)
the naive 6-of-40-positions random baseline (~0.15). This is a real,
disclosed failure mode: the controller is measurably worse at catching a
gradual, onset-free drift than a sharp one.

**A tempering nuance, also measured rather than assumed**: the
downstream LM-loss cost of THIS scenario's missed triggers was
**-0.0005** nats on average -- essentially zero, unlike the original 8
scenarios' **+0.0164**. The gradual-drift "midpoint" ground truth is a
real semantic marker, but attending to it specifically is not measured to
matter much for next-token prediction quality here -- so this is a real,
disclosed recall failure, but not shown to be an expensive one for this
particular adversarial construction. Both facts are reported together
deliberately, since either alone would be misleading (low recall alone
overstates the practical damage; near-zero cost alone would hide a real
detection gap).

## Summary for the plan's completion-definition item 8

- Trigger cost: quantified per-missed-position, not just counted
  (+0.0164 mean, with real negative-cost exceptions disclosed).
- Latency: measured directly; found to show NO current benefit from
  sparsity in the reference path, an honest gap pointing at C8's
  remaining model-level-integration work, not hidden.
- Failure modes: one new adversarial construction (gradual, onset-free
  drift) built and evaluated, with both its recall failure AND its
  measured downstream cost reported together.
