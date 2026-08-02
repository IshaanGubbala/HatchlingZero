# HZ-0B B11: Real-Model Capacity Pressure

Date: 2026-08-01. Closes the last of the disclosed real-model Stage 5
gaps named repeatedly this session: B8 Stage 5's own capacity-pressure
scenario is pure-simulator (facts written via direct `MemoryState.write()`
calls, oracle slot assignment, no LM, no learned write timing/slot
choice). This tests the REAL, LEARNED write mechanism under genuine
capacity pressure -- 10 distinct facts presented in a single example,
but the memory controller has only 8 slots (`NUM_SLOTS=8` <
`NUM_FACTS_PER_EXAMPLE=10`), unlike every earlier B11 real-model task
(which always had slots >= facts written per example). A single
random query per example asks for one of the 10 facts' values.

`scripts/hz0b_b11_real_model_capacity_pressure.py`. Real frozen HZ-0A
checkpoint, `precomputed_hidden` caching, the validated
`lambda_sparse=0.1` + `target_write_rate=0.1` config. 5 seeds,
steps=1000, lr=0.15, train_count=80, held_out_count=80, 4-way value
(chance=0.250).

## Result: near-chance for both conditions -- an honest, inconclusive finding

| Configuration | mean | std | range |
| --- | --- | --- | --- |
| Floor | 0.000 | -- | -- |
| Equal-param adapter | 0.220 | 0.067 | 0.163-0.325 |
| HZ-0B memory | 0.255 | 0.029 | 0.225-0.300 |

Reported plainly: **neither condition shows meaningful signal above
chance (0.250) at this scale.** The adapter's mean (0.220) is actually
BELOW chance; memory's mean (0.255) sits almost exactly AT chance.
Memory does edge out the adapter (+0.035) and has a tighter spread
(std 0.029 vs 0.067), but the margin is too small relative to the
per-seed noise to call this a real advantage. Unlike code-symbol
tracking/multi-hop/tool-result-reuse's overfitting signature, memory's
training loss here does NOT collapse to near-zero (final losses
0.27-0.96, not fully converged) -- consistent with genuine task
difficulty (encoding and distinguishing 10 simultaneous facts with a
692K-param budget and only 80 training examples) rather than a
memory-specific failure, since the adapter struggles equally or worse.

**Per-position retrieval accuracy (one seed, 10 fact positions,
n=5-13 held-out examples per position)** shows no clear monotonic
recency trend (position 0: 0.385, position 2: 0.000, position 7:
0.429, position 8: 0.000) -- too noisy at this held-out sample size
(80 examples split unevenly across 10 query positions) to draw a real
conclusion about which facts survive capacity pressure.

## What this adds to B11's real coverage

This is the last of B11's disclosed real-model Stage 5 gaps
attempted this session. Unlike the earlier 3 negative results (which
had a clear, explainable overfitting signature), this one is a
genuine near-null: both conditions are statistically close to chance,
most plausibly because 10 simultaneous facts in 80 training examples
is simply too little signal for either a 692K-param adapter or memory
controller to learn reliably within 1000 steps, not because memory
specifically fails under capacity pressure. Not yet tried: more
training data/steps (the fix that helped the passkey task, though it
hurt code-symbol tracking) or a larger held-out set for a less noisy
per-position breakdown. Honest status: inconclusive, not a positive
or negative finding for the exit gate -- reported as such rather than
rounded toward either direction.

Remaining B11 scope: real-model versions of 2 more Stage 5 scenarios
(contradictory info, near-identical keys) -- both currently tested
only against the pure B2 simulator.
