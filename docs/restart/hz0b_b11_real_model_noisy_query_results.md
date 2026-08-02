# HZ-0B B11: Noisy Associative Recall (Real-Model)

Date: 2026-08-02. One of the plan's 16 named eval tasks
(`plans/HZ-0B_Total_Restart_Plan.md`'s B11 list: "noisy associative
recall"), previously only tested via B8 Stage 5's pure-simulator
"noisy query" scenario (raw key vector + Gaussian noise, renormalized
-- no LM involved). This is the real-model analog: Gaussian noise
injected directly into the REAL frozen backbone's hidden state at the
read-trigger position, before it reaches the memory read query
(`query = hidden_state @ query_w + query_b`) -- the natural real-model
counterpart of "the query is an imprecise/noisy cue."

`scripts/hz0b_b11_real_model_noisy_query.py`. Reuses
`scripts/hz0b_b11_baseline_comparison.py`'s exact single-fact task and
the validated `lambda_sparse=0.1` + `target_write_rate=0.1` config.
Trained ONCE per seed on clean data; evaluated at 6 noise levels
(0, 0.5, 1.0, 2.0, 5.0, 10.0x the read-trigger hidden state's own
per-example std), injected only at eval time, only at that one
position.

## Result: robust through moderate noise, then a sharp collapse -- and the adapter is MORE robust to extreme noise

| Noise (x std) | Adapter mean | Memory mean |
| --- | --- | --- |
| 0.0 | 0.512 | 0.819 |
| 0.5 | 0.522 | 0.816 |
| 1.0 | 0.537 | 0.803 |
| 2.0 | 0.531 | 0.741 |
| 5.0 | 0.506 | **0.097** |
| 10.0 | 0.494 | **0.000** |

Two real, honest findings, not one:

1. **At low-to-moderate noise (0-2x the hidden state's own std) memory
   is genuinely robust**, retaining most of its advantage over the
   adapter (0.819 -> 0.741, still +0.210 over the adapter at noise=2.0).
   This is the realistic regime for "an imprecise cue" and the
   mechanism holds up.
2. **Between 2x and 5x std, memory collapses catastrophically**
   (0.741 -> 0.097, essentially to chance/below by noise=10.0), while
   the **adapter is completely unaffected across the entire range**
   (0.494-0.537, no real degradation even at 10x std). This is a real
   crossover: past a certain noise threshold, the simple no-memory
   adapter is the MORE robust mechanism, not the memory.

**Why (a real, disclosed mechanistic hypothesis, not fully verified
this pass)**: the adapter is a smooth, bounded 2-layer MLP -- large
input perturbations degrade its output continuously. The memory read
is a softmax over cosine similarities across 8 slots
(`reference/hz0b_memory_simulator.py::read`); once injected noise
dominates the true signal in the query, the correct slot's similarity
score becomes statistically indistinguishable from the wrong/empty
slots', and softmax attention effectively snaps to a near-random
choice -- a genuine phase transition rather than graceful degradation.
This is consistent with content-addressable, similarity-based
retrieval being structurally more brittle under extreme representation
corruption than a smooth learned function, even though it is clearly
superior under realistic, moderate noise.

## What this adds to B11's real coverage

One more of the 16 named tasks done for real, with a genuinely two-
sided result: memory's advantage holds under realistic noise but is
NOT unconditional -- there is a real regime (very large representation
corruption) where the simpler baseline wins. Reported plainly rather
than only citing the favorable low-noise numbers. Not attempted this
pass: locating the exact collapse threshold between 2x and 5x more
precisely, or testing whether `ste=True` (discrete write/read
decisions) changes the collapse's sharpness.
