# HZ Phase 1: does BDH's decode advantage survive scaling up? Real, surprising answer: no, it reverses

## ⚠️ Correction: the numbers below have a real measurement bug — see `docs/restart/hz0h_phase1_kv_cache_bdh_results.md`

`measure_bdh_decode_streaming` and `measure_transformer_decode_kv_cache`
(`scripts/hz0h_inference_benchmark.py`) both re-ran prefill INSIDE the
timed decode region on every call, silently folding one-time prefill
cost into the reported decode tokens/sec — worse at longer prompts. This
affected every number below. Fixed 2026-08-11, same day; the corrected
sweep (plus a new alternative BDH KV-cache-style decode path) is in
`docs/restart/hz0h_phase1_kv_cache_bdh_results.md`. The QUALITATIVE
conclusion (BDH's advantage does not survive to 71M params) still holds
in the corrected data, but several specific numbers and the 5M-scale
trend DIRECTION changed (the buggy version showed the advantage
shrinking with context; the corrected version shows it growing with
context, which is what the underlying O(1)-vs-O(context) mechanism
predicts). Read the corrected document for the current picture.

Date: 2026-08-11. Follow-up to `docs/restart/hz0h_phase1_inference_benchmark_results.md`'s
open question: at ~4.8M params, BDH's streaming decode beat the real
KV-cached Transformer baseline ~2.9-3.2x, but that KV-cache was *itself*
slower than naive replay at that tiny scale (a real small-model
overhead effect) — so it was unclear whether BDH's advantage was a real
architectural property or an artifact of comparing against an
underperforming baseline. This sweeps model scale (matched configs,
`scripts/hz0h_inference_benchmark.py`, Mac MPS, random/untrained
weights, batch=1, greedy decode) to find out.

## Matched configs

Byte-level vocab (256) throughout, so embedding size stays negligible at
every scale (BDH's untied embed+lm_head don't dominate its budget the
way they would with a larger subword vocab). BDH's shared/tied
`encoder`/`encoder_v`/`decoder` don't scale with depth, so `n_layer` was
matched to the Transformer's `num_layers` for compute-depth parity, and
`mlp_internal_dim_multiplier` (m) was solved per scale to match total
parameter count (body params = `3*m*D^2`, independent of `n_layer`).

| Scale | D | layers | heads | Transformer d_ff | BDH m | Transformer params | BDH params | Match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ~5M | 256 | 6 | 4 | 683 | 24 | 4,804,868 | 4,849,664 | 0.9% |
| ~25M | 512 | 8 | 8 | 1365 | 32 | 25,343,824 | 25,427,968 | 0.3% |
| ~71M | 768 | 10 | 12 | 2048 | 40 | 71,070,976 | 71,172,096 | 0.1% |

## Results: BDH streaming decode vs. real Transformer KV-cache

| Scale | ctx=128 | ctx=512 | ctx=1024 | ctx=2048 |
| --- | --- | --- | --- | --- |
| ~5M | **3.09x** | **2.95x** | **2.37x** | **1.58x** |
| ~25M | 0.82x | 0.70x | 0.61x | 0.49x |
| ~71M | 0.32x | 0.29x | 0.24x | 0.18x |

(>1x favors BDH, <1x favors the Transformer.) The trend is monotonic and
consistent, not noisy: BDH's relative position gets steadily worse at
every scale step, at every context length tested.

## Absolute throughput, all three scales (tok/s, decode)

| Scale | ctx | BDH streaming | Transformer KV-cache |
| --- | --- | --- | --- |
| 5M | 128 | 829.3 | 268.3 |
| 5M | 2048 | 342.2 | 216.9 |
| 25M | 128 | 179.8 | 220.0 |
| 25M | 2048 | 71.5 | 147.3 |
| 71M | 128 | 59.4 | 183.3 |
| 71M | 2048 | 22.1 | 123.9 |

BDH's own absolute throughput collapses fast with scale (829 -> 180 -> 59
tok/s at ctx=128, roughly a 14x drop from 5M to 71M params) while the
Transformer KV-cache's absolute throughput barely moves (268 -> 220 ->
183 tok/s over the same range, ~1.5x drop). This is the real mechanism
behind the reversal.

## What this shows: a real, monotonic reversal, not a fluke

**BDH's decode advantage at 5M params does NOT survive scaling up — it
reverses cleanly and gets steadily worse at every scale step tested
(5M -> 25M -> 71M), at every context length.** This is the opposite of
what the smaller pilot suggested and directly complicates the one clean
positive result found so far in this investigation.

**Why, mechanistically**: BDH's per-layer streaming state is shaped
`(nh, N, D)` with `N = m*D/nh`, so each token's state UPDATE cost scales
with `D^2` per layer (independent of context length -- this is real,
this is the O(1)-in-context property H2 proved and this benchmark
confirms holds). The Transformer's KV-cache attention step scales with
`D * context_length` per layer -- context-length-dependent (which is
exactly what BDH's state avoids), but its per-token, fixed-`D` cost is
comparatively cheaper, especially once `D` itself grows. At D=256 (5M
scale) BDH's `D^2` term is small enough that its context-independence
wins outright even against a modest context. At D=768 (71M scale),
BDH's own per-step cost has grown enough that it loses even at the
SHORTEST context tested (128 tokens) -- meaning at this scale, BDH's
"O(1) in context" property no longer matters because its constant factor
itself became too large relative to the Transformer's per-token cost.

**This means BDH's real inference advantage, if it exists at all, is
likely NOT "smaller D, longer context" but something closer to "very
long context relative to D" -- i.e. the O(1)-vs-O(context) crossover
point itself scales with `D`.** At larger D, the context length needed
for BDH's state-reuse to pay for itself is presumably much longer than
2048 tokens (not tested here). This is a real, testable, falsifiable
prediction for future work, not yet confirmed.

**Standing caveat, same as the smaller-scale result**: this still can't
fully separate "architectural FLOP cost" from "this specific PyTorch
implementation's per-step overhead" for either path (see
`docs/restart/hz0h_phase1_inference_benchmark_results.md`'s closing
note) -- profiling where each path's time actually goes remains real,
undone work. The `D^2`-vs-`D*context` mechanism above is a plausible,
consistent explanation for the observed monotonic trend, not yet
independently confirmed via profiling.

## Real next steps

1. Test much longer contexts (8K-128K) at the 71M scale specifically --
   the mechanistic account above predicts BDH's crossover point should
   exist somewhere, just further out than 2048 tokens at this D. Confirm
   or refute directly.
2. Profile per-step time breakdown (Python loop overhead vs. tensor-op
   time vs. cache-management cost) for both paths, at more than one
   scale, to separate real architectural cost from implementation
   overhead -- especially important given the `D^2` mechanistic story
   above is currently just a plausible account of aggregate numbers, not
   a profiled confirmation.
3. Repeat with trained (not random) weights, in case activation
   sparsity from real training changes the picture (BDH's ReLU sparsity
   was measured real and substantial in the Phase 1 training-metrics
   work, but was not connected to this inference benchmark).
4. CUDA energy sampling (already built, not yet run against these
   configs) to see whether joules/token tells a different story than
   tokens/sec at any of these scales.
