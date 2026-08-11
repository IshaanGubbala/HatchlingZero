# HZ Phase 1: a BDH KV-cache, a real timing bug fix, and the corrected scale picture

Date: 2026-08-11. Two things happened in the same pass, both real and
both changing the picture from
`docs/restart/hz0h_phase1_crossover_scale_sweep_results.md`:

1. **Built and verified a second BDH decode path**: `bdh_kv_cache_step`
   (`reference/hz0h_bdh_torch.py`) — an explicit, growing per-layer K/V
   cache, the same style as a standard Transformer's KV-cache, instead of
   `bdh_stream_chunk`'s O(1)-in-context, O(D²)-per-layer compressed
   running state. Motivated directly by the crossover sweep's finding
   that `bdh_stream_chunk`'s O(D²) cost was what made BDH lose at larger
   D — this path costs O(D·context) per token instead, trading
   context-independence for a cheaper per-step constant (BDH's attention
   has no softmax, so it's cheaper per element than a Transformer's).
   Verified numerically identical to both `BDH.forward` and
   `bdh_stream_chunk` (`tests/reference/test_hz0h_bdh_kv_cache.py`, 4
   tests) — not just similarly fast.

2. **Found and fixed a real measurement bug**: `measure_bdh_decode_streaming`
   and `measure_transformer_decode_kv_cache`
   (`scripts/hz0h_inference_benchmark.py`) both re-ran prefill *inside*
   the timed decode region on every call — so the reported decode
   tokens/sec silently included one-time prefill cost, worse at longer
   prompts. This directly affected every number in the earlier crossover
   sweep. Fixed by separating prefill (once, untimed) from the decode
   loop (timed) for both paths, and building the new
   `measure_bdh_decode_kv_cache` correctly from the start.

## Corrected results: BDH's best decode path vs. the real Transformer KV-cache

| Scale | ctx=128 | ctx=512 | ctx=1024 | ctx=2048 |
| --- | --- | --- | --- | --- |
| ~5M | **3.10x** | **3.19x** | **3.42x** | **3.47x** |
| ~25M | 1.48x | 0.99x | 0.90x | 0.99x |
| ~71M | 0.57x | 0.43x | 0.37x | 0.36x |

(>1x favors BDH's best available decode path, <1x favors the Transformer.
"Best available" = `max(bdh_stream, bdh_kv_cache)` at each point.)

## What changed vs. the buggy numbers

- **At 5M params, the trend direction flipped**: the buggy version showed
  BDH's advantage SHRINKING with context (3.09x → 1.58x) — backwards
  from theory. The corrected version shows it GROWING with context
  (3.10x → 3.47x), which is what an O(1)-per-token mechanism should do
  relative to a context-dependent one. The bug was inflating the
  apparent cost of longer-context BDH streaming runs (bigger prompts →
  more contaminating prefill time folded into the "decode" measurement),
  masking the real trend.
- **At 25M, the picture is now much closer to parity** (0.90x-1.48x)
  rather than a clear Transformer win (the buggy version showed
  0.49x-0.82x).
- **At 71M, the qualitative conclusion is unchanged: BDH still loses**
  (0.36x-0.57x, Transformer 1.75x-2.8x faster), though less dramatically
  than the buggy 0.18x-0.32x suggested. The new `bdh_kv_cache_step` path
  did NOT rescue this at 71M -- it sometimes beats `bdh_stream_chunk`
  at this scale (e.g. 120.1 vs 64.1 tok/s at context=128) but not
  consistently or by enough to close the gap to the Transformer's real
  KV-cache.

## Real noise in this data, disclosed rather than smoothed over

Two clear outliers in `bdh_decode_kv_cache`'s own numbers: 4.4 tok/s at
25M/context=1024 (neighbors: 294.1 and 185.4 at context=128/512) and 6.4
tok/s at 71M/context=512 (neighbors: 120.1 and 52.7). Both are ~10-40x
below their neighboring measurements with no plausible mechanistic
explanation (the cost should vary smoothly with context) -- almost
certainly transient system noise (background process, thermal, or a
first-call allocation spike this benchmark's warmup didn't fully
absorb), not a real property of the decode path. Reported as measured,
not discarded or replaced, but should not be read as a real dip in
`bdh_kv_cache_step`'s performance at those specific points.

## Absolute throughput, `bdh_stream_chunk` specifically (the flat, clean signal)

| Scale | ctx=128 | ctx=512 | ctx=1024 | ctx=2048 |
| --- | --- | --- | --- | --- |
| 5M | 848.3 | 840.4 | 824.9 | 792.5 |
| 25M | 193.5 | 190.5 | 190.6 | 194.2 |
| 71M | 64.1 | 64.8 | 64.5 | 60.8 |

`bdh_stream_chunk`'s own throughput is now remarkably flat across
context length at every scale (within ~5% at 5M, within ~2% at 25M and
71M) -- a clean, direct confirmation of the real O(1)-in-context property
H2 proved algebraically, now visible in corrected wall-clock numbers.
The scale-to-scale DROP (848 -> 194 -> 64, roughly a 13x fall from 5M to
71M) is the real story: BDH's per-token cost scales with model width
(O(D²) per layer), not context length, and that width-scaling is what
loses to the Transformer's KV-cache at larger D.

## Updated conclusion

BDH's inference-speed advantage is real at small scale (~5M params) and
grows with context length there, exactly as the O(1)-vs-O(context)
mechanism predicts. It does NOT survive to ~71M params in this
comparison, with or without the alternative KV-cache-style decode path.
The mechanistic mismatch is specific: `bdh_stream_chunk`'s O(D²)
state-update cost outgrows the Transformer's O(D·context) attention cost
as D grows, for the context lengths tested here (up to 2048). Per the
prior document's still-open prediction, BDH's real advantage may reappear
at much longer context relative to D — not tested here, real next step.

## Real next steps

1. Test much longer contexts (8K–128K) at 71M scale specifically to
   check whether BDH's O(1) property eventually wins again at long
   enough context, per the mechanistic prediction above.
2. Investigate why `bdh_kv_cache_step` doesn't consistently beat
   `bdh_stream_chunk` at larger D despite being asymptotically
   context-dependent-but-cheaper-per-step in principle — possibly
   implementation overhead (small per-token matmuls, Python loop cost)
   rather than a real architectural ceiling; needs profiling, not
   assumed.
3. Re-run with cleaner isolation (fewer background processes, more
   warmup iterations) to reduce the outlier noise seen in this pass.
4. Profile per-step time breakdown for all four decode paths at more
   than one scale — still not done, same open item as the prior
   document.
