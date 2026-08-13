# HZ Next-Phase Plan Phase D1: base+delta INT8 state -- real, substantial quality win over full-every-chunk INT8; GPU throughput pending

## Setup

Per `plans/HatchlingZero_Next_Phase_Plan.md` section 8 (Phase D):
"the current issue is not that INT8 destroys quality... the issue is
repeated quantization/dequantization overhead." Phase C already
confirmed plain full-INT8 has negligible quality cost at coarse
(32/64-token) chunk granularity
(`docs/restart/hz0h_phase_c_int8_state_results.md`); this phase tests
whether a two-level `S = S_base (INT8) + delta (full precision)` design
(`reference/hz0h_bdh_vb_torch.py`'s `bdh_vb_stream_chunk_int8_base_delta_state`,
`plans/HatchlingZero_Next_Phase_Plan.md` D1) can both reduce quality
drift at FINER streaming granularity AND reduce the per-step
quantization overhead, by amortizing the quantize/dequantize cost over
`merge_every_k` tokens instead of paying it on every chunk.

Correctness verified first (`tests/reference/test_hz0h_bdh_vb_torch.py`):
`merge_every_k=1` (merges every call) is numerically identical to the
existing full-every-chunk INT8 state; `merge_every_k` effectively
infinite (never merges within the test sequence) is numerically
identical to the plain unquantized state; an intermediate K shows real,
nonzero, strictly-smaller-than-full-INT8 quantization error.

Real checkpoint used: the same VB D/4 + curriculum seed=7 `_final.pt`
(step 8139, matching the `1.6309` baseline) from Phase C.
`--stream-chunk-length 8` this time (finer than Phase C's 32/64) --
deliberately chosen to stress-test the base+delta design at the
granularity where full-INT8's overhead is worst (more chunks = more
quantize/dequantize round trips for the plain full-INT8 arm).

## Real result: quality drift, 200 real validation sequences (CPU, authoritative for quality)

| arm | validation loss | drift vs plain | drift vs full-INT8 |
|---|---|---|---|
| plain BF16/FP32 state | 1.4029590725898742 | -- | -- |
| full-INT8 (every chunk) | 1.4062856674194335 | +0.003327 | -- |
| base+delta K=8 | 1.4062856674194335 | +0.003327 | 0.0 (K=8 == chunk length, merges every call, expected exact match) |
| base+delta K=16 | 1.4039105677604675 | +0.000951 | -0.002375 |
| base+delta K=32 | 1.403194305896759 | +0.000235 | -0.003091 |
| base+delta K=64 | 1.403044846057892 | +0.0000858 | -0.003241 |

Real, monotonic, substantial reduction in quality drift as K grows:
K=64's drift vs plain (0.0000858) is **~39x smaller** than full-INT8's
own drift (0.003327) at this same finer (8-token) chunk granularity.
Note this confirms something Phase C's own coarser-granularity numbers
already implied but didn't stress: full-INT8's quality cost scales with
how often quantization happens, not just whether it happens at all --
at 8-token chunks, full-INT8's drift (0.0033) is ~15x larger than what
Phase C measured at 32/64-token chunks (0.0002/0.00009) on the SAME
checkpoint. Base+delta directly fixes this by decoupling quantization
frequency from streaming chunk granularity.

## Real result: throughput, local CPU (build-time sanity only, NOT the authoritative number)

| arm | seconds/1000 tokens | speed vs plain | speed vs full-INT8 |
|---|---|---|---|
| plain BF16/FP32 state | 2.3046 | 1.00x | -- |
| full-INT8 (every chunk) | 2.7148 | 1.178x slower | -- |
| base+delta K=8 | 3.0197 | 1.310x slower | 1.112x slower than full-INT8 (expected -- same quantization frequency as full-INT8 PLUS extra base+delta bookkeeping, no amortization benefit yet at K==chunk length) |
| base+delta K=16 | 2.6889 | 1.167x slower | 0.990x (rough parity) |
| base+delta K=32 | 2.5854 | 1.122x slower | 0.952x (~5% faster than full-INT8) |
| base+delta K=64 | 2.5277 | 1.097x slower | 0.931x (~7% faster than full-INT8) |

Real qualitative crossover on CPU: base+delta starts out slower than
full-INT8 at K=8 (no amortization, pure extra overhead), reaches parity
around K=16, and becomes progressively faster than full-INT8 as K
grows further -- exactly the shape the design predicts. **Not the
authoritative throughput number** -- Phase D's real "decode throughput"
gate is about production CUDA decode latency, matching every other
throughput number in this plan (all measured on the RTX3060). CPU
timing here confirms the mechanism works as designed and is useful for
catching implementation bugs before spending GPU time, but the absolute
magnitude of the speedup (or whether it holds at all) needs the real
GPU measurement.

## Status: quality result is solid and real; GPU throughput dispatched, not yet returned

Real GPU K-sweep requested from Windows
(`hz0h_phase_d_gpu_throughput_request.txt`, same script with
`--device cuda`) -- not yet returned as of this writing. This doc will
be updated once that lands; not applying the plan's D2 promotion gate
("real RAM savings, acceptable quality, no major decode regression")
until the real decode-throughput number is in hand. Quality alone
already clears its half of the gate cleanly.
