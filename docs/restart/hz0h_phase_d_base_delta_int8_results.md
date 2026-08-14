# HZ Next-Phase Plan Phase D1: base+delta INT8 state -- real, substantial quality win over full-every-chunk INT8; GPU throughput pending

**CORRECTION (read this before trusting anything below the Setup section)**:
every throughput/memory number in this doc, including the "real GPU"
RTX3060 numbers, was measured under a real, disclosed bug --
`load_bdh_vb` never actually cast the model to bf16 despite loading a
bf16-trained checkpoint (`.to(device)` only moves device, not dtype;
`load_state_dict` silently preserves the target tensor's own existing
dtype, fp32 by default). Every number labeled "plain_bf16_fp32_state"
below was actually measured in FP32, not BF16 -- the label's own
hedge ("BF16/FP32") was more honest than the surrounding prose, which
repeatedly treated it as the real BF16-Speed-mode baseline. This
directly affects Phase E's locked HZ-Core-2 definition, which cites
this doc's throughput conclusion. Both the root cause (a real dtype
bug in `reference/hz0h_bdh_vb_torch.py` itself -- delta was hardcoded
to float32, contradicting the plan's own "BF16 delta" spec, which
would have crashed on a genuinely bf16-cast model) and this script's
own missing dtype cast are now fixed. See the "Real correction" section
at the end of this doc for the full account and, once available, the
re-verified real-bf16 numbers.

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

## Real GPU throughput result (authoritative, RTX3060)

Same checkpoint (`_final.pt`, step 8139, confirmed correct file), same
sweep, `--device cuda`, real CUDA-synchronized timing:

| arm | validation loss | quality drift vs plain | seconds/1000 tokens | speed vs plain | speed vs full-INT8 |
|---|---|---|---|---|---|
| plain BF16/FP32 state | 1.40296 | -- | 0.8923 | 1.00x | -- |
| full-INT8 (every chunk) | 1.40627 | +0.00331 | 1.2421 | 1.392x slower | -- |
| base+delta K=8 | 1.40627 | +0.00331 | 1.4128 | 1.583x slower | **1.137x SLOWER than full-INT8** |
| base+delta K=16 | 1.40390 | +0.00094 | 1.2236 | 1.371x slower | 0.985x (~1.5% faster) |
| base+delta K=32 | 1.40319 | +0.00023 | 1.1299 | 1.266x slower | 0.910x (~9.0% faster) |
| base+delta K=64 | 1.40305 | +0.00009 | 1.0830 | 1.214x slower | 0.872x (~12.8% faster) |

(`speed_ratio` is a TIME ratio -- `>1` means slower, `<1` means faster;
noting this explicitly since it reads backwards from a naive
tokens/sec-style "bigger is better" intuition.)

**Real surprise, not smoothed over**: at K=8, base+delta is 13.7%
*slower* than full-INT8, the opposite of the amortization hypothesis at
the smallest K tested -- because K=8 equals `stream_chunk_length`
itself, meaning every chunk still triggers a merge (zero actual
amortization), while base+delta pays real EXTRA bookkeeping cost
(tracking base and delta separately, one more tensor add per read) on
top of the same quantization frequency as full-INT8. The correctness
tests' K=1 boundary case established "K <= chunk length behaves like
full-INT8 in output," but didn't capture that it's strictly slower
there in wall-clock terms, not merely equal. Real amortization benefit
only appears once K exceeds the chunk length: parity around K=16, real
wins by K=32/64 (9.0% / 12.8% faster than full-INT8), monotonically
improving across every K tested.

**The bigger, more important finding**: even at the best K tested
(K=64), base+delta is still 21.4% SLOWER than plain (unquantized) BF16
state, and full-INT8 itself is 39.2% slower. **Neither INT8 variant is
faster than not quantizing at all, on this hardware, in this
(unfused) implementation.** Quality drift is genuinely negligible
across the board (all <0.0034 vs plain, consistent with Phase C), but
decode throughput is a real, substantial cost for both INT8 arms
relative to plain state -- not what a naive reading of "0% quality
degradation" might suggest about performance.

## Verdict: base+delta promoted as the memory-mode INT8 implementation; INT8-vs-plain speed tradeoff is real and accepted, not solved

Applying plan section 8's D2 gate ("real RAM savings; acceptable
quality; no major decode regression relative to the BF16-VB state")
honestly: quality clears cleanly (negligible drift, both variants).
RAM savings are real (INT8 state is 4x smaller than FP32/2x smaller
than BF16, established in Phase C). Decode regression relative to
plain BF16-VB state is NOT negligible for either INT8 variant (21-39%
slower) -- this is exactly why the plan's own Phase C already split
"HZ-Speed mode" (BF16 state, optimized for throughput/latency) from
"HZ-Memory mode" (INT8 state, optimized for total RAM/long-context
footprint), explicitly NOT claiming INT8 would match BF16 speed. Read
that way, this decode-throughput cost is the accepted, disclosed price
of choosing Memory mode over Speed mode in the first place, not a new
problem D1 was supposed to solve -- D1's actual job (per the plan's own
framing, "the issue is repeated quantization/dequantization overhead")
was to make INT8 CHEAPER THAN naive full-every-chunk INT8, which it
does, monotonically, once K exceeds the streaming chunk length.

**Recommendation: promote base+delta (K>=32) as the canonical
HZ-Memory-mode INT8 state implementation**, replacing naive
full-every-chunk quantization -- it strictly dominates full-INT8 on
both quality (up to ~39x less drift) and throughput (up to ~13% faster)
at the K values tested, with no real downside found. Do NOT claim INT8
state (of either kind) as a throughput win over BF16 state -- it isn't,
and pretending otherwise would misrepresent the real tradeoff to anyone
choosing between Speed and Memory mode later. Per plan D2's "secondary"
fused path (reading quantized state directly without materializing a
full-precision copy) -- not built here, remains a real, disclosed
option if closing the remaining BF16 gap further ever becomes a
priority; not pursued now since the plan explicitly calls it secondary
to the base+delta design and the current result already satisfies a
real Memory-mode use case as-is.

## Real correction (found while extending Phase F's inference benchmark, not this doc's own investigation)

**How this was found**: while adding real `--dtype` handling to
`scripts/hz0h_inference_benchmark.py` (a separate, later Phase F task),
casting a genuinely bf16-loaded model produced a real `RuntimeError`
(`expected m1 and m2 to have the same dtype`) inside
`bdh_vb_stream_chunk_int8_base_delta_state`'s `QR @ prefix_state`. That
error could only happen if the model was ACTUALLY bf16 -- which meant
every prior real run of this exact mechanism (including this doc's own
Windows-dispatched RTX3060 K-sweep) must never have hit it because the
model was, despite every appearance otherwise, still float32 the whole
time.

**Root causes, both real, both fixed**:

1. `reference/hz0h_bdh_vb_torch.py`'s `init_bdh_vb_states_int8_base_delta`
   hardcoded the delta component to `torch.float32` regardless of the
   model's actual dtype -- directly contradicting the plan's own D1
   spec ("`delta = small BF16 recent-update state`",
   `plans/HatchlingZero_Next_Phase_Plan.md` section 8). The companion
   full-INT8 function (`bdh_vb_stream_chunk_int8_state`) had the same
   underlying issue: `dequantize_state_int8` always returns float32 by
   design (the scale multiply is done in fp32 for numerical safety),
   and that fp32 result was combined directly with a real bf16 `QR` in
   the matmul with no cast in between. Fixed: delta now follows the
   model's real working dtype
   (`model.encoder.dtype`), and the dequantized state is explicitly
   cast to match `QR`'s dtype right before use, in both functions. A
   new regression test (`test_base_delta_works_on_a_genuinely_bf16_cast_model`,
   `tests/reference/test_hz0h_bdh_vb_torch.py`) casts a tiny model to
   real bf16 and confirms the full streaming call succeeds with correct
   dtypes throughout -- the test that should have existed from the
   start and would have caught this immediately.
2. This script's own `load_bdh_vb` only ever did `.to(device)` -- never
   `.to(device=..., dtype=...)` -- so even after fix (1), nothing would
   have changed: `load_state_dict` preserves the target tensor's own
   existing dtype (float32, from construction) when loading weights
   into it, regardless of what dtype those weights were originally
   saved in. Fixed: added an explicit `--dtype` flag (default
   `bfloat16`, matching real training/deployment precision) and an
   actual `.to(device=device, dtype=dtype)` cast.

**Why neither bug was caught until now**: no test in this project's
suite ever ran the base+delta or full-INT8 VB streaming functions
against a genuinely bf16-cast model before today -- every existing
correctness test used the framework's fp32 default, and every real GPU
dispatch script (Phase C's `hz0h_phase_c_int8_state_quality_eval.py`
too, checked directly -- also has no dtype handling at all, runs
CPU-only fp32 throughout, though that script's OWN framing already
correctly said "FP32 state" rather than claiming BF16, so it does not
need the same retraction) had the same "only moves device, never casts
dtype" gap. The bug and the blind spot that hid it were the same root
cause.

**What this means for the numbers already in this doc**: the RELATIVE
comparisons (base+delta beats full-INT8 on both quality and speed;
quality drift shrinks and speed improves as K grows) were measured
consistently across all arms under the same accidental FP32 precision,
so the qualitative conclusions are likely still directionally correct
-- FP32 has more headroom before quantization error dominates, so
absolute drift numbers were probably measured SMALLER than real BF16
deployment would show, and FP32 compute is real slower per-FLOP than
BF16 on tensor-core hardware, so absolute throughput numbers
(seconds/1000 tokens, the "21-39% slower than BF16" headline claim)
are NOT the real BF16-vs-BF16 comparison the "do not claim INT8 as a
throughput win over BF16" recommendation above was actually based on --
that recommendation compared BF16-labeled-but-actually-FP32 numbers
against other FP32 numbers, not real BF16 numbers at all. **Not
retracting the qualitative recommendation** (promote base+delta over
naive full-INT8) since that comparison stayed internally consistent,
but **the specific magnitude numbers in the "Real GPU result" section
above need re-verification at genuine BF16** before being cited as
authoritative -- a real BF16 rerun has been dispatched to Windows;
this doc will be updated with the corrected numbers once it returns.
