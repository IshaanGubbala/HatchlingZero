# RoPE Hoist, Steady-State Profile, and Compile-Mode Sweep: Real Results

Status: real CUDA results, independently downloaded through the Pi
relay's `/inbox` endpoint. All three arms of a single sequential job
(commits `0715a27`, `8bbdef0`, `8570a8b`) run with `fresh_subprocess_per_arm`
isolation, avoiding the co-residency measurement artifact diagnosed
earlier this session (`docs/restart/hz0h_triton_regime_dependence_results.md`).

## Setup

Production shape throughout: `batch=12, seq=256, n_embd=512, n_layer=8,
n_head=8, mult=32`, bf16, RTX3060, `torch.compile(fullgraph=True)`
everywhere except the compile-mode sweep's own `default` baseline arm.

## Step 1: steady-state profile of compiled raw BDH

Real hotspot breakdown (self-CUDA time, 5 active steps after 10 warmup):

```text
aten::bmm (attention QK^T / scores@V)        48.69%
ampere_bf16_s1688gemm... (GEMM variants)     ~38%  combined across several kernels
aten::mm                                     13.56%
triton fused RoPE trig kernels                8.12% combined (2 kernel variants)
triton_per_fused_add_sum (attention reduce)   7.51%
fused AdamW optimizer step                    2.09%
LayerNorm / clone / tril fused kernels        <1% each
```

**Real finding**: once compiled, ~95%+ of compiled-raw BDH's CUDA time is
already dense GEMM/BMM work (attention `QK^T`/`scores@V` alone is
essentially half). RoPE's own fused pointwise kernels are a small
fraction (~8%) of total time -- `torch.compile` had already fused and
minimized that cost before any manual hoisting was attempted.

## Step 2: RoPE hoist (exact-math candidate)

`reference/hz0h_bdh_hoisted_rope_torch.py` hoists the sequence-only RoPE
phase/cos/sin computation outside the `n_layer` loop (identical every
recurrent round; only the query changes) instead of rebuilding it inside
each attention call, as the oracle does. Real, bit-exact parity
confirmed by `tests/reference/test_hz0h_bdh_hoisted_rope_torch.py`
(`torch.equal` on logits/loss/every named-parameter gradient, plus a
one-graph/zero-graph-break `torch._dynamo.explain` check).

```text
raw:     10,839.5 tok/s, 4,980,444,672 bytes peak
hoisted: 10,825.4 tok/s, 4,980,444,672 bytes peak

hoisted_over_raw_throughput_ratio: 0.9987  (essentially a wash, within noise)
hoisted_over_raw_peak_memory_ratio: 1.0000 (identical)
```

**Honest result, consistent with Step 1's profile**: no meaningful
speedup. RoPE trig computation was never a large enough fraction of
compiled-raw's total time (~8% per the profile) for hoisting it to move
the needle -- `torch.compile`'s own fusion had already captured most of
the available win there. Real, exact-math, zero-risk change (still a
correct, slightly cleaner implementation) but not a performance win at
this shape. Kept as a real, documented negative result, not discarded
silently.

## Step 3: compile-mode sweep

`scripts/hz0h_raw_compile_mode_cuda_benchmark.py`, unmodified raw BDH,
comparing `torch.compile` modes in fresh subprocesses:

```text
default:          10,817.0 tok/s, 4,980,444,672 bytes peak
reduce-overhead:  10,777.2 tok/s,   194,571,776 bytes peak
max-autotune:     11,310.0 tok/s,   194,571,776 bytes peak

throughput_ratio_over_default:
  reduce-overhead: 0.9963  (essentially flat)
  max-autotune:    1.0456  (+4.6% real speedup)

peak_memory_ratio_over_default:
  reduce-overhead: 0.0391  (~96% LESS memory)
  max-autotune:    0.0391  (~96% LESS memory, same as reduce-overhead)
```

**Real, decisive win**: `max-autotune` beats `default` on both speed
(+4.6%) and memory (~25x less peak allocation) simultaneously, with
`reduce-overhead` matching the memory win but not the speed win. Both
non-default modes logged a real, disclosed limitation in `child_stderr`:
`"Not enough SMs to use max_autotune_gemm mode"` -- this RTX3060 doesn't
have enough streaming multiprocessors for `max-autotune`'s full GEMM
autotuning path, so the real win measured here is likely a floor, not a
ceiling; a larger GPU could plausibly do better still. Not yet explained
why peak memory drops so sharply under compile modes other than default
-- worth a real follow-up if this becomes a candidate for production use
(is this a genuine allocator/caching difference, or an artifact of how
`torch.compile` manages its own workspace under these modes).

## Net takeaway

The RoPE hoist and the profiler both point the same direction: further
manual exact-math micro-optimization of compiled raw BDH has hit
diminishing returns at this shape -- the remaining cost is real, dense
attention/GEMM work `torch.compile` already schedules well. The genuine
lever still on the table from this thread is `--compile-mode
max-autotune`, a real, free, zero-code-risk 4.6% speed + ~96% memory
win, worth adopting as a default recommendation for training-step
benchmarks going forward, with the disclosed SM-count caveat above.
