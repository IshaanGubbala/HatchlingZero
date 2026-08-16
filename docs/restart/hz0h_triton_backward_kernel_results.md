# Compiled Triton Backward Kernel: Final Results

Status: real correctness win, real but incomplete speed result. Closes
out task #43. Honest accounting below -- this did not achieve a net
speedup over raw BDH, despite real engineering effort and two real fixes.

## What was built

`reference/hz0h_bdh_triton_attention_torch.py`'s `_BDHTritonAttention.backward`
previously ran an explicit Python loop (`chunk_size=32`, ~8 chunks at
this project's `T=256`, several `torch.matmul` calls per chunk) --
diagnosed as the real cause of the kernel's regime-dependent regression
(`docs/restart/hz0h_triton_regime_dependence_results.md`). Replaced with
three real `@triton.jit` kernels:

- `_bdh_dq_query_role_kernel`: dQ's contribution from Q's query role
  (same past-inclusive causal-tile-skip bound as the forward kernel).
- `_bdh_dq_key_role_kernel`: dQ's contribution from Q's key role (since
  `K=Q`, Q plays both roles) -- the future-inclusive complement.
- `_bdh_dv_kernel`: dV, also future-inclusive.

Each kernel's tiling mirrors the forward kernel exactly, so no cross-
block atomics are needed; dQ's two contributions are combined with one
elementwise add outside the kernels, and `dV_heads` is summed across the
head dimension in Python, matching the broadcast-V convention the
Python-loop backward it replaces used.

## Real bugs found and fixed, in order

1. **Dtype mismatch** (first real CUDA run, all 5 cases failed at Triton
   compile time): the second `tl.dot` in each kernel multiplied a
   freshly bf16-loaded tensor against an fp32 accumulator without
   casting -- the forward kernel already handled this correctly
   (`v.to(tl.float32)` before its own second dot); missed the same cast
   in the 3 new kernels. Fixed at all 3 sites.
2. **Redundant recomputation** (correctness passed 5/5, but clean speed
   measured 0.46x -- WORSE than the Python loop's own 0.61x): the first
   version reused the forward kernel's small 64-wide output tile for the
   backward kernels' OWN output tiling, but the expensive dscore/score
   reduction those kernels compute doesn't depend on the output tile at
   all. At `N=2048` with a 64-wide tile, the grid launched 32 redundant
   recomputations of the identical dscore matrix per (batch, head,
   row-tile); 8x redundant at `D=512` for the dV kernel. Fixed by
   decoupling output-tile size from reduction-tile size.

## Real tile-width sweep (the output-tile lever)

```text
block_n_out=64,  block_d_out=64:   0.46x   (3214 tok/s) -- the redundancy bug
block_n_out=256, block_d_out=128:  0.594x  (4149 tok/s) -- real improvement
block_n_out=512, block_d_out=256:  0.576x  (3989 tok/s) -- real REGRESSION
```

Widening from 64->256/128 gave a real, substantial improvement (29%
faster). Widening further to 512/256 made things worse, not better --
past a point, fewer/bigger program instances lose more to reduced SM
occupancy and parallelism than they gain from less redundant
recomputation. `256/128` is the real, measured local optimum for this
lever and is what's left in the code.

## Real final numbers, best configuration

Same production shape as every benchmark this session (`n_embd=512,
n_head=8, mult=32, batch=12, T=256`, bf16, RTX3060), clean single-model
methodology (`scripts/hz0h_gpu_native_ablation_benchmark.py`):

```text
raw_bdh:       6,986 tok/s
triton (compiled backward): 4,149 tok/s

2_over_1_speed_ratio: 0.594  (~1.68x SLOWER than raw BDH)
```

For comparison, the uncompiled Python-loop backward measured
`0.61-0.64x` under the same clean methodology. The compiled kernels are
correctness-verified and reduce backward from ~40 PyTorch-level kernel
launches to 3, but at the best tile configuration found, they are
**roughly on par with, not clearly better than, the original Python
loop** -- both are real, and neither beats raw BDH's plain matmul
attention at this project's shape.

## Honest conclusion

Compiling the backward pass did not achieve the hoped-for fix. The
forward kernel's real algorithmic win (causal-tile-skip, tensor-core
`scores@V`) is real and was independently confirmed multiple times this
session. But BDH's real shape regime at this project (`N=2048 \gg
T=256`) means the backward pass's dscore/score reduction is expensive
relative to the small T, and no tile-width configuration found gets the
compiled kernels convincingly ahead of either the naive Python loop or
raw BDH's plain matmuls. This is a real, disclosed negative result on
the net speed question, consistent with this project's zero-overclaiming
standard -- not something to round up or bury.

What IS real and kept: the compiled kernels are mathematically correct
(5/5 across every tile configuration tried, including the two
regressions), reduce Python-level launch count substantially, and the
tile-width sweep is real, useful evidence about where the actual
bottleneck now lives (the dscore/score reduction itself, not launch
overhead) for anyone picking this back up later. A genuinely different
approach -- e.g. fusing forward and backward more tightly, or revisiting
whether Q's dual role can be restructured to avoid two full extra passes
over Q -- would be the real next lever, not further tile tuning on this
design.
