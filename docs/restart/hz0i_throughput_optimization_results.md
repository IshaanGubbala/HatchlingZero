# HZ-0I throughput optimization (2026)

Swept methods on the 0.3B rank-704 untied factorized BDH (MPS, same model/task,
one config at a time, 10-step probes, finite).

| Config | tok/s | note |
|---|---|---|
| batch 4, seq 64, FP32 (prior default) | 217 | baseline |
| batch 8, seq 128, FP32 | 405 | larger batch util |
| batch 8, seq 128, BF16 | 600 | dtype halves mem / fast matmul |
| **batch 16, seq 128, BF16** | **797** | **memory ceiling sweet spot** |
| batch 32, seq 128, BF16 | 40 | unified-memory thrash |
| batch 16, seq 160, BF16 | 421 | past ceiling, thrash begins |
| batch 8, seq 255, FP32 | 25 | thrash (huge MLP activations) |
| batch 16, seq 128, BF16, diag-every 10 | 786 | diagnostics not the bottleneck |

## Findings
- Bottleneck is **memory/arithmetic intensity from the wide MLP**, so the GPU is
  under-fed by tiny tiny batches. Raising tokens/step to ~2048 is the main lever.
- **BF16 is finite** for this model (fast-weight path is safe in BF16; FP16 is
  not without grad scaling) and roughly doubles matmul throughput on MPS.
- Diagnostics frequency and tracing (`--trace-every 20`) are negligible.
- Best live config: `--batch-size 16 --seq-len 128 --dtype bfloat16` ≈ **3.7x**
  over the old default, real run at ~670-800 tok/s, loss dropping, finite.

## Still-open larger levers (roadmap lane 5)
- MLX port of the 0.3B BDH + `mx.compile` (measured ~1.9-2.3x fused on toy oracle).
- Metal kernel fusion / grouped projections / fused state updates (pmetal-style).
- These are what produced the ~2k-4k tok/s remembered values, but only at ~10M
  toy scale so far; porting to 0.3B is the remaining work, not yet done.

### Final method sweep — negative results matter too (measured, batch16/seq128/BF16)
- Host-sync removal (no float(loss)/MoE `.item()` in loop): 819 vs 808 tok/s (-1.3%, noise).
  The earlier profile's `_local_scalar_dense` dominance was a profiler CPU-self-time
  artifact, not wall time; the step is GPU-compute-bound.
- Diagnostics frequency 1 vs 10: 797 vs 786 (noise).
- torch.compile: no MPS gain (AOT-eager tested slower historically).
- MLX eager at 0.3B: >1200 s for 8 steps (unfused, worse than torch).
- mx.compile: graph too large for a quick compile win at this scale; needs fused
  einsum kernels + memory-efficient tied-logit path (roadmap lane 5, next project).

**Concluded ceiling for the exact 0.3B model on this MPS box: ~800 tok/s
(compute-bound).** The 3.7x (batch16/seq128/BF16) is the delivered win;
further gains require fused Metal kernels, which change the codebase stack, not
tuning knobs.

### Chunked-CE batch-unlock test (final)
- `--ce-mode chunked` at batch20/seq128 (2560 tok/step): 403 tok/s -- slower than
  batch16 dense (797). The logit memory freed by chunked CE is NOT the binding
  constraint; the [B,H,T,N]=[B,12,T,9216] factorized-MLP intermediate is.
  Conclusion: chunked CE is correct and available, but on THIS model the memory
  ceiling is the N intermediate, so it does not raise the batch ceiling.

### Complete method matrix (all examined, 2026)
| method | result |
|---|---|
| BF16 | 2.8x |
| batch16+seq128 (BF16) | 3.7x total (217->797) |
| host-sync removal | no effect (compute-bound) |
| diagnostics/trace freq | no effect |
| torch.compile | no MPS gain |
| MLX eager 0.3B | slower (560 vs torch 600 at b8) |
| mx.compile | 0.96x (no win) |
| chunked-CE memory path | correct (diff 0.0); no batch unlock (N-bound) |
| bigger batch (>2048 tok/step) | memory thrash (N intermediate) |
| block-sparse Metal kernel over N | NOT built; the only genuine remaining lever |

### Overhead analysis (final, batch16/seq128/BF16 MPS)
- Isolated factorized einsum: ~58 TFLOP/s (near MPS peak) -> the big op is fast.
- Full step: ~0.7-2% of peak -> OVERHEAD-bound, not FLOP-bound: DAG of small ops,
  autograd, and the [B,H,T,N] intermediate materialized 8x/step.
- Capability layer (MoE+conditional+fast) overhead: ~10% (1016 vs 1117 tok/s).
- Runner-level (sampler bookkeeping, trace, diagnostics): ~10-25%.
- Practical torch MPS ceiling for this model: ~1000-1100 tok/s pure; ~800 live.

Remaining real levers are architectural (reduce N=9216 intermediate / block-sparse
Metal kernel), not tuning knobs.

- torch.compile(aot_eager) on MPS: **1.13x** (1122 vs 994 tok/s, batch16/seq128/BF16).
  Graph breaks in conditional-attention/MoE control flow limit the gain; still a
  real, landable speedup. Best combined config would be BF16 + batch16/seq128 +
  torch.compile(aot_eager).

### torch.compile in the real trainer (final)
- Landed as `--compile` (aot_eager). Micro-benchmark: 1.13x. Real run: 803 vs 797
  tok/s (~1%) -- the isolated gain is eaten by (a) per-step Python overhead in the
  runner (sampler sampling, per-domain bookkeeping) and (b) graph breaks in the
  learned-trigger/MoE path (the runner passes triggers=None so the trigger gate
  computes each step).

### Consolidated ceiling (this hardware, torch MPS)
Best real-run config = BF16 + batch16/seq128 + (optionally --compile) ≈ **800 tok/s
(~3.7x over baseline)**. The residual overhead is Python bookkeeping and the
conditional-compute graph, which are architectural, not tunable from the CLI.
Remaining genuine lever = reduce N=9216 (grouped/sparse projections) or a
block-sparse Metal kernel over N.

- Gradient checkpointing: negative (recompute overcomes batch gain). b8=706,
  b24=671, b32=625 tok/s -- all below plain b16 (~1100 core / 800 real). Memory
  frees but does not convert to throughput; batch16 plain stays best.
