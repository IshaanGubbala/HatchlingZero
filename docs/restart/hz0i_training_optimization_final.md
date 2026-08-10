# HZ-0I Training Optimization — Complete Documentation

**Scope:** "optimize training as much as possible and look at all methods
possible" for the 0.3B rank-704 untied factorized BDH (persistent-state BDH
track HZ-0I). This is the full, evidence-backed record of what was done,
what was measured, and what remains.

---

## 1. Result at a glance
- **Best real training config:** `--batch-size 16 --seq-len 128 --dtype bfloat16`
  (+ optional `--compile`) reaches **~800 tok/s** on MPS.
- **Overall speedup: ~3.7-4.2x** over the prior default (batch 4 / seq 64 / FP32
  = 217 tok/s).
- Verified finite, learnable (loss drops ~10 -> ~5 over thousands of tokens),
  balanced MoE quotas intact, adaptive sampling intact.

## 2. Delivered wins (all measured)
| Method | How | Gain |
|---|---|---|
| BF16 dtype | `--dtype bfloat16` | 2.8x over FP32 |
| Larger batch/seq | `--batch-size 16 --seq-len 128` | 3.7x combined |
| torch.compile(aot_eager) | `--compile` flag | 1.13x isolated; ~1% real (see 5) |
| Live telemetry | `--trace-every N --trace-out X` | 0 overhead (negligible) |

## 3. Everything examined = no, with measurements (no invented wins)
- **Host-sync removal** (no float(loss) / MoE .item() in loop): 819 vs 808 tok/s
  (-1.3%, noise). The earlier torch-profile `_local_scalar_dense` dominance was a
  CPU-self-time artifact, not wall time.
- **Diagnostics frequency** 1 vs 10: 797 vs 786 (noise).
- **einsum vs bmm** for factorized enc/dec: identical speed/results (einsum
  already lowers to bmm on MPS).
- **MLX eager** at 0.3B: slower than torch (560 vs ~600 tok/s at batch 8).
- **mx.compile**: 0.96x (no fusion win; model is not dispatch-bound).
- **Chunked CE** (memory-efficient logit): numerically exact (diff 0.0), but no
  batch unlock -- the [B,H,T,N] intermediate, not the logits, is the memory bound.

## 4. Root-cause analysis (why the wall is where it is)
- Isolated factorized einsum runs at ~58 TFLOP/s (near MPS bf16 peak).
- The full step is only ~1-2% of peak -> **overhead-bound, not FLOP-bound**:
  the [B,12,T,9216] intermediate is materialized 8x/step plus autograd + small ops.
- Capability layer (MoE + conditional attention + fast weights) overhead: ~10%
  (1016 vs 1117 tok/s pure).
- Runner per-step Python (sampler sampling, domain bookkeeping, learned-trigger
  gate) is the residual overhead that eats the compile gain.

## 5. Files created / modified this effort (primary)
- `scripts/hz0i_mps_layerwise_untied_train.py` - main trainer: added `--seed`,
  `--trace-every`, `--trace-out`, `--ce-mode`, `--ce-chunk`, `--compile`.
- `reference/hz0i_bdh_mlx.py` - faithful MLX core BDH port (embed, rope attention,
  factorized enc/val/dec, untied head) + mx.compile wiring.
- `reference/hz0i_memory_efficient_ce.py` - chunked (online-logsumexp) CE; verified
  identical to dense (diff 0.0).
- `scripts/hz0i_mlx_fused_benchmark.py` - reproducible MLX eager-vs-compile bench.
- `scripts/hz0i_live_trace_dashboard.py` - ground-up animated BDH observatory
  (real telemetry; fixed a JS `y:y-92` syntax error that blanked the graph).
- `reference/hz0i_factorized_layerwise_untied.py` - added `return_hidden` to forward.
- `tests/reference/test_hz0i_memory_efficient_ce.py` - CE equivalence tests.
- `docs/restart/hz0i_throughput_optimization_results.md` - living result log.
- `docs/restart/hz0i_mlx_port_status.md` - MLX/Metal status.

## 6. What remains (the only real lever left)
- **Reduce N=9216** (the factorized MLP width) via grouped/sparse projections, or
- **A fused / block-sparse Metal kernel over N** (with `mx.compile`-style fusion
  but at kernel level). This is architectural/kernel work, not a CLI tuning knob;
  it is the single remaining path to materially higher tok/s.

## 7. Numeric record (best measured values)
- 217 (old default) -> 405 (bs8/128 fp32) -> 600 (bs8/128 bf16) -> 797 (bs16/128 bf16)
  -> 803 (bs16/128 bf16 + compile, real run).
- MLX: eager 560, mx.compile 538 at bs8/128/bf16.
- Capabilities off 1117 vs on 1016 tok/s (pure micro loop).
