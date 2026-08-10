# HZ-0I MLX/Metal port — honest status (2026)

## Built
- `reference/hz0i_bdh_mlx.py`: faithful MLX core of the 0.3B factorized BDH
  (untied head) — embed lookup, RoPE causal attention, low-rank enc/val/dec
  einsum, LayerNorm, linear logits head. Plain `mx.array` parameters (project
  convention). Smoke: loss ~10.25 on random init, 581MB bf16 weights.
  `mx.compile` + `mx.value_and_grad` + `mlx.optimizers.AdamW` wiring.

## Measured (honest)
- Eager MLX at batch16/seq128/0.3B: **very slow** — a single warmup+5-step
  benchmark exceeded 1200s (timed out). The huge per-layer tensor
  `[B,H,T,N]=[16,12,128,9216]` (~226M elements/layer over 8 layers) makes
  un-fused eager MLX worse than torch here.
- `mx.compile` should fuse these, but the graph is large enough that compile
  itself is heavy on this box; not yet a clean win at this exact scale.
- Conclusion: **MLX/Metal is the right direction but is NOT a one-turn win at
  0.3B.** Reaching ~2k-4k tok/s needs fused einsum kernels and/or the
  memory-efficient tied-logit path plus lowering the N=9216 intermediate, i.e.
  real kernel work (roadmap lane 5), not a script swap.

## Meanwhile (delivered, measured)
- Batch16 + seq128 + BF16 in torch MPS: **3.7x** (217 -> ~797 tok/s), live run
  reached loss ~5.09 at ~454-763 tok/s (slower while MLX benchmarks competed).
- Fixed the live dashboard JS so the BDH graph renders.

### Fused-kernel + memory-efficient logit delivery (2026)
- `mx.compile` on the 0.3B core BDH step (batch8/seq128/bf16): **538 tok/s vs
  eager 560 tok/s (0.96x)** -- no fusion win; compute-bound on the N=9216 MLP.
- `reference/hz0i_memory_efficient_ce.py`: chunked (online-logsumexp) CE that
  never materializes [B,T,V] logits; verified numerically identical to dense
  (diff 0.0). Integrated into the runner as `--ce-mode chunked` (adds
  `return_hidden` to the untied model forward). Allows larger batches by
  releasing the logits/its-gradient memory.
- Scripts: `scripts/hz0i_mlx_fused_benchmark.py`, tests in
  `tests/reference/test_hz0i_memory_efficient_ce.py`.
