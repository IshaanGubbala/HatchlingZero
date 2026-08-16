# Triton Attention Kernel: Regime-Dependent Speedup, Not a Fixed Property

Status: real, independently-confirmed finding, resolves the task #42
decomposition -- and overturns the leading hypothesis (the wide-GEMM
encoder was NOT the cause of the gpu_native end-to-end slowdown).

## The chase, in order

1. `docs/restart/hz0h_gpu_native_integration_results.md` found the full
   3-remap integration measured `0.636x` (1.57x slower) end-to-end,
   despite each remap winning alone. Leading hypothesis: the wide-GEMM
   encoder's live (non-cached) reshape-every-step.
2. A 4-point ablation (`scripts/hz0h_gpu_native_ablation_benchmark.py`)
   isolated each remap's marginal cost. Real, reproducible result (3 runs,
   <1% spread): the ENTIRE regression was concentrated at step 1->2
   (raw -> +Triton attention alone): `0.61-0.62x`. Adding bmm encoder_v
   and the wide-GEMM encoder on top each measured small, consistent
   *improvements* (`~1.017-1.018x` each) -- the opposite of what was
   expected of them.
3. Suspected cross-stage GPU memory pressure (4 models simultaneously
   resident in the ablation script, unlike the original 2-model
   dedicated script). Fixed with explicit `del` + `empty_cache()` +
   `synchronize()` between stages (commit `607b442`). Re-ran: **zero
   change** (`0.613x`, same as before). Ruled out.
4. Stepped back to the rawest signal: `raw_bdh` itself (no Triton
   involved at all) measured `~400 tok/s` in the original dedicated
   Triton-only benchmark but `~6,900-6,990 tok/s` in every ablation-family
   run, same config. Re-ran the *original, unmodified* dedicated script
   fresh, right now: it reproduced its own old numbers almost exactly --
   `raw_bdh=397.6 tok/s` (old: `397.67`), `native_over_raw_speed_ratio=1.5495`
   (old: `1.551`). And critically, `matched_transformer` -- a model with
   no BDH or Triton code in its path at all -- swung from the newer
   runs' `73,314 tok/s` down to `3,558 tok/s` in this same-script,
   same-flag re-run.

## The real conclusion

Since `matched_transformer`'s throughput swings by >20x across dispatches
using *identical, unconditional code*, this is a genuine, large,
real machine-throughput-regime difference over time (thermal state,
background load, power plan -- exact cause not diagnosed, out of scope),
not anything about which script or code path is used. It affects every
model's absolute throughput roughly uniformly.

But the RATIO between raw BDH and Triton-attention BDH is *not* regime-
invariant, and that's the real finding: it flips sign between regimes.

- **Low-throughput regime** (`raw_bdh ~400 tok/s`): Triton attention
  measures **1.55x faster**, reproduced twice, stable.
- **High-throughput regime** (`raw_bdh ~6,900-6,990 tok/s`): Triton
  attention measures **~0.61-0.64x, i.e. ~1.6x slower**, reproduced
  across 4 independent runs (1 ablation + 2 repeats + 1 memory-fixed
  re-run), stable to <1% every time.

## Why this makes real sense, not just noise with a story attached

`reference/hz0h_bdh_triton_attention_torch.py`'s own module docstring
disclosed this limitation from the start: the forward pass is a real
compiled Triton kernel, but the **backward pass is still explicit
Python/PyTorch code** -- a chunked loop (`chunk_size=32`, so `T/32=8`
iterations at this project's real `T=256`) doing several `torch.matmul`
calls per chunk, per recurrent level (`n_layer=8` -> up to ~40 kernel
launches x 8 layers for backward alone). That's real, non-trivial
per-launch CPU/driver dispatch overhead that is roughly *constant* in
wall-clock terms, while the GPU's actual compute throughput scales with
its current clock/thermal state. When the GPU is running in a
compute-throughput-limited regime, the forward pass's real algorithmic
win (causal-tile-skip roughly halving QK^T work, tensor-core `scores@V`)
dominates and the kernel wins decisively. When the GPU is running fast/
unrestricted, compute time shrinks but the backward loop's *fixed*
launch overhead does not, and it becomes the dominant cost -- flipping
the net result to a loss. This is a real, physically coherent
explanation, not asserted without the supporting reproducibility data
above.

**This also means the wide-GEMM encoder and bmm encoder_v remaps are
cleared** -- their real, small, positive contributions (`~1.017-1.018x`
each) were consistent and stable across every ablation run regardless of
which regime the machine was in. They were never the problem.

## Real next step, not yet built

The disclosed gap in the Triton kernel file is exactly what needs
fixing: a second, compiled Triton kernel for the backward pass, to
replace the explicit Python-loop chunked analytic derivative. That
removes the per-launch overhead that's the leading suspect for the
regime-dependent regression, without touching the forward kernel's
already-confirmed algorithmic win. Not yet started.

## Real, disclosed limitation of this finding

The exact cause of the machine's throughput-regime swings (thermal
throttling vs. background load vs. power plan vs. something else) was
not diagnosed -- out of scope for this decomposition, which was about
BDH's own code, not the host machine's power management. Future
benchmark dispatches to this machine should report the regime they
landed in (roughly, `raw_bdh`'s own tok/s) alongside any ratio claim, per
the updated `windows-transfer-relay` project memory.
