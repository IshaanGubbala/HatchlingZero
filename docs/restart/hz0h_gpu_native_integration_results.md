# GPU-Native BDH Forward Integration Results

Status: real CUDA correctness pass + real CUDA end-to-end training-step
benchmark, both independently downloaded through the Pi relay's `/inbox`
endpoint. **This is a real, honest negative result on the speed axis,
reported plainly rather than rounded toward what the isolated benchmarks
predicted.**

## What this is

`reference/hz0h_bdh_gpu_native_torch.py`'s `bdh_gpu_native_forward`
combines the three Stage 1 remaps validated separately this session:

- Triton attention kernel: real measured **1.551x** end-to-end (forward +
  backward + optimizer step already included in that benchmark)
- wide-GEMM encoder: real measured **1.705x**, but **forward-only**,
  frozen weights, using a cached/detached wide view (no backward pass)
- bmm encoder_v: real measured **1.509x**, also **forward-only**, frozen
  weights, no backward pass

This is the first time all three ran together in a real trainable
forward+backward pass. The encoder remap in particular had to change: a
trainable model cannot use a stale cached wide view across optimizer
steps, so `wide_encoder_step_live` rebuilds it from the *live* parameter
every forward call -- exactly the permute-every-forward cost that
component's own module docstring warned against and explicitly flagged
as untested before this integration.

## Correctness

Real CUDA run caught a genuine bug on the first attempt: `bdh_triton_attention`
checked `triton_available()` (does this *machine* have CUDA+Triton) but
never checked whether the actual input tensor was on a CUDA device. The
test's models/tensors are created without an explicit `.to(device)` call
(a normal device-agnostic test pattern) -- correctly fell back to native
attention on Mac (no CUDA), but crashed on the CUDA machine with `Pointer
argument cannot be accessed from Triton (cpu tensor?)` since
`triton_available()` was `True` regardless of where the tensors actually
lived. Fixed by also checking `Q.is_cuda` (commit `335c8f5`) -- a real,
independently-useful dispatch-robustness fix, not specific to this test.

After the fix, real end-to-end parity on CUDA:

```text
tests/reference/test_hz0h_bdh_gpu_native_torch.py: 3 passed
```

Logits, loss, and gradients on every named parameter (`encoder`,
`encoder_v`, `decoder`, `embed.weight`, `lm_head`) match the oracle.

## Real end-to-end training-step benchmark

Same production config as every other benchmark this session (batch 12,
seq 256, `n_embd=512`, `n_layer=8`, `n_head=8`, `mult=32`, bf16, RTX3060,
20 timed steps after 5 warmup, `--attention-backend gpu_native`):

```text
raw_bdh:      6,991.27 tok/s, peak memory 8,081,824,256 bytes (7.53 GiB)
gpu_native:   4,444.64 tok/s, peak memory 4,582,742,528 bytes (4.27 GiB)
matched_transformer: 73,314.14 tok/s, peak memory 942,245,376 bytes (0.90 GiB)

native_over_raw_speed_ratio:        0.636  (~1.57x SLOWER, not faster)
native_over_raw_peak_memory_ratio:  0.567  (~43% LESS memory -- real win)
```

**This reverses the sign predicted by the three isolated wins.** The most
likely real cause, matching the risk this integration's own docstring
disclosed in advance: the wide-GEMM encoder's live (non-cached) permute +
reshape has to run on both the forward AND backward pass now, at every
recurrent level, every training step -- a cost the isolated forward-only
encoder benchmark (which used a cached, detached, gradient-free wide
view) never paid. The Triton kernel alone already had a genuine
end-to-end win with backward pass included (`1.551x`, see
`docs/restart/hz0h_triton_kernel_v2_results.md`); the bmm encoder_v
remap's isolated benchmark also never included a backward pass. Combined,
whichever of these newly-backward-tested components is the dominant
cost has not yet been isolated -- that decomposition (Triton kernel alone
vs. +bmm encoder_v vs. +live encoder reshape, each with real
forward+backward+optimizer timing) is the natural next real experiment,
not yet run.

The real memory win (43% less peak memory) is genuine and independent of
the speed regression -- worth keeping regardless of what the speed
decomposition finds.

## Why this is being reported as-is, not softened

Per this project's standing zero-overclaiming discipline: the three
isolated forward-only benchmarks were real, correctly measured, and
correctly reported at the time -- they were never claimed to predict an
end-to-end number, and this integration exists specifically to test
whether they compose. They don't, at least not as currently wired. That
is itself a real, useful finding: forward-only isolated micro-benchmarks
of individual ops are not a reliable predictor of real training-step
throughput once backward-pass cost and per-step (rather than cached)
weight-layout overhead are accounted for. The Stage 1 plan's own gate
("no material quality regression... meaningful speedup") is not met by
this specific combination as built; the real memory improvement is met.
