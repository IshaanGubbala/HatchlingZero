# HZ-0C C8: Metal Backward Dispatch and Python-Reference Parity

Date: 2026-08-04. Closes two of the tracker's four remaining C8 open items
("Metal backward dispatch" and "Python-reference machine-readable parity
report"); "grouped/cache-optimized dispatch" and "model-level integration"
remain open.

## Metal backward dispatch

`restart/hz0a_pmetal/crates/hz0a-pmetal-gpu/src/lib.rs`:
`MetalConditionalAnchorAttentionBackward`, a real Metal-dispatched port of
the already finite-difference-verified CPU backward
(`conditional_anchor_attention_backward_f32`). Deliberately
SINGLE-THREADED (one grid thread looping over batch/token/head internally)
rather than parallel-per-batch-with-atomics: the shared weight/bias
gradient buffers accumulate across every token, so real parallelism needs
either float atomics (MSL support varies by GPU family/OS) or a
partial-buffer reduction pass -- both real, explicitly deferred to the
SAME "grouped/cache-optimized dispatch" item the forward kernel already
named as open, not attempted here. Correctness-first, matching this
project's standing discipline (the CPU tensor crate was proven before any
GPU dispatch was attempted at all).

Two new Rust tests, both passing, in `hz0a-pmetal-gpu`:
- `conditional_attention_backward_gpu_matches_cpu_reference` (single
  batch)
- `conditional_attention_backward_gpu_matches_cpu_for_multi_batch_shared_weights`
  (two batches sharing the same qkv/out weight buffers -- exercises real
  cross-batch accumulation into `grad_qkv_weight`/`grad_out_weight`)

Max diffs across all 5 gradient families (`grad_x`, `grad_qkv_weight`,
`grad_qkv_bias`, `grad_out_weight`, `grad_out_bias`) are below `1e-3` in
both tests.

## Python-reference machine-readable parity (real bug found and fixed)

`scripts/hz0c_generate_pmetal_conditional_attention_fixture.py` dumps
random inputs/weights and `masked_anchor_attention`'s actual forward
output plus `mx.grad`-computed gradients (the Python reference, not just
Rust-vs-Rust) to
`restart/hz0a_pmetal/crates/hz0a-pmetal-kernel/tests/fixtures/conditional_attention_parity.json`.
`tests/parity_with_python_reference.rs` (new, in `hz0a-pmetal-kernel`)
loads it and checks the Rust CPU forward/backward against Python exactly
-- the first real cross-language check for this kernel (all prior tests
only checked Rust-vs-Rust CPU/GPU parity, which cannot catch a divergence
present in BOTH Rust implementations).

**This found a real bug on the first run**: both the CPU and GPU forward
kernels computed `output = out_bias + attention @ out_weight` for a
triggered token and just `out_bias` (unscaled) for a non-triggered one --
never multiplying by the trigger value at all. The Python reference
instead computes `out = (attention @ out_weight + out_bias) * trigger`,
scaling the WHOLE output (including `out_bias`) by trigger. For a purely
binary trigger this only matters when `out_bias != 0`: a non-triggered
position leaked `out_bias` through instead of producing exact zero. Every
prior Rust-only test used either `out_bias = 0` or a construction that
happened not to expose it (one existing test,
`conditional_attention_matches_scalar_causal_reference_and_isolates_batches`,
had explicitly hard-coded the WRONG value, `0.25` instead of `0.0`, as its
expected output for a non-triggered position with `out_bias = 0.25` --
corrected here, with the history kept in the test's own comment rather
than silently changed).

**Fixed in four places**, keeping CPU and GPU, forward and backward, all
consistent:
1. `conditional_anchor_attention_f32` (CPU forward): final output now
   multiplied by `trigger[token]`.
2. `conditional_anchor_attention_backward_f32` (CPU backward): every use
   of `grad_output[token]` now pre-scaled by `trigger[token]` (chain rule
   through the elementwise trigger multiply, trigger treated as a
   constant per this function's own existing contract). The backward's
   own dead `if trigger>0.0 {0.0} else {0.0}` branch (always zero either
   way) was removed as part of this fix, not left in place.
3. `CONDITIONAL_ATTENTION_SOURCE` (GPU forward Metal kernel): same final
   multiply.
4. `CONDITIONAL_ATTENTION_BACKWARD_SOURCE` (GPU backward Metal kernel,
   new this pass): same trigger pre-scaling, ported directly from the
   corrected CPU version rather than the original buggy one.

All existing finite-difference and GPU-vs-CPU parity tests still pass
after the fix (internal self-consistency was never broken -- a
self-consistent forward+backward pair can still be wrong relative to an
external reference, which is exactly why this fixture-based check matters
and the earlier Rust-only tests could not have caught this).

## A second, real, but out-of-scope divergence -- documented, not "fixed"

Trying a genuinely FRACTIONAL trigger value (`0.35`, not just `0.0`/`1.0`)
in the fixture surfaced a second, different divergence (~0.018 max
output diff, well outside the `1e-3` tolerance): Python's key-masking
uses additive `(1 - trigger) * -1e9` before softmax (needed so gradients
flow through a soft trigger during training), while Rust's kernels use a
hard skip (`if trigger[source] <= 0.0 { continue; }`) that treats ANY
positive trigger as a fully-visible key with no partial suppression.

This is real but was NOT fixed, and the fixture was reverted to
binary-only triggers rather than loosening the tolerance to paper over
it. Reasoning: PMetal's entire purpose is the sparse, skip-non-triggered
positions INFERENCE path -- this project's own Hard Constraint
("Inference triggering must be deterministic and reproducible") means
real deployed triggers are always discretized before reaching this
kernel, which is exactly what makes the skip-based optimization valid and
worth having. Matching Python's soft-masking exactly would require
computing every causal key's score unconditionally (losing the sparse
compute savings that are the whole point of C8) to serve a training-time
differentiability property PMetal was never meant to reproduce. Documented
explicitly in `conditional_anchor_attention_f32`'s own doc comment as a
binary-trigger-only contract, so a future caller does not assume
soft-trigger parity that was checked and found not to hold.

## What remains open for C8

## Follow-up optimization: fused QKV projection

The scalar per-source projection hotspot was split into a dedicated Metal
QKV projection pass, so each token's Q/K/V projections are computed once and
the attention pass reuses them. The change passed all GPU forward/backward,
decode, full-block, and AdamW tests.

Release benchmark at sequence length 128:

| Trigger rate | Before (ms) | After (ms) |
|---:|---:|---:|
| 0% | 5.8760 | 2.0861 |
| 15% | 6.8956 | 1.8764 |
| 100% | 33.7290 | 1.8608 |

This is approximately 2.8x, 3.7x, and 18.1x faster respectively. The
remaining gap to MLX is a separate dispatch/kernel-efficiency question.

- **Grouped/cache-optimized dispatch**: the forward kernel is one thread
  per output element; the backward kernel is fully single-threaded. Real
  parallelization (per-batch/per-head with atomics or a reduction pass)
  is unattempted.
- **Model-level integration**: nothing in the actual Python training/eval
  loop (`scripts/hz0c_c6_conditional_attention_eval.py` etc.) dispatches
  to these Rust/Metal kernels yet -- they are proven correct in isolation,
  not wired into the real forward pass. No Python-Rust FFI/binding
  mechanism (e.g. PyO3) exists in this repo yet for that; the fixture
  approach here (fixed values dumped to JSON, replayed in Rust) is a
  proof of numerical correctness, not a live integration path.
