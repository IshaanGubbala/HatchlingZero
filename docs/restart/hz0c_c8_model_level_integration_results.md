# HZ-0C C8: Model-Level Integration (Real Python<->Rust FFI Bridge)

Date: 2026-08-04. Closes the "model-level integration" gap named
repeatedly (`docs/restart/hz0c_c8_pmetal_attention_results.md`,
`docs/restart/hz0c_c8_pmetal_backward_and_parity_results.md`,
`docs/restart/hz0c_c9_end_to_end_report_results.md`): PMetal's kernels
were correctness-proven in isolation (Rust-vs-Rust, Rust-vs-Python
fixture) but nothing in the actual Python model/eval path could ever
call them -- no Python<->Rust FFI mechanism existed anywhere in this
repo. It now does, and its real performance was measured honestly
rather than assumed.

## What was built

`restart/hz0a_pmetal/crates/hz0a-pmetal-bridge` (previously a 44-line
scaffold with no real functionality) now builds as a real `cdylib`,
exposing `hz0c_conditional_attention_forward`/`_backward`: flat-pointer C
ABI functions wrapping the SAME
`conditional_anchor_attention_f32`/`_backward_f32` CPU kernel already
verified against the real Python `masked_anchor_attention` reference
(`hz0a-pmetal-kernel/tests/parity_with_python_reference.rs`). 4 new Rust
tests call the `extern "C"` functions directly (including a rejected-
invalid-shape case) and all pass.

`reference/hz0c_pmetal_bridge.py`: `ctypes`-based Python wrapper --
dependency-free (no PyO3/maturin), matching this project's dependency-
conscious convention on the Rust side. **A real safety issue was found
and fixed before it shipped**: the Rust side trusts the caller's declared
`batch`/`seq`/`dim`/`heads` and reads exactly that many elements from
each raw pointer (a bare `*const f32` carries no length of its own) --
an undersized NumPy array would have been an out-of-bounds read
(undefined behavior, likely a segfault) at the FFI boundary, not a clean
error. Added `_validate_shapes` in Python to check every array's shape
BEFORE any pointer crosses into Rust, so a mismatch raises a normal
`ValueError` instead. Found this by writing the test for it first and
noticing the planned assertion (`pytest.raises(RuntimeError)`, expecting
the Rust side to catch it) was actually testing for undefined behavior,
not a real error path -- caught before running, not after a crash.

`tests/reference/test_hz0c_pmetal_bridge.py` (3 tests, skips cleanly if
the cdylib isn't built): reuses the EXISTING
`conditional_attention_parity.json` fixture (real Python `mx.grad`-computed
ground truth) to check the full path -- Python NumPy arrays, through
`ctypes`, through the Rust FFI function, through the same CPU kernel,
back through `ctypes`, compared against real MLX output. All 3 pass.

## Honest performance result: the mechanism works, the kernel does not win yet

`scripts/hz0c_c8_ffi_latency_benchmark.py` measures the FFI-dispatched
Rust kernel against the MLX Python reference directly, at the model's
real `dim=768`/`heads=12` shape, across three trigger rates and two
sequence lengths:

| Shape | Seq | Rate | MLX mean | FFI mean | FFI vs MLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| C3-scenario scale | 40 | 0% | 0.99ms | 34.9ms | **0.028x (35x slower)** |
| C3-scenario scale | 40 | 15% | 0.61ms | 35.0ms | **0.017x (58x slower)** |
| C3-scenario scale | 40 | 100% | 0.64ms | 34.9ms | **0.018x (54x slower)** |
| Production scale | 128 | 0% | 0.71ms | 110.7ms | **0.006x (156x slower)** |
| Production scale | 128 | 15% | 0.59ms | 110.8ms | **0.005x (189x slower)** |
| Production scale | 128 | 100% | 0.77ms | 113.0ms | **0.007x (147x slower)** |

**The FFI-dispatched Rust CPU kernel is 35-190x SLOWER than the MLX
reference at production scale, not faster.** This is reported plainly, not
hidden or reframed: `conditional_anchor_attention_f32` is an
UNOPTIMIZED, correctness-first scalar Rust loop (no SIMD, no BLAS, no
threadgroup-level GPU parallelism) explicitly built to be a slow, obvious
reference for the Metal kernel to match, matching this project's own
established two-phase discipline (the CPU tensor crate was proven before
any GPU dispatch was attempted, the SAME pattern used for HZ-0B's B2
simulator before PMetal). MLX's own attention ops are Accelerate/Metal-
backed and heavily optimized; a naive scalar competitor was never
expected to beat them, and now that expectation is a measured fact
instead of an assumption.

**A second real, consistent finding across an independent script**:
MLX's own latency barely varies with trigger rate here either (0.99ms at
0% vs 0.64ms at 100% for seq=40) -- confirming
`docs/restart/hz0c_c9_end_to_end_report_results.md`'s finding that the
reference path does not currently translate trigger sparsity into real
wall-clock savings, now corroborated by a second, independently-written
benchmark rather than resting on one measurement.

## What "model-level integration" means now, precisely

- **Done**: a real, tested, safety-checked Python<->Rust FFI mechanism
  exists and is proven numerically correct against the actual Python
  reference. This did not exist anywhere in this repo before today.
- **Not done, and not claimed**: a performance win. That requires either
  the already-implemented Metal GPU kernel
  (`MetalConditionalAnchorAttention`, not yet exposed through this same
  FFI bridge -- a real, bounded, named next step) or a vectorized/BLAS-
  backed CPU kernel -- both squarely inside the tracker's own separate
  "grouped/cache-optimized dispatch" item, not this one.
