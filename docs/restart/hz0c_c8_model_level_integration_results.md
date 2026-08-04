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

## GPU FFI added, and the real bottleneck found and partially fixed (2026-08-04, same day)

The named next step was taken immediately: `MetalConditionalAnchorAttention`
(the already-built GPU kernel) is now exposed through the SAME bridge --
`hz0c_metal_conditional_attention_create`/`_destroy`/`_forward` (a
reusable opaque handle, since creating a Metal device/pipeline/queue is
real amortizable setup cost that must not be smuggled into a per-call
latency number), wrapped in Python as
`reference/hz0c_pmetal_bridge.py::MetalConditionalAttention`. 2 new Rust
tests (handle reused across two calls, both correct; null-handle and
double-close safety) and 2 new Python tests (matches the real fixture
across two calls; using a closed handle raises cleanly) all pass.

**Before benchmarking it, a real redundant-work bug was found and fixed
in the forward Metal shader itself.** The original kernel recomputed
each source position's O(D) query-key dot product up to `2 + head_dim`
times per (position, head) -- once for the max-score pass, once for the
denominator pass, and once more INSIDE the per-output-component loop --
and recomputed the query projection itself (which does not even depend
on the source position) inside every one of those passes too. Fixed by
caching the query projection once per head (`q_cache`) and each source's
raw score once per head (`scores`, reused across all three passes) --
same algorithm, same numerical result (all existing Rust tests still
pass, including GPU-vs-CPU parity), only the redundant recomputation
removed. This is a real, verified improvement, kept regardless of what
the benchmark below shows next.

**The full honest benchmark, rerun with the GPU path included and the
redundant-work fix applied**:

| Shape | Seq | Rate | MLX | CPU FFI | GPU FFI |
| --- | ---: | ---: | ---: | ---: | ---: |
| C3-scenario scale | 40 | 0% | 0.93ms | 35.4ms (0.026x) | 88.3ms (0.011x) |
| C3-scenario scale | 40 | 15% | 0.30ms | 34.6ms (0.009x) | 305.9ms (0.001x) |
| C3-scenario scale | 40 | 100% | 0.28ms | 34.9ms (0.008x) | 2,040.9ms (0.0001x) |
| Production scale | 128 | 0% | 0.28ms | 115.9ms (0.002x) | 89.3ms (0.003x) |
| Production scale | 128 | 15% | 0.28ms | 115.3ms (0.002x) | 914.6ms (0.0003x) |
| Production scale | 128 | 100% | 0.28ms | 118.2ms (0.002x) | **13,796ms (0.00002x)** |

**The GPU path is not just failing to beat MLX -- at higher trigger
rates it is dramatically SLOWER than even the naive CPU kernel**, and
gets catastrophically worse as trigger rate or sequence length grows
(13.8 SECONDS at seq=128/100% trigger, versus the CPU kernel's 118ms at
the same shape). This exposed a SECOND, larger, and more architecturally
fundamental redundancy that the within-thread fix above does not touch:
the forward kernel dispatches ONE THREAD PER OUTPUT ELEMENT
(`token * D + row`), so all `D = 768` threads that share the same
`token` (one per output row/feature) independently recompute the exact
same `q_cache` and `scores` arrays from scratch -- a 768x cross-thread
duplication of work that grows with trigger rate (more visible sources
to score) and sequence length (more query positions), exactly matching
the pattern observed. The CPU kernel does not have this problem (it is a
single sequential loop, not 768-way parallel redundant dispatch), which
is why it stays roughly flat and, at the higher end, ends up faster than
the naively-parallel GPU version.

**This is precisely what "grouped/cache-optimized dispatch" means, now
characterized with real numbers instead of being a vague future label**:
the fix is to dispatch one THREADGROUP per token (not one thread per
output element), with the `D` threads in that group cooperating via
Metal `threadgroup` shared memory to compute `q_cache`/`scores` ONCE and
reuse them across all `D` rows -- the same threadgroup-shared-memory
reduction pattern already proven in this codebase for the GDN2 backward
kernel. Not attempted this pass; a real, now well-specified next step
rather than an open-ended one.

## What "model-level integration" means now, precisely

- **Done**: a real, tested, safety-checked Python<->Rust FFI mechanism
  exists for BOTH the CPU and GPU kernels and is proven numerically
  correct against the actual Python reference. This did not exist
  anywhere in this repo before today.
- **Done, and a real (if partial) improvement**: one genuine redundant-
  work bug in the GPU forward shader was found and fixed (verified
  correct, no numerical change) via the very benchmark built to measure
  performance honestly.
- **Not done, and not claimed**: a performance win. The dominant
  remaining cost is now understood precisely -- 768x cross-thread
  redundant recomputation from a one-thread-per-output-element dispatch
  design -- and the fix (threadgroup-per-token with shared memory) is
  named concretely rather than left as an open-ended "optimize this
  later." Still squarely inside the tracker's "grouped/cache-optimized
  dispatch" item, not this one.
