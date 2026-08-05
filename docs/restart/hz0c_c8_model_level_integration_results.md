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
kernel.

## Grouped/cache-optimized dispatch implemented (2026-08-04, same day)

The fix named above was implemented, not left as a future item. The
forward kernel is now one THREADGROUP PER TOKEN with `D` threads
cooperating via `threadgroup` shared memory:

1. `q_cache[D]`: thread `row` computes its own query-projection element
   (the `(head, key)` index space is exactly the `D` index space), then
   `threadgroup_barrier`.
2. `scores[H*S]`: the `H*(t+1)` real (head, source) score pairs are
   distributed round-robin across the `D` threads (each pair's key
   projection computed exactly once, not once per output row), then
   barrier.
3. Per-head softmax normalization, done by the first `H` threads (cheap,
   pure shared-memory reads), then barrier.
4. `attended[D]`: the `H*head_dim = D` (head, component) attended values
   map 1:1 onto the `D` threads, then barrier.
5. Every thread computes its own output row as the inherent `out_w` mix
   over the now-shared `attended[D]` -- the one part of the per-row work
   that was never redundant (a dense output projection always needs a
   full mix).

Same algorithm, same numerical result: all 8 Rust tests in this crate
(including both GPU-vs-CPU parity tests, at the small AND the locked
dim=768 A1 shape) and all 5 Python bridge tests (reusing the real
Python-reference fixture) still pass unchanged.

**Rerunning the exact same benchmark**:

| Shape | Seq | Rate | MLX | CPU FFI | GPU FFI (before) | GPU FFI (after) | Improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C3 scale | 40 | 0% | 0.57ms | 35.2ms | 88.3ms | 3.62ms | **24.4x faster** |
| C3 scale | 40 | 15% | 0.28ms | 34.6ms | 305.9ms | 3.59ms | **85.2x faster** |
| C3 scale | 40 | 100% | 0.28ms | 34.7ms | 2,040.9ms | 5.89ms | **346.6x faster** |
| Production | 128 | 0% | 0.28ms | 110.7ms | 89.3ms | 5.60ms | **15.9x faster** |
| Production | 128 | 15% | 0.28ms | 110.6ms | 914.6ms | 6.60ms | **138.6x faster** |
| Production | 128 | 100% | 0.28ms | 112.8ms | 13,796.3ms | 33.36ms | **413.5x faster** |

**The GPU kernel now consistently beats the CPU kernel at every
configuration tested** (3.4x-19.8x faster, reversing the earlier finding
where GPU was up to 122x SLOWER than CPU at the worst shape) -- the
cross-thread redundancy was, as characterized, the dominant cost, and
removing it changed the GPU/CPU ordering completely, not just the
absolute numbers.

**Still honestly behind MLX -- by 6.3x to 119x depending on shape/rate,
not the narrower "6x-21x" this doc first reported** (that first summary
undercounted the worst case; corrected here rather than left standing):

| Shape | Seq | Rate | GPU FFI vs MLX |
| --- | ---: | ---: | ---: |
| C3 scale | 40 | 0% | 6.3x slower |
| C3 scale | 40 | 15% | 12.9x slower |
| C3 scale | 40 | 100% | 21.4x slower |
| Production | 128 | 0% | 19.7x slower |
| Production | 128 | 15% | 23.4x slower |
| Production | 128 | 100% | **118.8x slower** |

The gap widens sharply with sequence length AND trigger rate together
(worst at seq=128/100%) -- consistent with the kernel's remaining
per-source work (still an O(D) scalar loop for each key/value
projection, not vectorized) scaling with the number of visible sources,
which is exactly `seq * rate`.

## Attributing the remaining gap: kernel cost, not marshaling overhead (2026-08-04, same day)

The doc originally named two uninvestigated candidate causes for this
gap -- MLX's own kernel tuning, and Python/NumPy/`ctypes` marshaling
overhead not yet isolated from device time. Measured directly rather
than left as speculation:
`restart/hz0a_pmetal/crates/hz0a-pmetal-gpu/examples/gpu_forward_latency.rs`
times `MetalConditionalAnchorAttention::forward` directly from Rust --
no Python process, no `ctypes`, no NumPy involved -- at the identical
shapes/rates.

| Shape | Seq | Rate | Pure Rust | Python ctypes | Marshaling overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| C3 scale | 40 | 0% | 3.333ms | 3.621ms | 0.288ms (8.6%) |
| C3 scale | 40 | 15% | 3.482ms | 3.590ms | 0.108ms (3.1%) |
| C3 scale | 40 | 100% | 5.843ms | 5.888ms | 0.045ms (0.8%) |
| Production | 128 | 0% | 5.594ms | 5.599ms | 0.005ms (0.1%) |
| Production | 128 | 15% | 6.477ms | 6.600ms | 0.123ms (1.9%) |
| Production | 128 | 100% | 33.592ms | 33.362ms | -0.230ms (noise) |

**Marshaling overhead is negligible -- 0-9%, mostly within run-to-run
noise (one row even came out slightly negative, i.e. the "overhead" is
smaller than measurement jitter).** The candidate cause named as (b) is
resolved: it is NOT the explanation. The full 6.3x-119x gap to MLX is
real Metal kernel dispatch cost -- candidate (a), MLX's own kernel being
more aggressively tuned (SIMD-group-level reductions, vectorized/fused
per-source projections instead of this kernel's still-scalar O(D) inner
loop for each key/value projection), is the actual, now-confirmed
explanation. A vectorized inner loop (e.g. `float4`/SIMD-group
reductions for the per-source key/value projections) is the concrete
next optimization this points to, not attempted this pass.

## What "model-level integration" AND "grouped/cache-optimized dispatch" mean now, precisely

- **Done**: a real, tested, safety-checked Python<->Rust FFI mechanism
  exists for BOTH the CPU and GPU kernels and is proven numerically
  correct against the actual Python reference. This did not exist
  anywhere in this repo before today.
- **Done**: the within-thread redundant-work bug (query projection and
  per-source score recomputed up to `2+head_dim` times) was found and
  fixed.
- **Done**: the larger cross-thread redundancy (768x duplicated work
  from one-thread-per-output-element dispatch) was found, precisely
  characterized with real numbers, AND fixed via a threadgroup-
  cooperative redesign -- not just named as future work. Verified
  correct (all Rust and Python parity tests unchanged) and verified fast
  (24x-413x faster than the pre-fix GPU kernel across every measured
  shape/rate; now consistently 3.4x-19.8x faster than the CPU kernel too,
  reversing the earlier ordering).
- **Not done, and not claimed**: beating MLX itself. The GPU-FFI path is
  now within 6.3x-119x of MLX (down from up to ~49,000x slower measured
  immediately before the threadgroup fix -- the within-thread fix alone
  was already applied by then, so this is specifically what the
  cross-thread/grouped-dispatch fix closed). The gap's cause is no
  longer speculative: a direct pure-Rust measurement
  (`hz0a-pmetal-gpu/examples/gpu_forward_latency.rs`) found Python/
  `ctypes` marshaling overhead negligible (0-9%, mostly noise) -- the
  full gap is real Metal kernel dispatch cost, pointing at a concrete
  next step (vectorizing the still-scalar per-source key/value
  projection loop) rather than an open-ended "something in the FFI path
  might be slow."
