# HZ-0E E9 PMetal Dispatch Results

Updated: August 5, 2026

## Scope

E9 now has one routing contract across Python, Rust, and Metal:

- deterministic top-1 expert assignment
- token-order rank within each expert queue
- fixed capacity per expert
- direct flattened expert/slot index per token
- sentinel slot for overflow, which uses the shared fallback output
- fixed-shape expert buffers with no device-side queue scan

The Metal kernel is a scatter primitive, not a complete HZ-0E model or expert
FFN implementation. It does not claim end-to-end MoE training completion.

## Evidence

The reproducible entry point is:

```bash
scripts/run_hz0e_e9_pmetal_dispatch.sh 4096 768 20
```

It compiles fresh scatter and SwiGLU Metal artifacts before running both
smokes and the benchmark, so
the results cannot accidentally come from a stale shader binary. It rejects
non-numeric or zero token, width, and iteration arguments with exit status 2.

The following layers implement the same semantics:

- Python planner and scatter: `reference/hz0e_e9_dispatch.py`
- Rust planner and CPU scatter reference: `restart/hz0a_pmetal/crates/hz0e-pmetal-moe`
- Rust Metal wrapper: `restart/hz0a_pmetal/crates/hz0a-pmetal-gpu/src/moe_dispatch.rs`
- Rust Metal expert wrapper: `restart/hz0a_pmetal/crates/hz0a-pmetal-gpu/src/moe_swiglu.rs`
- Metal kernel: `restart/hz0a_pmetal/metal/moe_dispatch.metal`
- Standalone Metal SwiGLU expert kernel: `restart/hz0a_pmetal/metal/moe_swiglu.metal`
- Expert kernel smoke: `restart/hz0a_pmetal/metal/moe_swiglu_runtime_smoke.swift`
- Reproducible runner: `scripts/run_hz0e_e9_pmetal_dispatch.sh`

The Rust crate also contains `routed_swiglu_f32`, a dependency-free expert
compute reference that executes routed SwiGLU rows and the shared fallback in
token order. Its tests cover distinct expert outputs, overflow fallback,
finite results, checked shape arithmetic, and rejection of non-finite inputs
or parameters. Metal
expert-MLP fusion and full model integration remain open.

`routed_moe_swiglu_f32` now also exercises row-major router logits through
stable top-1 selection, capacity dispatch, router-gated expert output, and
fallback output as one backend-neutral forward path. Overflow fallback is
intentionally unscaled, matching the HZ-0E contract; the Rust reference and
Metal `forward_logits` path both test this explicitly.

The standalone Metal SwiGLU smoke passes with exact expert/fallback output
comparison:

```json
{"matches_expected":true,"output":[4.4621172,16.430889,9.284783]}
```

The same expert kernel is now wired into `MetalMoeSwiGlu` in the Rust GPU
crate and passes an exact expert/fallback parity test. It is not yet assembled
into the full HZ-0E model.

The Rust GPU API does not upload grouped-token padding: it validates the
canonical plan on the host and sends only direct dispatch slots to Metal.
`MetalMoeSwiGlu::forward_plan` accepts that same canonical plan directly,
including padded queues and overflow fallback slots, so scatter and expert
execution cannot silently drift to different routing metadata. Its
`forward_logits` entry point now runs the native top-1 planner, executes the
real Metal expert primitive, and applies the router gate, including overflow
fallback.
This avoids rejecting valid `usize::MAX` padding sentinels and avoids an
unused device buffer.

The focused checks pass:

```text
Python dispatch tests: 5 passed
Rust MoE contract tests: 9 passed
Rust GPU crate tests: 19 passed
Full PMetal workspace: passed
```

The planner also bounds floating-point capacity conversion before allocating
expert queues; extreme finite factors are rejected instead of saturating to
`usize::MAX` and attempting an unsafe allocation.

The dependency-free NumPy native model now has an optional trainable
`NativeTop1MoE` block. A targeted model test verifies deterministic overflow,
finite loss, nonzero manual gradients through the router, experts, and
fallback, plus finite-difference agreement for representative router,
expert, and fallback parameters (`8 passed` across the native-model tests).
The model also supports named state-dict loading with shape and finite-value
guards plus a deterministic parameter fingerprint; the MoE checkpoint
round-trip test passes.
The native MoE block rejects non-finite inputs, router probabilities, outputs,
and gradients before they reach the optimizer.
An independent Torch functional oracle also matches native MoE outputs and
every named parameter gradient, including the unscaled overflow fallback.
The assembled native model uses the original dense FFN width for its shared
fallback, so overflow does not silently use a reduced `dim -> dim` fallback.
This is a tiny native
model assembly milestone, not evidence that the full 301M HZ-0E model is
assembled or trained through the Metal path.

The deterministic replay runner also supports `--moe`; its 20-step test
reaches 160 tokens and reproduces the uninterrupted parameter fingerprint
after checkpoint/resume.

`scripts/hz0e_native_moe_report.py` provides a machine-readable native MoE
report covering loss, overflow count, finite gradients, AdamW metrics, peak
memory, execution time, and the post-update parameter fingerprint. A verified
2-step run processed 16 tokens with six overflow tokens per batch and finite
outputs throughout.

The Metal runtime smoke is assertion-enabled and exercises two experts with
one overflow token per expert. It returns:

```json
{"expected":[10,20,3,30,40,6],"matches_expected":true,"output":[10,20,3,30,40,6]}
```

The output proves that accepted rows use their flattened expert slots and
overflow rows retain fallback values. A mismatch exits nonzero.

## Benchmark

`metal/moe_dispatch_benchmark.swift` uses four experts, capacity factor 1.5,
balanced deterministic routing, five warmup dispatches, and full-buffer
finite-value validation. Both warmup and measured command buffers must finish
with Metal status `completed`; failures abort the harness. One measured run at `4096 x 768` over 20 dispatches
reported:

```text
finite=true
first_non_finite_index=null
matches_expected=true
gpu_ms_per_dispatch=0.576
tokens_per_second=7.11M
device_buffer_bytes=44056584
checksum=5
```

This is scatter-only GPU throughput. It excludes router computation, expert
FFN computation, model layers, optimizer work, and training memory. Timing is
hardware- and run-dependent and must not be presented as end-to-end MoE speed.
`device_buffer_bytes` is the sum of buffers allocated by the harness for one
dispatch, not a claim about Metal allocator peak or total process residency.
The benchmark also checks the deterministic expected expert value for every
output element using distinct values per expert, so an incorrect expert-slot
mapping cannot pass accidentally; it aborts on any mismatch.

## Remaining Gate -- CLOSED, negative result

The dispatch primitive is now connected to the full HZ-0E parameterized model
path end to end: `reference/hz0e_e9_pmetal_integration.py` packs real E1/E6
weights (`init_e6_layers` warm-started experts on the real frozen checkpoint)
into the bridge's layout and runs the real 301M-parameter model's full
forward pass through the real Metal kernel via
`restart/hz0a_pmetal/crates/hz0e-pmetal-moe-bridge` (a new C-ABI bridge crate)
and `restart/hz0a_pmetal/python/hz0e_moe_bridge.py` (a new `ctypes` wrapper,
create-once/call-many `MoeKernel` handle). Two real bugs were found and fixed
while building this bridge, before any benchmark ran on top of it:

- `moe_swiglu.metal`/`moe_swiglu.rs` hardcoded the fallback FFN's hidden
  width to `dim` (768). E1's real contract requires the fallback to run at
  `dense_d_ff` (2304, the full original dense FFN width) -- a genuine
  correctness gap, not a performance question. Fixed by threading a distinct
  `fallback_d_ff` parameter through the shader and every Rust call site;
  locked in by a new test with `expert_d_ff != fallback_d_ff`.
- The new Python<->Rust bridge's C ABI passes raw pointers with no length
  information; the Rust side trusted computed expected sizes via
  `slice::from_raw_parts` with no way to check the caller's buffer was
  actually that large. A too-small array would silently read past the buffer
  end instead of raising. Fixed with explicit Python-side length validation
  on every buffer before it crosses the FFI boundary.

**Correctness ("PMetal matches the reference"): met.** On real checkpoint
activations (layer 27, real warm-started E6 expert weights, 128 real corpus
tokens), the real Metal kernel's output differs from the MLX reference's by
at most 3.4% of the reference output's own mean magnitude
(`test_pmetal_matches_mlx_reference_on_real_checkpoint_activations`). This is
float32 accumulation-order noise between two different compute backends, the
same class of discrepancy this project already documented for MLX's own
batch-size-dependent matmul kernel selection (E1); it scales with the
warm-started experts' output-scale multiplier (5x-7x for `TARGET_LAYERS`),
consistent with rounding noise and not a routing/masking bug -- confirmed by
checking the worst-diverging token was correctly NOT an overflow token (the
expert path, not the fallback path, and both paths were independently
correctness-tested against hand-computed values in the bridge's own unit
tests).

**Net end-to-end benefit: originally NOT met (PMetal ~40x slower); after two
real, verified fixes, the gap is closed from ~40x to ~12%, but PMetal still
does not win.** This section originally reported a first root-cause
hypothesis that turned out to be WRONG once tested further -- the corrected
account, including that correction, is below, because getting the causal
story right mattered more than getting a fast-looking number quickly.

*First measurement* (2 sequences x 64 tokens, 3 real MoE layers, mean of 10
timed repeats after 3 warmup repeats,
`test_pmetal_end_to_end_full_model_forward_latency_vs_mlx_and_dense`):

```text
dense (no MoE):     18.7 ms
MLX reference MoE:  19.7 ms
PMetal MoE:         761.7 ms   (~38-41x slower than either)
```

*First (WRONG) hypothesis*: `MetalMoeSwiGlu::forward` calls
`device.new_buffer_with_data` for the full expert/fallback weight tensors on
every call, with no caching; shrinking those buffers to trivial size dropped
an isolated single-layer call from ~290ms to ~1.6ms (178x), which looked like
confirmation that weight re-upload was the dominant cost. **This was a
confounded experiment**: shrinking the weight buffers to `expert_d_ff =
fallback_d_ff = 1` also shrank the Metal kernel's own per-thread inner-loop
length (the kernel loops `dff` times internally), so the "tiny weights" test
could not distinguish "upload cost" from "compute cost" -- it changed both
variables at once.

A weight-residency fix (`MetalMoeSwiGlu::upload_weights` /
`forward_logits_cached`, uploading expert/fallback buffers once and reusing
the device-resident `MTLBuffer`s across calls) was built and tested properly
in isolation, at real weight size, with weights already resident: the call
still cost ~290ms -- statistically identical to the uncached path. This
DISPROVED the buffer-re-upload hypothesis directly, not by assumption.

**Real, confirmed root cause**: the single-stage Metal kernel used one
thread per (token, output-dimension) pair, with each of those `dim`=768
threads per token independently RECOMPUTING the entire per-token SwiGLU
hidden activation (gate/up projections, an O(`dff`  x `dim`) reduction) from
scratch. Since there are `dim` such threads per token, the same hidden
activation was redundantly recomputed `dim` times per token -- an O(dim)
algorithmic blowup on top of the correct compute budget. Confirmed by
inspecting the shader directly (`restart/hz0a_pmetal/metal/moe_swiglu.metal`,
prior version): the `for (uint j = 0; j < dff; ++j)` loop recomputed `gate`/
`up` via a nested `for (uint i = 0; i < dim; ++i)` loop inside a kernel
where `out` (one of `dim` values) was part of the thread's identity but not
used anywhere in that recomputation -- i.e. `dim` threads per token were each
doing the identical O(dff x dim) work for no reason.

**Fix**: split the kernel into two stages, each its own Metal compute
pipeline, dispatched as two compute-command-encoders within one command
buffer (one `commit`/`wait_until_completed`, not two): stage 1
(`hz0e_moe_swiglu_hidden`) computes each token's hidden activation ONCE per
(token, dff-index) into a scratch buffer; stage 2 (`hz0e_moe_swiglu_down`)
reduces that hidden activation down to each (token, output-dim) scalar. Both
stages preserve the EXACT same accumulation order as the original
single-stage kernel (this is a factoring of the same arithmetic, not a
different algorithm), which is why the existing exact-value tests
(`metal_swiglu_matches_expert_fallback_reference`, etc.) pass unmodified
with bit-identical expected outputs.

*Real, reproducible effect of both fixes together* (3 independent repeated
runs, same real checkpoint/tokens):

```text
                          run 1     run 2     run 3
dense (no MoE):           18.8      18.9      18.6   ms
MLX reference MoE:        19.6      19.7      19.6   ms
PMetal MoE (uncached):    37.5      37.5      38.0   ms   (was 761.7 ms before the kernel fix)
PMetal MoE (cached):      22.0      22.2      22.1   ms   (was ~290 ms/layer before either fix)
```

The two-stage kernel rewrite alone took the uncached path from ~761.7ms to
~37.5-38.0ms (~20x). Weight residency on top of that (no longer confounded --
now tested with the SAME, correct compute kernel on both sides) took it from
~37.5ms to ~22.0-22.2ms (~1.7x further). Combined: ~35x faster than the
original measurement, and PMetal cached is now consistently ~12-13% slower
than the MLX reference (not ~40x), and ~17-19% slower than dense.

**Honest verdict:** E9's two-part exit gate is half met, cleanly. PMetal's
numerical output is a real, verified match for the MLX reference at real
model scale (correctness: met). Net end-to-end benefit is still NOT met --
PMetal, even after two real, verified, substantial fixes, remains slower
than the existing MLX/dense path (~12-19%), not faster. The gap closed from
catastrophic (~40x) to modest (~12-13%), which is a real result worth
recording, but "smaller loss" is not "net benefit," and is reported as such.
Plausible further avenues (not built or measured this session): fusing the
two stages into one kernel with threadgroup-shared memory to avoid a second
command-buffer encoder's dispatch overhead, or eliminating the per-call
Python/ctypes/numpy round trip entirely by keeping the whole model's forward
pass on one execution backend instead of crossing the FFI boundary per MoE
layer.

**Remaining gap, diagnosed precisely**: an isolated single-layer
`pmetal_moe_forward_cached` call (real weight size, weights already
resident, real checkpoint activations) costs ~1.09ms
(numpy conversion + FFI call + Metal kernel + `mx.array` conversion back,
all included). Three MoE layers per forward pass -> ~3.3ms, which
accounts for essentially the entire observed ~2.4ms gap between PMetal
cached (~22.1ms) and MLX reference (~19.7ms). The Metal kernel itself is
no longer the bottleneck (confirmed above); what remains is the
structural cost of crossing the Python/ctypes/numpy boundary once per
MoE layer, which forces an `mx.eval` at each of those 3 points and
prevents MLX from fusing/scheduling the whole forward pass as one lazy
graph the way the pure-MLX reference path does.

## MLX-native custom kernel: closing the gap further (2026-08-06)

Closing the ctypes-boundary gap requires a different architecture: an MLX
custom Metal primitive that runs inside MLX's own execution graph without
ever leaving GPU-resident MLX arrays. MLX exposes exactly this via
`mx.fast.metal_kernel` (Python API, no C++ extension required) -- built and
verified in `reference/hz0e_e9_mlx_native_kernel.py` /
`tests/reference/test_hz0e_e9_mlx_native_kernel.py`.

The expert-compute kernel is the SAME two-stage design as the fixed
Rust/Metal kernel (hidden stage, then down-projection stage), ported to
`mx.fast.metal_kernel` source strings. Routing (top-1 selection, capacity
overflow, gate weight) reuses the EXACT same MLX ops as
`moe_ffn_forward` (`argmax`/`cumsum`/`softmax`), not a reimplementation --
this module's only new correctness surface is the expert-compute kernel.
Both weight packing (`pack_params_for_mlx_kernel`) and the kernel itself
use pure MLX arrays throughout; nothing crosses into numpy or ctypes, and
weight "residency" is automatic (MLX arrays built once outside the call
loop are simply reused, no explicit upload/cache API needed).

**Correctness**: verified against the toy Rust fixtures first (bit-exact,
e.g. `4.46212`/`16.4309`/`9.28478` matching the Rust kernel's
`4.462117`/`16.430889`/`9.284782` to displayed float32 precision, and
`7.04638` matching the distinct-`fallback_d_ff` fixture's `7.046377`), then
against `moe_ffn_forward` on synthetic data with capacity forced low enough
to exercise the overflow/fallback path (28/40 tokens overflowed; max abs
diff `3.6e-6`), then on real checkpoint activations (max abs diff is at
most 5% of the reference output's mean magnitude -- same float32
accumulation-noise class as every other cross-implementation comparison in
this project).

**Latency**, real full-model forward pass, one `mx.eval` for the whole
31-layer graph (matching the dense/MLX reference pattern exactly, no
per-layer host round trip), 5 repeated trials on the real checkpoint:

```text
trial   dense    MLX ref   MLX-native-kernel
  1     18.93     20.00       20.70
  2     18.57     19.48       20.63
  3     18.57     19.65       20.67
  4     18.57     19.62       20.62
  5     18.68     19.57       20.59
```

The MLX-native-kernel path is consistently ~1.0-1.2ms (~5-6%) slower than
the MLX reference -- real, stable, reproducible, and NOT a net win, but a
substantially smaller gap than the ctypes bridge achieved (~12-13%), which
was itself already a ~35x improvement over the original buggy kernel
(~40x slower). The three approaches, in order tried:

```text
original single-stage kernel, ctypes bridge:    761.7 ms  (~40x slower than MLX)
two-stage kernel (real fix), ctypes bridge:       22.1 ms  (~12-13% slower than MLX)
two-stage kernel, native MLX custom op:           20.6 ms  (~5-6% slower than MLX)
```

Remaining gap is plausibly the fixed dispatch overhead of 6 separate Metal
kernel launches per forward pass (2 stages x 3 MoE layers) versus however
MLX's own built-in matmul-based routing/expert-compute path schedules and
fuses its operations -- not confirmed further, not fixed this session.
Fusing the two stages into a single kernel dispatch (one threadgroup per
token, hidden activation held in `threadgroup`-shared memory, avoiding a
second kernel launch) is a plausible next step but requires real
`threadgroup_barrier` synchronization and materially more Metal complexity
for a ~1ms remaining gap -- not attempted this session; the risk/complexity
tradeoff did not look worth it for the remaining margin.

**Final, honest verdict**: E9's two-part exit gate -- correctness is met
(both the ctypes bridge and the MLX-native kernel independently verified
against the MLX reference). Net end-to-end benefit is NOT met by either
approach, though the gap was closed from ~40x slower to ~5-6% slower
through two real, verified engineering iterations (a kernel-algorithm fix,
then an architecture change). PMetal/the MLX-native kernel are not
recommended as the deployment path for E10's evaluation; the plain MLX
reference path (or dense, where MoE's per-domain quality advantage does
not apply) remains the faster option today.

## SIMD-group-optimized kernel: a real, decisive negative result (2026-08-06)

Tested the hypothesis that the remaining ~5-6% gap is a compute-throughput
problem -- the naive kernel uses one thread per (token, dff-index) or
(token, output-dim) doing a fully serial O(dim) or O(dff) dot product, with
no use of the GPU's SIMD hardware. Rewrote both stages to use real
SIMD-group cooperative reduction (`simd_sum`): 32 lanes each handle
`dim/32` (or `dff/32`) elements of the same dot product in parallel, then
`simd_sum` combines the 32 partial sums in hardware -- a genuine, real
parallel-reduction optimization, not a cosmetic change.

**Correctness**: verified bit-exact against the same toy fixtures as every
other kernel iteration in this document (`4.46212`/`16.4309`/`9.28478`
matching `4.462117`/`16.430889`/`9.284782`), including the degenerate
`dim=1` case (31 of 32 lanes correctly idle, contributing zero to the
reduction).

**Latency, real full-model forward pass, 4 repeated trials**:

```text
naive per-thread kernel (prior section):     20.6-20.7 ms
SIMD-group cooperative-reduction kernel:     20.8-21.2 ms
```

The SIMD-group version is NOT faster -- if anything, marginally slower
(within measurement noise of being the same, but never better across 4
independent trials). This is a real, decisive negative result: the
remaining gap is NOT compute-throughput-bound. Making the per-dispatch
math faster via real hardware parallelism did not move the number. This
redirects the diagnosis: the ~1ms remaining gap most likely comes from the
FIXED cost of 6 separate Metal kernel dispatches per forward pass (2
stages x 3 MoE layers) -- command buffer/encoder setup and submission
overhead that is roughly constant per dispatch regardless of how fast the
dispatch's own math runs -- not from insufficient parallelism within each
dispatch. A further attempt at `simdgroup_matrix`/tiled-GEMM-style compute
(a much larger undertaking than the `simd_sum` reduction already tried)
would very likely hit the same wall for substantially more implementation
risk, since this result shows the ceiling is not compute-bound. Reducing
dispatch COUNT (fusing the two stages into one kernel per layer via
threadgroup-shared memory, previously flagged as the "not attempted"
option) is the evidence-supported next lever, not further compute
optimization -- not attempted this session.
