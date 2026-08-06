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

## Remaining Gate

Still open: connect this dispatch primitive to the full HZ-0E parameterized
model path, measure end-to-end residency and overhead, and compare complete
MoE execution against the existing dense baseline.
