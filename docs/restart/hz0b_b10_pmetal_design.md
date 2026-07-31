# HZ-0B Phase B10: PMetal Implementation -- Design and Progress

Date: 2026-07-30/31. Per the plan's own gate: *"Once reference semantics
are stable"* -- satisfied by B2/B3 being complete, tested (20 + 11 Python
tests), and having had 2 real bugs found and fixed today
(`docs/restart/hz0b_b8_stage5_results.md`). B9's fine-tuning work does
NOT block B10 -- B10 ports the memory *operations* (B2/B3), which haven't
changed since being fixed, not B9's model-training experiments.

## Scope, staged like HZ-0A's own PMetal work

HZ-0A's PMetal port (`restart/hz0a_pmetal/`) is 4 crates: `-kernel` (core
types), `-tensor` (CPU reference execution, parity-tested against
Python), `-gpu` (real Metal kernels via the `metal` crate), `-bridge`
(Python binding). It reached that structure across substantial, real,
incremental work (visible in this repo's own commit history) -- B10
follows the same staging, not a shortcut:

1. **CPU tensor reference** (this pass): a real, complete, tested Rust
   port of all 10 B2 operations, matching
   `reference/hz0b_memory_simulator.py` exactly as it stands today
   (including today's 2 fixes: 0.999 match threshold,
   confidence-weighted reads).
2. **Metal GPU kernels** (not built this pass): a `hz0b-pmetal-memory-gpu`
   crate mirroring `hz0a-pmetal-gpu`'s structure, once there's a real
   reason to need it -- see "Why GPU kernels aren't built yet" below.
3. **Python bridge** (not built this pass): mirrors `hz0a-pmetal-bridge`,
   needed once B6-B9's integration work wants to call the Rust/Metal path
   instead of the pure-MLX reference it currently uses.

## What's built and verified (stage 1)

`restart/hz0a_pmetal/crates/hz0b-pmetal-memory/` (added to the existing
workspace, `#![forbid(unsafe_code)]`, zero production dependencies,
matching `hz0a-pmetal-kernel`'s own conventions): all 10 B1-contract
operations (`reset`, `read`, `write`, `reinforce`, `update`, `protect`,
`forget_or_decay`, `delete`, `serialize`, `restore`) as flat f32/i32
buffer functions, batch-dimension-aware.

**Two independent verification layers, both passing:**

1. **8 native Rust correctness tests** (`tests/correctness.rs`), mirroring
   `tests/reference/test_hz0b_memory_simulator.py`'s own coverage:
   exact store/retrieve, same-key overwrite routing, the near-identical-
   keys fix (0.999 threshold), protection blocking a direct overwrite,
   reset wiping everything, decay + protection-aware slowdown, the
   confidence-weighted-read fix (fresh beats stale in a genuine tie),
   capacity pressure with a protected survivor.
2. **1 cross-language parity test** (`tests/parity.rs` +
   `scripts/hz0b_generate_rust_parity_fixture.py`): a real 12-step
   operation sequence (write, write, read, protect, a rejected attack
   write, update, 5x decay) is run through the ACTUAL Python reference,
   every input/output recorded to `tests/fixture.json`, then replayed
   through the Rust port -- every intermediate result and the final state
   (keys, values, confidence, protection) match to float precision
   (<1e-4). This is the same "strongest form of agreement" pattern
   `hz0a-pmetal-tensor`'s own parity test established for HZ-0A's model
   code.

`cargo test -p hz0b-pmetal-memory`: 9/9 pass. `cargo build` (whole
workspace): still clean, no regressions to HZ-0A's own PMetal crates.

## Why GPU (Metal) kernels aren't built yet -- an honest scope call

B2's memory operations, at the scale every B6-B9 experiment has actually
used (`num_slots` 8-16, `key_dim`/`value_dim` 16-32), are cheap --
milliseconds of CPU or MLX time per call, never the measured bottleneck
in any of this session's real training runs (the frozen HZ-0A backbone's
own forward/backward pass dominates every wall-clock number recorded so
far). A fused Metal kernel would be a real, worthwhile speed win **once**
this project moves to training memory-augmented HZ-0A at real batch
sizes/sequence lengths over many steps -- exactly the same reasoning
HZ-0A's own PMetal work used to prioritize the GDN-2 recurrence (the
actual per-token bottleneck) first and build outward from there, not the
reverse.

Building GPU kernels now, before there's a real workload that needs
them, would be effort spent without a way to verify it matters -- the
same "don't retrain HZ-0A on a mechanism advantage that hasn't been shown
to matter yet" discipline this session applied to the GDN-3 investigation
(`docs/restart/hz0a_gdn3_candidate_design.md`). The CPU reference tier
built here is both a real deliverable on its own (a working, tested,
parity-verified alternative to the MLX path) and the correct-by-
construction foundation any future Metal kernel would need to be checked
against, exactly as HZ-0A's `-tensor` crate was built before `-gpu`.

## Real next steps, in order, if this continues

1. A Python binding for this CPU crate (PyO3, mirroring
   `hz0a-pmetal-bridge`) -- lets B6-B9's probe scripts optionally call the
   Rust path instead of MLX, a real cross-check independent of MLX's own
   numerics.
2. Benchmark the Rust CPU path against the MLX path at realistic B6-B9
   scale -- if Rust CPU is already fast enough, Metal kernels may not be
   needed at all; measure before building.
3. Only if step 2 shows a real bottleneck: build `hz0b-pmetal-memory-gpu`
   with actual Metal Shading Language kernels, GPU-vs-CPU parity tests
   (mirroring `hz0a-pmetal-gpu/tests/decode_equivalence.rs`'s own
   structure), before trusting any GPU-path result.
