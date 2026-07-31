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
2. **Python bridge** (this pass, added after the CPU tier landed): a
   `hz0b-pmetal-memory-bridge` cdylib + ctypes wrapper -- see "Python
   bridge" below.
3. **Metal GPU kernels** (not built this pass): a `hz0b-pmetal-memory-gpu`
   crate mirroring `hz0a-pmetal-gpu`'s structure, once there's a real
   reason to need it -- see "Why GPU kernels aren't built" below, now
   backed by a real benchmark rather than just an a priori argument.

**Correction to this doc's own first draft**: it originally said stage 3
(Python bridge) would "mirror `hz0a-pmetal-bridge`." Checked directly
before building it -- `hz0a-pmetal-bridge` (`crates/hz0a-pmetal-bridge/
src/lib.rs`) is not actually a real FFI binding; it is a config-summary
scaffold (`restart_bridge_summary()`, returns a formatted string) with no
Python-callable surface at all. `python/model_bridge.py` is pure
Torch/numpy and never calls into any Rust crate. So there was no real
precedent in this workspace for an actual Rust<->Python binding -- B10's
bridge (below) is the first one, built as a plain `cdylib` + `ctypes`
rather than introducing a new PyO3/maturin toolchain dependency this
workspace has never used.

## What's built and verified (stage 1: CPU tensor reference)

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

## What's built and verified (stage 2: Python bridge)

`restart/hz0a_pmetal/crates/hz0b-pmetal-memory-bridge/` -- a `cdylib`
exposing all 8 mutating/reading B2 ops (`reset`/`read`/`write`/
`reinforce`/`update`/`protect`/`forget_or_decay`/`delete`) as
`extern "C"` functions over flat pointers. `serialize`/`restore` are
deliberately NOT exposed: in the Rust reference they're pure clones, and
a Python caller already holding every field it passed in gains nothing
from round-tripping them across the FFI boundary. All `unsafe` is
confined to this one crate's pointer marshalling
(`from_c`/`write_out`); the actual op logic still runs inside
`hz0b-pmetal-memory`, which stays `#![forbid(unsafe_code)]`.

`restart/hz0a_pmetal/python/hz0b_memory_bridge.py` wraps it via `ctypes`
(no PyO3/maturin -- see correction above) in a numpy-backed functional
API with the exact same function names/signatures/immutable-state style
as `reference/hz0b_memory_simulator.py`, so callers can swap the import
with no other code changes. Every buffer is numpy-owned and pre-sized by
the caller (shapes are fully determined by `batch`/`num_slots`/
`key_dim`/`value_dim`), so nothing allocates across the FFI boundary in
either direction and there's no `free`-style function to pair.

**Verified two ways, both passing** (`tests/reference/
test_hz0b_memory_rust_bridge.py`, 2/2):
1. Replays the SAME 12-step fixture `tests/parity.rs` already replays
   Rust-side, but now through the full Python -> ctypes -> Rust ->
   ctypes -> Python round trip -- proves the bridge itself is correct,
   not just the underlying crate.
2. A second, independent test runs a fresh random sequence through the
   LIVE Python/MLX reference and the bridge in the same process
   side-by-side, catching any drift a frozen fixture could hide.

## Why GPU (Metal) kernels aren't built -- backed by a real measurement

The design's original reasoning (B2 ops are cheap, never the measured
bottleneck in any real run) is now backed by an actual number instead of
just an a priori argument. `scripts/hz0b_bridge_benchmark.py` runs
write+read+forget_or_decay (the same three ops any real training step
touches per position) 2000 times at B6-B9's actual scale (`num_slots=16,
key_dim=value_dim=32, batch=1`):

```
Python/MLX reference: 0.7540s total, 0.3770ms/iteration
Rust bridge (ctypes): 0.1360s total, 0.0680ms/iteration
speedup: 5.54x
```

The Rust CPU path is already 5.5x faster than the MLX path it would
replace -- and both are already sub-millisecond per iteration, negligible
next to a single forward/backward pass through HZ-0A's real 301M-param
backbone (tens of milliseconds at minimum, per this session's own
training-throughput numbers). A fused Metal kernel could only shave
microseconds off an already-microsecond-scale operation; it would not be
measurable against the backbone's own cost at any scale B6-B9 has
actually run. This confirms, rather than assumes, that GPU kernels are
not warranted yet -- the same "measure before building" discipline the
design's step 2 called for.

## Real next steps, if this continues

Only reopen the Metal GPU tier if a future integration point (e.g. B11's
full eval suite, or training memory-augmented HZ-0A end-to-end rather
than isolated probes) profiles the memory ops as a measurable fraction
of a real step's wall-clock time -- re-run
`scripts/hz0b_bridge_benchmark.py`'s pattern at THAT scale first; if the
ops still don't show up, still don't build it.
