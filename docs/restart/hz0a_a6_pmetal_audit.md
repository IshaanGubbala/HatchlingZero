# HZ-0A A6 PMetal Reference Audit

Date: July 28, 2026

## Status

The first A6 PMetal-facing restart scaffold now exists in the live repo and is test-backed against the locked HZ-0A A1 spec and the A2 NumPy oracle.

## Workspace

- `/Users/ishaangubbala/Documents/Training/restart/hz0a_pmetal/`

Key contents:

- Rust workspace contract:
  - `Cargo.toml`
  - `crates/hz0a-pmetal-kernel/`
  - `crates/hz0a-pmetal-bridge/`
- Python parity helper:
  - `python/pmetal_reference.py`
- Parity test:
- `/Users/ishaangubbala/Documents/Training/tests/reference/test_pmetal_reference.py`

The Python parity surface now also includes a deterministic AdamW update contract with explicit first/second-moment state and update-norm reporting.

The forward cache now has an explicit `gdn2_backward(...)` reverse-scan implementation returning gradients for Q, K, V, all three gate-logit tensors, and the initial recurrent state. Its contract is validated against an independent torch recurrence oracle.

The Rust kernel crate now includes an executable dependency-free CPU `gdn2_forward_f32` implementation with flat `[batch, sequence, heads, channel]` inputs, explicit `[batch, heads, value, key]` state, shape validation, state carry, and chunk equivalence tests.

It now also exposes `gdn2_backward_f32` for the same flat contract. The reverse scan returns Q/K/V, decay/erase/write-logit, and initial-state gradients; its Q gradient passes a Rust finite-difference test on a two-token case.

`gdn2_backward_chunked(...)` now checkpoints the recurrent state at chunk boundaries and propagates the state cotangent backward across uneven chunks. Full-vs-chunked gradients are covered by a 7-token sequence with chunk size 3.

## Verified Results

Executed successfully:

```bash
cargo test
python3 -m pytest -q tests/reference/test_pmetal_reference.py tests/reference/test_gdn2_reference.py tests/reference/test_gdn2_gradients.py
```

Observed results:

- Rust workspace tests: passed
- Python parity/reference/gradient suite: `14 passed`

## What This Proves

- the PMetal restart surface now has a clean A1-aligned operator/cache contract
- the future backend can target a stable `forward -> outputs, final_state, backward_cache` API
- the current Python PMetal-style forward wrapper matches the NumPy oracle
- the current Python PMetal-style block wrapper matches recurrent and attention reference blocks
- the current Python PMetal-style tiny-model wrapper matches logits/state flow and a simple loss path
- the Rust CPU reference executes the recurrence and carries state across chunk boundaries
- the Rust CPU backward returns the required gradient families and passes a finite-difference check
- the optimizer reference matches the closed-form first AdamW update, including serialized moment state
- the PMetal-style GDN-2 backward matches the independent A3 recurrence oracle for all required gradients

## What This Does Not Yet Prove

- actual PMetal tensor execution
- actual PMetal tensor/device execution
- device-side PMetal/Metal execution and optimizer integration
- end-to-end PMetal training
- chunked checkpoint/recompute backward execution
- real PMetal tensor execution
- end-to-end PMetal training

## A6 Current Assessment

A6 is meaningfully in progress, but not complete. The current milestone is a contract-and-parity bring-up across operator, block, and tiny-model-loss surfaces, not a finished PMetal reference implementation.

The config-driven PyTorch reference at `reference/hz0a_torch_model.py` now covers the locked full topology, recurrent state API, periodic causal attention, tied LM head, and model-level parameter accounting. A meta-device test confirms the implementation reports exactly `301,178,112` parameters; it does not claim PMetal tensor execution.
