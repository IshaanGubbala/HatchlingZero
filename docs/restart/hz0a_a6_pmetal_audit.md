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

A6 is complete for the immediate scope defined in `restart/hz0a_pmetal/PLAN.md`: a parity-oriented Python/Rust PMetal reference surface with explicit forward, cache, backward, final-state, chunk, block/loss, and AdamW contracts. Actual PMetal device execution, full-model device training, and fused Metal optimizer integration remain later A8/A11 work and are not silently counted as A6 evidence.

The config-driven PyTorch reference at `reference/hz0a_torch_model.py` now covers the locked full topology, recurrent state API, periodic causal attention, tied LM head, and model-level parameter accounting. A meta-device test confirms the implementation reports exactly `301,178,112` parameters; it does not claim PMetal tensor execution.

The scaled recurrent-only model also has an exact full-sequence versus chunked-state-carry regression, establishing the model-level state boundary that a PMetal/fused implementation must preserve.

The native Metal recurrence now also has a bounded cached reverse-scan kernel. Its Q/K/V/gate/initial gradients match a Torch autograd oracle on a deterministic three-token case; longer production sequences still require chunked checkpointing.

This establishes native Metal operator forward/backward parity, but the workspace still does not contain a PMetal tensor graph, full-model device bridge, or end-to-end device optimizer loop.

`reference/hz0a_mlx_model.py` now provides a clean MLX model surface for the locked topology, with exact A1 gate/state semantics and tied output embeddings. Its initial scaled GPU forward/state test passes; it is the bridge point for replacing the MLX recurrence with the native Metal operator.

`reference/hz0a_mlx_metal.py` now dispatches a native Metal recurrence through MLX's `metal_kernel` API. A deterministic MLX-vs-native test covers output and final-state parity; this is forward integration only, and the cached backward/optimizer bridge remains open.

The clean MLX model accepts `native_metal=True` to route recurrent blocks through that kernel. During bring-up, a custom VJP uses the independent MLX recurrence so native forward participates in a trainable graph; this is gradient integration, not yet the native cached Metal backward/optimizer path.

`scripts/hz0a_mlx_training_smoke.py` exercises that route with MLX AdamW for two real updates and verifies finite loss plus a nonzero parameter delta. It is a scaled integration smoke, not a completion claim for full PMetal training.
