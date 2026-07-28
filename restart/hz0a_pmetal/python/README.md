# Python Parity Helpers

This directory holds thin Python-side validation helpers for the PMetal
restart.

The current rule is:

- Python may validate and compare.
- Rust pins the backend-facing API contract.
- Neither side should reintroduce the old mixed legacy training stack.

Current helper:

- `pmetal_reference.py` implements a PMetal-style `gdn2_forward(...)` result
  with outputs, final state, and explicit cache tensors.
