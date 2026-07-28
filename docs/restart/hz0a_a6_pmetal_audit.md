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

## What This Does Not Yet Prove

- actual PMetal tensor execution
- optimizer-update parity
- end-to-end PMetal training

## A6 Current Assessment

A6 is meaningfully in progress, but not complete. The current milestone is a contract-and-parity bring-up across operator, block, and tiny-model-loss surfaces, not a finished PMetal reference implementation.
