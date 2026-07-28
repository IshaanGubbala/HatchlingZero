# HZ-0A PMetal Restart

This workspace is the fresh A6 restart surface for the HZ-0A PMetal path.

It is intentionally narrow:

- Rust defines the restart-facing kernel and bridge contracts.
- Python provides parity-oriented validation against the NumPy oracle.
- The current goal is not fused performance yet. The current goal is a clean,
  testable interface that matches the locked A1 spec and the passing A2/A3
  reference work.

## Current Contents

- `PLAN.md` documents the PMetal restart thesis and phased target.
- `KEEP_FROM_LEGACY.md` records which archived assets are safe to consult.
- `crates/hz0a-pmetal-kernel/` pins down the operator/cache API shape.
- `crates/hz0a-pmetal-bridge/` pins down the bridge/runtime summary surface.
- `python/pmetal_reference.py` implements a PMetal-style forward/cache API in
  NumPy for parity checks.

## Current Success Bar

This workspace is useful if:

1. the Rust contracts compile and test cleanly
2. the Python parity harness matches the NumPy oracle
3. future PMetal operator work can plug into the same forward/cache interface
