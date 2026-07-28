# HZ-0A PMetal Restart

This directory is the clean restart track for HZ-0A.

It exists because the legacy project accumulated multiple partially-overlapping
implementations, status narratives, and experiment branches. The goal here is
to restart around a single backend direction:

- training backend target: PMetal-style explicit forward-cache/backward operators
- reference path during bring-up: existing MLX/Python reference implementations
- inference target: fused Metal path once numerically validated

This restart workspace is intentionally separate from the legacy code under
`src/hz0/`.

## Scope

This restart keeps only what is still useful:

- audited evidence from the legacy HZ-0A runs
- mathematically trusted GDN-2 references
- direct benchmark and checkpoint audits
- a clear phased migration plan

It does not assume the legacy training stack is the final architecture.

## Layout

- `PLAN.md`: canonical restart plan
- `KEEP_FROM_LEGACY.md`: exact legacy artifacts to preserve/reference
- `STATUS.md`: live restart status
- `Cargo.toml`: isolated Rust workspace
- `crates/hz0a-pmetal-kernel/`: future fused GDN-2 operator crate
- `crates/hz0a-pmetal-bridge/`: future bridge layer for Python/MLX interop
- `python/`: thin Python-side helpers for validation and migration

## Success condition

The restart is successful when HZ-0A has:

1. A trusted reference GDN-2 path
2. A PMetal-style explicit backward path
3. A parameter- and token-honest training story
4. A real memory-task win
