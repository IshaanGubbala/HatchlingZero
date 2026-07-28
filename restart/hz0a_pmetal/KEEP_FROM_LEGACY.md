# Keep From Legacy

Only these legacy artifacts should be treated as canonical inputs to the
restart.

## Status / audit documents

- `/Users/ishaangubbala/Documents/Training/docs/master-plan-status-2026-07-28.md`
- `/Users/ishaangubbala/Documents/Training/docs/audit-step2153.md`
- `/Users/ishaangubbala/Documents/Training/docs/hz0a-audit.md`
- `/Users/ishaangubbala/Documents/Training/docs/hz0a-benchmark-report-2026-07-26.md`

## Direct HZ-0A evidence

- `/Users/ishaangubbala/Documents/Training/docs/hz0a-step300-direct.json`
- `/Users/ishaangubbala/Documents/Training/docs/hz0a-step325-direct.json`
- `/Users/ishaangubbala/Documents/Training/docs/hz0a-memory-probe-associative-step325.json`

## Trusted mathematical references

- `/Users/ishaangubbala/Documents/Training/src/hz0/metal_gdn2/reference/gdn2_numpy.py`
- `/Users/ishaangubbala/Documents/Training/src/hz0/metal_gdn2/reference/gdn2_mlx.py`
- `/Users/ishaangubbala/Documents/Training/src/hz0/metal_gdn2/reference/gdn2_streaming.py`

## Metal implementation references to study, not blindly inherit

- `/Users/ishaangubbala/Documents/Training/src/hz0/metal_gdn2/kernels/`

## Explicitly not canonical

The following are useful history, but should not define the restart:

- old root progress/status markdowns now in `docs/archive/`
- mixed legacy PyTorch training scripts under `src/hz0/`
- ambiguous "110M" Phase 14 output directory naming
- scratchpad/HZ-0B experimental branches unless deliberately re-imported
