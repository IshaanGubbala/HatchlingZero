# HZ-0A PMetal Restart Plan

Date: July 28, 2026

## Intent

The HZ-0A restart uses PMetal as the future training backend target, but the
first requirement is numerical honesty rather than fused speed.

This workspace exists to keep the PMetal work aligned with:

- the locked A1 model specification
- the A2 NumPy oracle
- the A3 backward derivation and gradient tests

## Immediate A6 Scope

Build a parity-oriented PMetal reference surface with:

- explicit forward input contract
- explicit backward-cache contract
- final-state contract
- one Python parity implementation that matches the NumPy oracle
- one Rust contract crate that pins the future operator shape

## Not Yet In Scope

- fused Metal kernels
- end-to-end training
- full transformer baseline
- large-scale checkpoint replay

Those remain later HZ-0A phases.
