# HZ-0C C8 PMetal Attention Results

Status note (2026-08-04): this initial forward-parity note is superseded by
`docs/restart/hz0c_c8_pmetal_backward_and_parity_results.md` and
`docs/restart/hz0c_c8_model_level_integration_results.md`, which record the
subsequently completed backward, Python parity, CPU/GPU FFI, and grouped
dispatch work.

Date: 2026-08-03.

The dependency-free PMetal CPU reference for conditional causal attention now
has deterministic numerical coverage in
`restart/hz0a_pmetal/crates/hz0a-pmetal-kernel/src/lib.rs`:

- exact scalar causal reference values, including output bias
- triggered-query and triggered-key masking
- batch isolation
- non-triggered output behavior
- invalid shape rejection

`cargo test -p hz0a-pmetal-kernel` passes all **7** unit tests and doc tests.
The GPU crate now also exposes `MetalConditionalAnchorAttention::forward`, a
correctness-first native Metal dispatch using the same flat parameter arrays.
The GPU-vs-CPU parity fixture passes with maximum absolute error below `2e-3`;
the complete GPU suite passes 6 unit tests, 5 decode tests, and 3 full-block
tests. The CPU reference now also exposes an explicit parameterized backward
contract and passes finite-difference checks for input, QKV projections, output
projection, and both bias families. This closes forward parity and the CPU
backward oracle for the small fixture. Metal backward dispatch, model-level
integration, grouped/cache-optimized dispatch, and the Python-to-PMetal
machine-readable report remain open.
