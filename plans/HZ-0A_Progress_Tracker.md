# HZ-0A Progress Tracker

Updated: July 28, 2026

## Mission

Rebuild HZ-0A from zero as an approximately 300M-parameter recurrent-hybrid LM with GDN-2 recurrence, periodic causal attention, a clean PMetal training path, and a matched transformer baseline.

## Current Status

- Overall phase: `A6 in progress`
- Confidence level: high for archaeology findings, low for any legacy implementation reuse
- Active artifacts:
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_history_audit.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_recovered_spec.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a1_spec.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a4_tokenizer_spec.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a4_tokenizer_audit.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a5_data_audit.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a6_pmetal_audit.md`

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| A0 | history audit + recovered spec | Complete | Restart docs written on July 28, 2026 |
| A1 | authoritative `hz0a_300m` spec | Complete | JSON spec, math doc, and parameter-count script created on July 28, 2026 |
| A2 | tiny mathematical reference | Complete | NumPy reference model and tests now exist and pass |
| A3 | backward derivation + validation | Complete | backward math doc and gradient tests now exist and pass |
| A4 | tokenizer rebuild | Complete | tokenizer artifact, corpus manifest, runtime wrapper, and audit now exist |
| A5 | data pipeline rebuild | In progress | source-manifest audit and token-packing scaffolding now exist |
| A6 | PMetal reference implementation | In progress | fresh PMetal workspace now has Rust contract tests plus operator/block/loss Python parity checks |
| A7 | training harness rebuild | Not started | deterministic counters, resume, audit, config snapshotting still needed |
| A8 | explicit PMetal GDN-2 backward | Not started | forward-cache/backward path still pending |
| A9 | deterministic optimizer replay | Not started | replay and reproducibility gate still pending |
| A10 | matched transformer | Not started | must be parameter-matched and launcher-compatible |

## Confirmed Historical Findings To Carry Forward

- A real legacy ~110M HZ-0A baseline existed.
- A later PMetal/MLX run labeled "110M" was actually about 292M.
- Old streaming decode was known-bad and disabled.
- Separate decay / erase / write gate parameterization is part of the intended identity.
- Conservative LR evidence clusters around `1e-4` to `2e-4`.

## Rejected Historical Inputs

- legacy streaming decode implementation as a correctness baseline
- HZ-0B memory logic as part of HZ-0A
- mislabeled 292M checkpoint lineage as a 110M comparison point
- optimistic "production-ready" claims from legacy status docs

## A1 Definition Checklist

- [x] exact vocabulary size
- [x] exact hidden size
- [x] exact layer count
- [x] recurrent-to-attention ratio
- [x] head count
- [x] key dimension
- [x] value dimension
- [x] MLP expansion ratio
- [x] normalization type
- [x] residual ordering
- [x] recurrent state shape
- [x] state initialization rule
- [x] gate parameterization
- [x] output-head tying policy
- [x] precision policy
- [x] initialization scheme
- [x] deterministic parameter-count derivation
- [x] architecture hash scheme

## Exit Gates

- A0 exit gate: satisfied
- A1 exit gate: satisfied
- A2 exit gate: satisfied
- A3 exit gate: satisfied for the recurrence core
- A4 exit gate: satisfied
- Program rule: no PMetal kernel work until A2 and A3 exist and pass tests

## Next Actions

1. Expand A5 beyond scaffolding into a more complete deterministic dataset pipeline.
2. Deepen A6 from parity wrappers into more realistic PMetal-side execution and optimizer-step parity.
3. Begin A7 training-harness rebuild once the PMetal-side parity surface is broad enough to support end-to-end counters and resume logic.
