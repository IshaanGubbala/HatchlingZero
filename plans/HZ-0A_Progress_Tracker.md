# HZ-0A Progress Tracker

Updated: July 28, 2026

## Mission

Rebuild HZ-0A from zero as an approximately 300M-parameter recurrent-hybrid LM with GDN-2 recurrence, periodic causal attention, a clean PMetal training path, and a matched transformer baseline.

## Current Status

- Overall phase: `A3 complete`, `A4 in progress`
- Confidence level: high for archaeology findings, low for any legacy implementation reuse
- Active artifacts:
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_history_audit.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_recovered_spec.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a1_spec.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a4_tokenizer_spec.md`

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| A0 | history audit + recovered spec | Complete | Restart docs written on July 28, 2026 |
| A1 | authoritative `hz0a_300m` spec | Complete | JSON spec, math doc, and parameter-count script created on July 28, 2026 |
| A2 | tiny mathematical reference | Complete | NumPy reference model and tests now exist and pass |
| A3 | backward derivation + validation | Complete | backward math doc and gradient tests now exist and pass |
| A4 | tokenizer rebuild | Complete | tokenizer artifact, corpus manifest, runtime wrapper, and audit now exist |
| A5 | data pipeline rebuild | In progress | source-manifest audit and token-packing scaffolding now exist |
| A6 | baseline transformer rebuild | Not started | Must be parameter-matched and launcher-compatible |
| A7 | PMetal training implementation | Not started | Correctness before performance |
| A8 | fused Metal inference path | Not started | Inference-only optimization until validated |
| A9 | evaluation + comparison suite | Not started | Must answer the HZ-0A research question fairly |

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
- Program rule: no PMetal kernel work until A2 and A3 exist and pass tests

## Next Actions

1. Finish A4 by producing a real corpus manifest, tokenizer artifact, and tokenizer audit.
2. Begin A5 data-pipeline rebuild once tokenizer source-of-truth files exist.
3. Use the reference and gradient suites as the numerical oracle for A6 PMetal parity checks.
