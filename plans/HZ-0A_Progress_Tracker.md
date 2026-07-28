# HZ-0A Progress Tracker

Updated: July 28, 2026

## Mission

Rebuild HZ-0A from zero as an approximately 300M-parameter recurrent-hybrid LM with GDN-2 recurrence, periodic causal attention, a clean PMetal training path, and a matched transformer baseline.

## Current Status

- Overall phase: `A0 complete`, `A1 next`
- Confidence level: high for archaeology findings, low for any legacy implementation reuse
- Active artifacts:
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_history_audit.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_recovered_spec.md`

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| A0 | history audit + recovered spec | Complete | Restart docs written on July 28, 2026 |
| A1 | authoritative `hz0a_300m` spec | Not started | Needs exact dims, parameter formula, hashes, precision rules |
| A2 | tiny mathematical reference | Not started | NumPy or plain MLX path only |
| A3 | backward derivation + validation | Not started | Must precede fused backward |
| A4 | tokenizer rebuild | Not started | Byte-level vs 24K BPE must be decided explicitly |
| A5 | data pipeline rebuild | Not started | Must produce hashed manifests |
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

- [ ] exact vocabulary size
- [ ] exact hidden size
- [ ] exact layer count
- [ ] recurrent-to-attention ratio
- [ ] head count
- [ ] key dimension
- [ ] value dimension
- [ ] MLP expansion ratio
- [ ] normalization type
- [ ] residual ordering
- [ ] recurrent state shape
- [ ] state initialization rule
- [ ] gate parameterization
- [ ] output-head tying policy
- [ ] precision policy
- [ ] initialization scheme
- [ ] deterministic parameter-count derivation
- [ ] architecture hash scheme

## Exit Gates

- A0 exit gate: satisfied
- A1 exit gate: pending
- Program rule: no PMetal kernel work until A2 and A3 exist and pass tests

## Next Actions

1. Draft the final `hz0a_300m` spec document.
2. Add the mathematical recurrence to repo docs.
3. Stand up the tiny reference code and tests.
