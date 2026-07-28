# HZ-0A Progress Tracker

Updated: July 28, 2026

## Mission

Rebuild HZ-0A from zero as an approximately 300M-parameter recurrent-hybrid LM with GDN-2 recurrence, periodic causal attention, a clean PMetal training path, and a matched transformer baseline.

## Current Status

- Overall phase: `A5/A7/A11/A12 in progress`
- Confidence level: high for archaeology findings, low for any legacy implementation reuse
- Last verified checkpoint: `July 28, 2026 - Rust PMetal CPU forward, A12 recurrent inference, A11 staged protocol, A10 matched transformer, A9 replay, A8 chunked backward, A6 parity, A5 packing, and A7 safety pass`
- Active artifacts:
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_history_audit.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_recovered_spec.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a1_spec.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a4_tokenizer_spec.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a4_tokenizer_audit.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a5_data_audit.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a6_pmetal_audit.md`
  - `/Users/ishaangubbala/Documents/Training/docs/restart/hz0a_a7_harness_audit.md`

## Phase Tracker

| Phase | Deliverable | Status | Evidence / Notes |
| --- | --- | --- | --- |
| A0 | history audit + recovered spec | Complete | Restart docs written on July 28, 2026 |
| A1 | authoritative `hz0a_300m` spec | Complete | JSON spec, math doc, and parameter-count script created on July 28, 2026 |
| A2 | tiny mathematical reference | Complete | NumPy reference model and tests now exist and pass |
| A3 | backward derivation + validation | Complete | backward math doc and gradient tests now exist and pass |
| A4 | tokenizer rebuild | Complete | tokenizer artifact, corpus manifest, runtime wrapper, and audit now exist |
| A5 | data pipeline rebuild | In progress | validated manifests, seeded packing, resumable cursor, exact/near-duplicate and cross-split contamination reporting now exist |
| A6 | PMetal reference implementation | In progress | Rust CPU GDN-2 forward/state/chunk execution now passes alongside Python backward, block/loss, and AdamW parity |
| A7 | training harness rebuild | In progress | deterministic harness now runs a real tiny reference-model forward/loss path with exact resume coverage and CLI smoke verification |
| A8 | explicit PMetal GDN-2 backward | In progress | Python and Rust CPU cached reverse scans now return required gradients; device-side PMetal execution remains |
| A9 | deterministic optimizer replay | In progress | 100-step fixed-seed AdamW replay now produces exact repeated metrics and parameter fingerprint |
| A10 | matched transformer | In progress | checked-in 301.180M dense baseline config/count and deterministic tiny reference now pass |
| A11 | training stages | In progress | shared 4-stage protocol plus real multi-parameter tiny hybrid/transformer training, validation, and exact resume smoke now pass; full pretraining remains |
| A12 | fused Metal inference | In progress | recurrent and causal-attention KV-cache reference equivalence, state serialization/reset, and separate timing now pass; fused backend remains |

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
- A7 partial gate: satisfied for deterministic accounting, checkpoint/restart, and model-aware scalar-loss stepping
- A7 safety sub-gate: satisfied for finite-value refusal and checkpoint accounting audit
- A8 ordinary-reference backward sub-gate: satisfied for all Q/K/V/gate/initial-state gradients
- A8 chunked-recompute sub-gate: satisfied for uneven full-vs-chunked gradient equivalence
- A8 native-CPU backward sub-gate: satisfied for Rust flat-buffer gradients and finite-difference Q validation
- A9 deterministic optimizer sub-gate: satisfied for fixed tiny-parameter 100-step replay
- A10 matched-count/reference sub-gate: satisfied within 1,816 parameters and shared tiny LM-loss conventions
- A11 stage-protocol sub-gate: satisfied for ordered budgets and shared optimizer/data/checkpoint settings
- A11 tiny-training sub-gate: satisfied for real gradients, parameter updates, deterministic comparison, and exact resume
- A12 recurrent-reference inference sub-gate: satisfied for tokenwise equivalence, state serialization/reset, and separate prefill/decode timing
- A12 attention-cache sub-gate: satisfied for full-vs-tokenwise causal attention equivalence and cache serialization
- Program rule: no PMetal kernel work until A2 and A3 exist and pass tests

## Detailed Progress

### A5 Data Pipeline

- Completed:
  - deterministic source manifest scaffold
  - source-manifest audit script
  - token-packing script and packed-data audit output
  - required-field, split, source-existence, and exact-content duplicate validation
  - normalized-shingle near-duplicate reporting with fixture coverage
  - cross-split contamination reporting with exact-leakage fixture coverage
  - seeded document ordering and reproducible packing regression tests
  - resumable seeded packed-dataset iterator with exact cursor restore tests
- Remaining:
  - broader corpus ingestion automation
  - near-duplicate removal and contamination checks
  - resumable large-scale iterator and staged mixture reconstruction
  - restart-era dataset validation beyond current 16.9K-token local scaffold

### A6 PMetal Reference

- Completed:
  - clean restart workspace under `restart/hz0a_pmetal`
  - Rust crate split for kernel and bridge
  - parity coverage for recurrence operator, blocks, and loss path
  - deterministic AdamW optimizer-state and update-norm parity test
  - explicit cached GDN-2 backward with independent torch-oracle coverage
  - chunked backward with state-cotangent propagation across boundaries
  - executable Rust CPU GDN-2 forward with shape validation and chunk carry tests
  - executable Rust CPU GDN-2 backward with required gradients and finite-difference validation
  - 100-step fixed-seed optimizer replay with exact repeated output and parameter hash
  - matched-transformer config/count tool and deterministic tiny reference
  - machine-validated 10M/100M/500M/1B staged-training protocol
  - real PyTorch tiny hybrid/transformer smoke training with checkpoint/resume parity and validation/throughput/parameter metrics
  - recurrent reference inference benchmark with zero full-vs-tokenwise logit difference
  - causal-attention KV-cache equivalence and serialization tests
- Remaining:
  - fuller execution path beyond parity wrappers
  - backward-path implementation for GDN-2
  - tighter coupling to harness/optimizer replay requirements

### A7 Harness

- Completed:
  - deterministic token accounting and historical effective-batch tracking
  - config snapshot output with serialized model-shape metadata
  - checkpoint save/load and exact resume test coverage
  - real tiny-model forward pass wired into microbatch execution
  - real next-token cross-entropy and accumulated scalar gradient stepping
  - CLI smoke run verified from `restart/hz0a_harness.py`
  - non-finite refusal gates for logits, loss, gradients, and optimizer updates
  - checkpoint audit command validating finite payloads and token/record accounting
- Current limitations:
  - only a scalar `model_logit_scale` is train-updated today
  - scheduler remains a bookkeeping stub
  - bounded reference attention policy still needs validation against the intended PMetal numerical policy
  - PMetal/device checkpoint audit is still pending; the tiny PyTorch multi-parameter checkpoint now has an independent integrity auditor
  - scheduler and separate validation-split execution are now checkpointed and resume-tested
  - stage data-budget gate now reports available versus required tokens and refuses under-budget launches
  - native fused GDN-2 Metal forward kernel now compiles into AIR; runtime equivalence and speed remain unverified

## Next Actions

1. Validate the bounded tiny reference attention policy against the intended PMetal numerical policy.
2. Expand A5 beyond scaffolding into a more complete deterministic dataset pipeline.
3. Deepen A6 from parity wrappers into more realistic PMetal-side execution and optimizer-step parity.
4. Connect A9/A10 to real model parameters, data batches, validation, checkpoints, and PMetal execution.
5. Implement attention KV-cache and PMetal/fused Metal inference after reference equivalence is established.
