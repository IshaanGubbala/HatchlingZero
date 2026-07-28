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
| A5 | data pipeline rebuild | In progress | 100M-token Wikitext reconstruction, streaming packing, resumable cursor, exact/near-duplicate and cross-split contamination reporting now pass; broader mixture remains |
| A6 | PMetal reference implementation | In progress | Rust CPU GDN-2 forward/state/chunk execution now passes alongside Python backward, block/loss, and AdamW parity |
| A7 | training harness rebuild | In progress | deterministic harness now runs a real tiny reference-model forward/loss path with exact resume coverage and CLI smoke verification |
| A8 | explicit PMetal GDN-2 backward | In progress | Python/Rust CPU and native Metal cached reverse scans now return required gradients; PMetal model/training integration remains |
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
- A8 native-Metal backward sub-gate: satisfied for Q/K/V/gate/initial gradients against Torch autograd on a multi-value smoke case
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
  - actual PMetal tensor/device execution and optimizer integration
  - full-model PMetal training path beyond CPU/reference wrappers
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
  - deterministic local text ingestion now discovers approved sources, records provenance/hashes, assigns stable splits, and excludes generated dependency trees
  - large-manifest near-duplicate auditing now uses an inverted shingle candidate index with exact Jaccard verification
  - archived Wikitext-103 JSONL normalization now preserves split boundaries and emits provenance/hash records for token packing
  - bounded-batch tokenizer counting now measures large-corpus token budgets without materializing all IDs
  - Wikitext-103 source budget measured at `196,028,717` tokens with train/validation/test hashes; protocol-aligned length-1024 packed-output gate passes
  - streaming train packing produced `1,524,594` fixed-length sequences and offset-indexed resumable iteration is now test-backed
  - auditable stage runner now executes tiny hybrid/transformer comparisons with checkpoints and explicit smoke/budget-complete flags
  - stage runner now supports explicit CPU/MPS execution and records device identity in stage reports/checkpoints
  - fp16 MPS stage smoke now accumulates validation cross-entropy in fp32 and refuses non-finite validation metrics
  - MPS fp16 path now keeps fp32 master parameters/AdamW state and uses autocast activations; finite one-step throughput is verified
  - staged runner now honors configurable validation cadence; full Stage 1 launch uses the declared 100-step cadence instead of validating every update
  - tiny GDN training recurrence now uses a TorchScript exact scan, preserving token-level math while removing Python dispatch from the MPS path
  - stage checkpoints now use atomic replace semantics, preventing interrupted writes from being mistaken for resumable state
  - stage runner resume restores model/optimizer/RNG/dataset cursor and matches uninterrupted fingerprints
  - stage runner now evaluates and checkpoints a separate validation packed split
  - stage runner now derives sequence length from packed data; protocol-aligned length-1024 packing is tracked separately from length-128 smoke data
  - standalone checkpoint evaluator now reports shared loss/perplexity/token/parameter/throughput metrics
  - locked 301M reference model now has a real/meta forward smoke command with memory and finite-logit reporting
  - real locked-model CPU forward smoke passed at sequence length 1 with finite logits and measured resident memory
  - bounded full-parameter AdamW smoke path now exists with finite-loss/gradient refusal and parameter-change evidence
  - locked 301M one-step AdamW smoke completed with finite loss/gradients and measured 6.5GB resident memory
  - exact locked-model training smoke now supports CPU and Apple MPS execution paths
  - native fused GDN-2 Metal forward kernel now compiles into a Metal library and matches a deterministic runtime smoke case; full-model integration and speed remain unverified
  - native Metal recurrence now matches CPU output/final state at the locked 12-head 64x64 state shape
  - native Metal AdamW first-step parameters/moments now match the NumPy optimizer contract
  - native Metal cached GDN-2 backward gradients now match Torch autograd on a deterministic multi-token case
  - clean MLX locked-topology model surface now exists with scaled GPU forward/state coverage
  - MLX now dispatches a native Metal GDN-2 forward kernel with deterministic parity coverage
  - native-forward MLX model now has a real AdamW two-step smoke with finite loss and parameter-update evidence
  - config-driven PyTorch full topology now matches the locked `301,178,112` parameter target on a meta-device audit
  - model-level recurrent-only chunked state carry now matches full-sequence logits/state in regression coverage

## Next Actions

1. Validate the bounded tiny reference attention policy against the intended PMetal numerical policy.
2. Expand A5 beyond scaffolding into a more complete deterministic dataset pipeline.
3. Deepen A6 from parity wrappers into more realistic PMetal-side execution and optimizer-step parity.
4. Connect A9/A10 to real model parameters, data batches, validation, checkpoints, and PMetal execution.
5. Implement attention KV-cache and PMetal/fused Metal inference after reference equivalence is established.
