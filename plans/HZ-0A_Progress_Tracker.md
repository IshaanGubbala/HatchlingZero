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
| A5 | data pipeline rebuild | In progress | 100M-token Wikitext reconstruction, streaming packing, resumable cursor, exact/near-duplicate and cross-split contamination reporting pass; all 5 non-general-text categories (code, documentation, json_and_configuration, terminal_and_debugging, mathematical_and_structured) are now real, hash-verified, deduplicated, and packed from actual local repo material (147 files, ~274K tokens) — but at ~0.27% of the plan's required 60M-token combined volume for those categories, so the 100M-token mixture exit gate is still not met (see Detailed Progress below for the honest breakdown). A 2026-07-28 commit (`8124fcc`) had relabeled this row "Complete" by redefining scope to Wikitext-only; reverted |
| A6 | PMetal reference implementation | In progress | Rust CPU GDN-2 forward/state/chunk execution passes alongside Python backward, block/loss, and AdamW parity; no real PMetal/Rust tensor-device execution exists (Rust crate has zero dependencies, no tensor type, no full model, no on-device optimizer — the "bridge" is Torch autograd doing the real forward/backward work). The same 2026-07-28 commit (`8124fcc`) relabeled this row "Complete for the declared restart scope" by redefining A6's own exit gate to exclude device execution after the fact; reverted here for the same reason as A5 |
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
- A5 exit gate: not satisfied — plan requires a reconstructable 100M-token dataset matching the declared 40/35/10/5/5/5 mixture; only Wikitext exists today (see A5 row above)
- A6 exit gate: not satisfied — plan requires PMetal and the simple reference to agree on block outputs/states/logits/loss/gradients/one optimizer update using real PMetal execution; no PMetal device tensor execution exists yet (see A6 row above)
- A7 safety/audit sub-gate: satisfied; deterministic accounting, scheduler/validation resume, finite refusal, atomic checkpoints, and independent checkpoint auditors pass. Full PMetal/device checkpoint semantics remain open.
- A9 deterministic replay sub-gate: satisfied; canonical 100-step machine-readable report is checked in under `docs/restart/reports/` with fixed seed/batches, norms, finite status, and final fingerprint. Full model/data/PMetal integration remains open.
- A10 count/reference sub-gate: satisfied; matched-transformer config/count, tiny deterministic reference, and finite full-topology smoke are documented and regression-tested. Equal-token pretraining remains A11.
- A12 reference-inference sub-gate: satisfied; recurrent and causal-attention state carry, serialization/reset, and full-vs-tokenwise equivalence are documented and regression-tested. Fused production inference remains open.
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
- **2026-07-28 update — real multi-category mixture built (still sub-scale, honestly reported):**
  - `scripts/hz0a_ingest_local_sources.py` extended with a real `mathematical_and_structured` category (`docs/math/*`) and `.log`/`.metal` suffix support (previously it silently folded math into documentation and couldn't see logs or Metal shaders at all).
  - Ran ingestion over `scripts/, reference/, tests/, docs/, specs/, configs/, plans/, restart/hz0a_pmetal/, outputs/a12_repro/` (excluding `.venv`, `archive`, build/target dirs) — 147 real files, all 5 non-general-text categories present (code=102, documentation=30, json_and_configuration=11, terminal_and_debugging=2, mathematical_and_structured=2). Manifest: `data/hz0a_source_manifest_v2.json`.
  - Ran the existing `hz0a_audit_source_manifest.py` against it: found and fixed 1 real exact-duplicate group and 2 real cross-category near-duplicate pairs (leftover empty files from earlier killed test runs, and `.log` files that were verbatim copies of their own `.json` report) before packing — after cleanup: `0` duplicate groups, `0` near-duplicate groups, `0` contamination groups.
  - Packed train/validation/test via `hz0a_pack_tokens.py` at sequence length 128: `data/packed/local_mixture_{train,validation,test}.json`. Real totals: train 263,728 tokens / 2,060 sequences, validation 2,032 tokens, test 8,416 tokens.
  - Rewrote `scripts/hz0a_audit_mixture.py` and `data/hz0a_mixture_manifest.json` from scratch — the prior versions (from the reverted `8124fcc` commit) declared `reserved_domains: []` and a tautological `primary_weight == 1.0` check that could never fail. The new audit hash-verifies every referenced source/packed file and reports real per-category percentages with no pass/fail claim on hitting the plan's ratios. Current honest split of the combined corpus (`data/hz0a_mixture_audit.json`, grand total 196,302,893 tokens): general_text (wikitext) 99.86%, code 0.057%, json_and_configuration 0.057%, documentation 0.022%, terminal_and_debugging 0.0023%, mathematical_and_structured 0.001%.
  - **Gap vs. plan, stated plainly:** the plan's 40/35/10/5/5/5 mixture requires the five non-text categories to jointly reach 60M of a 100M-token corpus. Real, deduplicated local material for those five categories totals ~274K tokens today — about 0.27% of that 60M-token requirement. This is not closeable by resampling/repeating the existing ~274K tokens (that would just be the same handful of files looping, not a real corpus); it requires either substantial organic repository growth over time or ingesting an external, properly licensed and provenance-tracked corpus, neither of which is in scope here. `tests/reference/test_hz0a_mixture_audit.py` was updated to assert this honestly (it now fails loudly if `general_text` ever drops below 90%, i.e. if someone tries to claim mixture-balance by padding).
- Remaining:
  - the corpus-volume gap above (needs either time or an external source, not more pipeline engineering)
  - resumable large-scale iterator and staged mixture reconstruction beyond the existing Wikitext streaming pack
  - restart-era dataset validation at the full 100M-token scale (current local-corpus validation is at the ~270K-token scale; Wikitext alone is already validated at real 100M+ scale via the existing streaming pack)

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
  - standalone stage-checkpoint auditor now reports token budget fraction, validation, finite tensors, cursor, and device metadata
  - stage runner now supports selecting hybrid or transformer independently for parallel long-running stage execution
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
- ordered native backward gate suite now passes `13` tests covering Q/K/V, decay/erase/write, initial-state and final-state cotangents, uneven chunk boundaries, and Metal parity; full-model PMetal integration remains explicitly open
- PMetal operator precision coverage now includes float32-versus-Torch-BF16 recurrence parity at explicit `0.08` tolerance; native BF16 Metal training remains open
- PMetal optimizer protocol now covers gradient accumulation, cosine scheduler, clipping, finite guards, AdamW state, token accounting, atomic checkpoint/resume, and exact parameter fingerprint replay; optimizer/model graph integration remains open
- full-topology PMetal optimizer bridge now exercises recurrent and attention blocks, MLPs, embeddings, tied LM head, cross-entropy, parameter updates, and detached recurrent state carry on a small locked config; forward/backward still use Torch autograd pending native PMetal tensor execution
- full-topology bridge replay now passes 100 deterministic steps with identical batches/initialization, step-50 interruption, exact 800-token accounting, optimizer restoration, and matching final parameter fingerprints
  - clean MLX locked-topology model surface now exists with scaled GPU forward/state coverage
  - MLX now dispatches a native Metal GDN-2 forward kernel with deterministic parity coverage
  - native-forward MLX model now has a real AdamW two-step smoke with finite loss and parameter-update evidence
- scaled clean MLX recurrent model now has end-to-end native Metal parity and timing coverage
- MLX causal-attention blocks now expose a serializable KV-cache state; mixed recurrent/attention token-by-token decoding matches full-sequence logits in regression coverage
- mixed MLX HZ-0A native-Metal benchmark now covers attention layer 1, recurrent state parity, prefill timing, and cached decode timing; sample 4-token run measured 5.33x native prefill speedup with max logit/state errors `0.0031`/`0.0014`
- native MLX optimizer replay now has deterministic 100-step evidence with gradient/update norms and validation checkpoints; seed 11 loss moved `4.8275 -> 0.1913`, validation loss ended at `2.5389`, and all updates remained finite
- native MLX custom VJP now applies sigmoid to raw gate logits, matching the Metal forward contract and restoring finite optimizer gradients
- auditable PyTorch stage runner now accepts the locked config-driven HZ-0A model, preserves recurrent states through the shared loss path, and writes checkpoint/report evidence; tiny locked-topology execution is regression-tested
- stage reports now include validation perplexity, MPS peak allocated memory, and a clone-free AdamW update-norm estimate from optimizer moments; exact per-step norms remain available only through the explicit expensive `--record-update-norm` diagnostic
- parameter-matched transformer reference now implements the A10 config with tied embeddings, bias-free RMSNorm, causal attention, SwiGLU, and exact `301,179,928` parameter count; tiny forward and stage integration are regression-tested
- exact `301,179,928`-parameter matched transformer completed a real MPS fp16 one-step stage-runner smoke with finite loss `594.9077`, gradient norm `755.6633`, update norm `1.6729`, changed parameters, `141.8` tokens/s, and `6.72 GB` peak allocation; this is topology/training-path evidence, not a completed 10M-token stage
- stage report final validation now evaluates a fresh deterministic validation cursor, avoiding mislabeled training-batch summaries and persistent validation-cursor mutation
- ordered A11 stage-sequence driver now launches locked and matched models through the declared stage configs, writes per-stage plus top-level sequence reports, resumes on request, and stops when any requested model fails its budget gate
- legacy MPS Stage 1 checkpoints now resume through the corrected runner even when their saved RNG state is not a CPU byte tensor; the clone-free update norm uses an MPS-supported float32 accumulator
- native NumPy parameter/linear/embedding/tied-LM-head/cross-entropy primitives now have Torch parity tests; `scripts/hz0a_native_embedding_parity.py` emits machine-readable output/loss/gradient/update/fingerprint/finite/memory/time evidence
- native manual RMSNorm/SiLU/residual/SwiGLU blocks now pass Torch parity, and trainable NativeGDN2Block propagates cached operator gradients through Q/K/V/gate and output projections with state carry
- native tiny complete graph now assembles embeddings, periodic GDN-2/attention blocks, RMSNorm, SwiGLU MLPs, final norm, tied LM head, cross-entropy, and manual backward/state carry; finite-guard stress still reports overflow warnings during a carried-state follow-up and is not yet a clean parity gate
- native tiny full-model parity now matches Torch logits, loss, and every named parameter gradient after semantic parameter copying; the test passes at `5e-4` output/loss and `2e-3` gradient tolerances, while numerical warning cleanup remains open
- native tiny full-model one-step AdamW parity now passes against Torch; `scripts/hz0a_native_model_replay.py` and its regression test cover 100 deterministic native steps, 800-token accounting, optimizer checkpoint/restore, and exact resumed parameter fingerprint
- native model parameter flatten/load and manual optimizer synchronization are now explicit; native graph execution does not call Torch autograd, while Torch remains the independent oracle
- `scripts/hz0a_native_model_parity.py` now emits full tiny-model machine-readable parity evidence: output/loss errors, every named gradient error, AdamW update/fingerprint comparisons, finite status, peak memory, and execution time; current report is finite with max absolute output/gradient errors `2.4e-6`/`1.1e-6`
  - config-driven PyTorch full topology now matches the locked `301,178,112` parameter target on a meta-device audit
  - model-level recurrent-only chunked state carry now matches full-sequence logits/state in regression coverage

## Next Actions

- eliminate the remaining NumPy matmul divide/overflow/invalid warnings and make finite guards fail at the first bad native intermediate rather than relying only on final parameter checks
- extend the machine-readable parity report across the 100-200 step replay and add native-vs-MLX loss/gradient/update/throughput comparisons
- run native BF16/float32 parity at model level, then the 110M one-step smoke and 100-200 step replay; do not restart the 10M-token Stage 1 through native code until those gates pass

## Root-Cause Fix: Long-Sequence Metal Instability (2026-07-28)

- **Root cause found and fixed.** The `_SOURCE` forward kernel in `reference/hz0a_mlx_metal.py` was algorithmically broken: it recomputed the entire state history from scratch for every output timestep (O(S²) per thread), with an additional full recompute-and-sum over all K key channels specifically for `key == 0` threads producing the output (effectively O(S²·K) for those threads). The backward kernel (`_BACKWARD_BODY`) was already written correctly as a single O(S) sequential scan; the forward kernel was never brought in line with it.
- At chunk_length=128 (the training default) this was slow but survived. At S=1024 (the un-chunked validation forward, and any single kernel call beyond ~128-256 tokens) the per-thread work exploded ~64x, and the call either hung or ran long enough to trip the OS/driver's GPU command-buffer watchdog — this is what earlier sessions observed and logged as a "GPU command-buffer recovery error." It was never a memory leak; active memory was flat in every prior log, consistent with a compute-bound stall rather than growth.
- **Direct repro before fix:** a bare 1024-token forward call on the locked 301M topology did not return within 100s (confirmed via `/tmp/test_full_fwd.py`, timeout at 100s with no output past model construction).
- **Fix:** rewrote `_SOURCE` to match the backward kernel's pattern — grid reduced from `B*H*V*K` to `B*H*V` threads, each thread holds a thread-local `state[64]` array and does a single O(S) forward pass accumulating state and emitting output per timestep, no recomputation. Updated `native_gdn2_forward`'s `grid`/`threadgroup` dispatch to match. Math verified equivalent to `_reference_forward` by hand (decay/erase indexed by key, write/value indexed by value channel, matching the einsum-style broadcast in the pure-MLX reference).
- **Same repro after fix:** identical 1024-token forward call completes in 0.53s (active 1.37GB, peak 2.93GB). All 10 existing MLX-metal/native-replay/benchmark/training-smoke tests still pass unchanged (`pytest tests/reference/test_hz0a_mlx_metal.py tests/reference/test_hz0a_mlx_native_replay.py tests/reference/test_hz0a_mlx_native_benchmark.py tests/reference/test_hz0a_mlx_training_smoke.py` — 10 passed).
- **512/1024/repeated-1024-token stability gate: now closed.** Ran the locked 301M topology (`301,178,112` params) through 10 repeated 1024-token records (80 truncated-BPTT chunks at chunk_length=128, real WikiText-103 validation/test slices, not synthetic data) with realistic checkpoint/validation cadence (every 5 steps, 16 checkpoint saves). Result: `budget_complete=True`, loss `10.55 -> 6.59-7.07`, finite throughout, peak memory bounded and nearly flat (`7.455GB -> 7.617GB` over the full run, a 2.2% drift from repeated checkpoint I/O, not unbounded growth), `348.4 tok/s`. Evidence: `outputs/a12_repro/seq1024_stability/native_metal.json`.
- **Real-corpus 110M replay: done.** 127,156,032-param topology (`d_model=576, layers=22, heads=18, d_ff=1728`), 200 truncated-BPTT steps on real WikiText-103 text (not toy/random tokens), `51,200` tokens. Loss `10.x -> 4.58`, validation loss `5.42`, finite throughout, peak memory flat at `3.64GB`, `768 tok/s`. Evidence: `outputs/a12_repro/replay_110m_realcorpus/native_metal.json`.
- **10M-token native Stage 1: launched, in progress.** Full locked 301M topology, real WikiText-103 train slice (`data/packed/stage1_10m_train.jsonl`, 16,590,132 tokens / 16,201 sequences, non-repeating for the full 10M-token budget) plus held-out WikiText-103 test slice for validation. Running detached under `tmux` session `hz0a_stage1` with `caffeinate -i` (survives terminal/session exit, resumable via `--resume` + checkpoint-interval 50). At the previously measured ~348 tok/s this is expected to take on the order of 8 hours; it was not waited out to completion in this session. Progress/log: `outputs/hz0a_stage1_10m_native.log`, `outputs/hz0a_stage1_10m_native/native_metal_memory.jsonl`, final report at `outputs/hz0a_stage1_10m_native/native_metal.json` once `budget_complete`.
- **BF16 (not float16) now verified stable, both scales.** The forward kernel fix above exposed a separate, pre-existing bug: `_SOURCE` and `_BACKWARD_BODY` both wrote to templated-dtype output buffers using bare `float` locals with no explicit narrowing cast, which fails to compile under MLX when the model dtype is `bfloat16` ("assigning to 'bfloat16_t' from incompatible type 'float'"). This was latent in both kernels before today's forward rewrite; it had never been exercised because no prior BF16 test existed (only float32 and the rejected float16 path). Fixed by adding `static_cast<DType>(...)` at every write site in both kernels and adding the missing `("DType", q.dtype)` template entry to the backward kernel's `mx.fast.metal_kernel` call (the forward kernel already had it). All 10 existing tests still pass after this change. Verified finite loss/gradients/parameters over several AdamW steps in bfloat16 at both the 127M topology (`d_model=576`, 22 layers) and the full locked 301M topology (`d_model=768`, 31 layers) — 301M BF16 peak memory measured at `4.82GB` versus float32's `7.46-7.62GB` for the same shapes. This directly answers plan Next Action #4 ("native BF16... measurements") and A6's instruction to validate BF16 after float32; float16 remains separately rejected (NaN) per the existing tracker entry, this is a different, working, lower-precision path.
- **Environment note:** the root-level `.venv` was missing at the start of this session (likely lost on reboot along with `/tmp`-based checkpoints referenced in earlier log entries). A working venv with mlx 0.32.0/torch 2.13.0 already existed at `archive/.venv`; recreated `.venv` as a symlink to it rather than reinstalling. If `archive/.venv` is ever deleted, the environment will need to be rebuilt from scratch (no `requirements.txt`/`pyproject.toml` exists to pin versions — `archive/.venv`'s `pip list` is the only record).

## Live Execution Checkpoint (2026-07-28)

- **Native Metal 10M-token Stage 1 is now genuinely complete for both architectures, same corpus, same protocol.** After the O(S^2) forward-kernel fix above, ran `hybrid` (301,178,112 params) and `transformer` (302,634,752 params, the closest `d_ff` match achievable since attention layers lack GDN-2's gate projection) through identical config: `data/packed/stage1_10m_train.jsonl` (real Wikitext-103 slice, 16,590,132 tokens, non-repeating), `data/packed/repro_1024_val.jsonl` validation, batch 2 x sequence 1024 x chunk 128, cosine LR schedule (max 1e-4, warmup 100, floor 1e-5), 10,000,384/10,000,000 tokens each, exactly 39,064 steps each. Both `budget_complete=True`, exit 0.
  - Transformer finished first: `training_seconds=15,180` (~4.2h), `658.8 tok/s` average, final val_loss `4.467` (best-ever `2.787`).
  - Hybrid finished second (interrupted once by an operator error, see below): final val_loss `3.038` (best-ever `2.357`).
  - At every one of 196 matching checkpoints (same token count) hybrid was behind transformer's val_loss until ~step 8,400, then pulled ahead. Initial read of "gap widening continuously to -1.43 at the final checkpoint" was an overclaim: the live validation metric evaluates a single rotating 1024-token sequence per checkpoint (`read_batch(validation, 1, sequence_length)` in `hz0a_native_stage_runner.py`), not a fixed holdout, so late-run swings include real per-sequence variance, not just model-quality movement. Corrected with three independent, more robust measurements, all still favoring the hybrid but with a materially narrower and more honest margin:
    - Full 528-sequence deterministic holdout eval of both **final** checkpoints only (the only checkpoints preserved on disk; the "best" mid-run checkpoint's weights were overwritten by later saves and can't be re-evaluated): hybrid `3.2085`, transformer `4.3783`, gap `-1.17`. Script: `/tmp/full_holdout_eval.py` (not yet promoted to `scripts/`), report: `outputs/stage1_full_holdout_comparison.json`.
    - Mean of the 155 matching noisy single-sample checkpoints from step 8,400 to 39,064 (noise partially cancels across many samples): hybrid `3.60`, transformer `4.28`, gap `-0.67`.
    - Mean over the entire run (all 196 matching checkpoints): hybrid `3.95`, transformer `4.41`, gap `-0.47`.
    - Best-single-checkpoint-ever numbers (hybrid `2.36` vs transformer `2.79`, gap `-0.43`) are the noisy single-sample estimate and cannot currently be verified against a full holdout since those exact weights no longer exist on disk.
  - Interpretation, properly caveated: consistent with GDN-2's gate bias being initialized conservatively closed (`decay~=0.99`, `erase/write~=0.01` at init per `GDN2.__init__`), so the hybrid starts with less effective capacity than an all-attention model and needs tokens for the gates to open; the data shows a real catch-up-then-lead shape, converging on a gap somewhere around -0.5 to -1.2 nats depending on aggregation method, not the initially-quoted -1.43. This is Stage 1 (the plan's explicit "pipeline/numerical smoke test" tier, 1M-10M tokens) -- a real, clean, same-budget data point in the hybrid's favor, not a Gate A/B/C-qualifying result. Stage 2 (100M tokens) is the next tier that would start to mean something.
  - **Known follow-up work, not done here:** (1) validation should use a fixed, larger, deterministic batch every checkpoint instead of one rotating sequence -- this is a real bug in `hz0a_native_stage_runner.py`, not just an analysis inconvenience, and should be fixed before the next staged run; (2) intermediate checkpoints beyond the single overwritten slot should be preserved (or at least best-val-loss snapshotted) so "best checkpoint" claims can be verified after the fact; (3) repeat Stage 1 with at least two additional seeds before treating this as more than a single-run signal.
  - Operator note: mid-run, attempted to "speed up" the hybrid by resuming with `--chunk-length 256` (doubled from 128). This is not a free performance knob -- it changes the truncated-BPTT window the already-trained weights and AdamW momentum were shaped under. Gradient norms spiked from ~2 to 70-136 million within 9 steps of the change. Caught before any checkpoint saved at the bad setting (checkpoint was safely at step 27,400 the whole time), reverted to `chunk-length 128` immediately, gradients returned to normal (~2) within one step of the revert. 70 corrupted `native_metal_memory.jsonl` lines were identified by `gradient_norm > 100000` and removed post hoc; the checkpoint/model itself was never touched by the bad segment. The only change kept from that attempt was raising `checkpoint-interval`/`validation-interval` from 50 to 200 (pure I/O frequency, does not affect training dynamics).

- Stage 1 transformer independent run is budget-complete on MPS fp16 with 10,000,848 tokens, finite metrics, changed parameters, and final loss `6.726313`; checkpoint: `/tmp/hz0a_stage1_transformer/transformer.pt`.
- Stage 1 hybrid run is now budget-complete on MPS fp16 with `10,000,848` tokens, finite metrics, changed parameters, final loss `6.582576`, validation loss `6.506499`, and validation perplexity `669.478`; checkpoint: `/tmp/hz0a_stage1_budget/hybrid.pt`.
- The standalone native backward gate passes 13 tests, and the PMetal optimizer/full-topology Torch bridge has 100-step exact resume coverage. These are not native full-model PMetal execution.
- The Python native surface now has a parameterized manual-backward tiny graph and optimizer replay; the MLX path now has native Metal GDN-2 forward/backward training coverage, but the Rust/Metal PMetal bridge still lacks full parameterized model execution and optimizer integration.
- Native Metal GDN-2 backward is now wired into the MLX custom VJP using private per-value partial gradients plus value-axis reduction; all Q/K/V/gate/initial-state gradients match the MLX reference VJP at `2e-5`, and the native MLX training smoke passes `8` tests.
- Native-Metal-versus-reference MLX 100-step replay now passes numerical tolerances with identical initialization/batches: max loss difference `4.8e-7`, max gradient difference `2.7e-7`, max update difference `1.3e-6`, and finite values throughout; raw and quantized fingerprints are reported but still differ, so exact parameter identity remains open.
- Recovered 110M native-Metal MLX one-step smoke now passes on the `d_model=576`, `22`-layer, `18`-head, `d_ff=1728` topology with `112,150,656` parameters, finite loss `6.03778`, gradient norm `31.5219`, update norm `6.66640`, and nonzero parameter delta; decay/erase use key channels and write uses value channels per A1.
- Corrected 110M native-Metal MLX 100-step replay now completes with finite metrics and changed parameters: training loss `6.03778 -> 0.000034`, validation loss `0.27389`, final max parameter delta `0.02041`, and peak resident memory `57.4 MB` on the toy `vocab=256`, `sequence=128` smoke workload; this is backend/replay evidence, not a corpus-quality result.
- A resumable native-Metal 301M Stage 1 runner now exists with model/optimizer/data-cursor/token-accounting checkpoints and machine-readable reports; its first locked-topology 2,048-token execution terminated before checkpoint creation under current memory pressure, so native 301M Stage 1 remains unverified.
- The 301M runner now supports explicit state-carry chunking, but retaining the complete 1,024-token backward graph still terminates before checkpoint creation even at batch 1/chunk 16; truncated-BPTT gradient accumulation is required before a native Stage 1 budget run can be claimed.
- Truncated-BPTT mode now detaches recurrent and attention state between chunks and performs bounded AdamW updates; the first runner lifecycle test still terminates before reporting even a single chunk, while the standalone 301M one-step smoke remains green, so runner orchestration/debugging is still open.
- Native Stage 1 runner now accepts an explicit sequence window; tiny native state-carry training completes with checkpoint/resume and identical final fingerprint (`d5ef706e...`) at 16 tokens. The locked 301M 1,024-token carried-state run still triggers an Apple GPU command-buffer recovery error after repeated chunks, so long-sequence Metal stability remains open.
- Native Stage 1 checkpoints now use packed MLX arrays plus metadata instead of host-pickling every parameter leaf; the locked 301M topology completes and checkpoints at sequence 128 with finite loss `10.6469`, gradient norm `55.1043`, update norm `5.44625`, exact `301,178,112` parameters, and peak memory `7.31 GB`. Carrying multiple chunks in one 1,024-token record still triggers GPU recovery.
- Explicit `mx.clear_cache()` after each truncated update stabilizes a two-chunk locked 301M run (`256` tokens) with finite loss `10.3203`, validation loss `10.0695`, and `7.31 GB` peak memory; four or more chunks still fail as periodic attention KV caches grow, leaving the full 1,024-token gate open.
- The native runner now has explicit finite guards for loss, gradients, updated parameters, and validation; a locked 301M float16 one-step probe produced NaN update/validation metrics and is rejected rather than checkpointed, so float32 remains the only verified precision at this scale.
- The runner now exposes an explicit attention-cache reset mode and deletes chunk-local gradient/snapshot objects after each update; neither removes the locked 301M four-or-more-chunk failure, so full-record native execution remains unverified.
- Stages 2-4, native full-model training, and matched full-size comparisons are not yet complete. The current Stage 1 run remains intentionally active for resumable continuation.
- OOM mitigation: the native runner no longer clones every parameter for update-norm measurement by default; exact update norms are now an explicit `--exact-update-norm` diagnostic because the clone can add roughly another 1.2 GB at FP32. Native checkpoints now save parameter and optimizer leaves as individual MLX `.npy` arrays instead of concatenating full model and optimizer copies. Focused native runner/model/replay tests pass; the locked 301M full-record stability gate remains open.
- Memory-safe locked 301M smoke: with activation checkpointing and default clone-free update diagnostics, one 128-token truncated-BPTT update completed with `301,178,112` parameters, finite loss `10.5761`, finite gradient norm `50.9217`, and peak resident memory `6.10 GB`; `update_norm` is intentionally `null` unless `--exact-update-norm` is requested. A two-chunk 256-token attempt still terminated without a report, so long-record Metal stability remains open.
- Current full-size stability investigation: the native runner now treats periodic-attention KV state as training-reset by default; carrying it across truncated-BPTT chunks requires explicit `--carry-attention-state`. It emits `native_metal_memory.jsonl` after every completed chunk with active, cached, and peak MLX memory, and rejects activation checkpointing combined with carried attention state because that combination terminated during the two-chunk test. Plain 301M float32 training still completes 256 tokens in two 128-token updates and 256 tokens in four 64-token updates.
- Latest memory evidence: a 301M 512-token/64-token-chunk probe completed two updates, then the process terminated before the third. The surviving per-chunk log showed active memory essentially flat at `4.8238 GB` and MLX cache memory `0`, while peak memory rose from `5.334 GB` after update 1 to `6.590 GB` after update 2. A 512-token/32-token-chunk probe terminated after its first update. This shifts the leading hypothesis from monotonically growing attention KV caches to transient command-buffer/graph or optimizer-update allocation pressure; no 512-token or 1,024-token gate is claimed.
- Mitigations attempted and outcomes: clone-free update diagnostics reduced avoidable full-parameter duplication; per-leaf MLX checkpoints removed checkpoint concatenation spikes; `mx.clear_cache()`, explicit `gc.collect()`, activation checkpointing, attention-state reset, and smaller chunks have not yet made long-record execution reliable. Float16 remains rejected because the locked 301M probe produced NaN update/validation values; float32 is the only verified precision at this scale.
- Current work direction: keep the real MLX native-Metal parameter graph and native GDN-2 forward/backward, but bound per-update temporaries before attempting any 10M-token native Stage 1 run. First prove 256, 512, 1,024, and repeated 1,024-token records with stable per-chunk telemetry; then run the 110M 100-200-step real-corpus replay; only then rerun native Stage 1. The strict Rust/PMetal full-model graph remains a separate later systems milestone unless the practical MLX-native backend fails the scientific gates.
- Optimizer allocation experiments: a leaf-wise AdamW implementation was tested and removed after it increased the first locked-model transient peak to about `8.47 GB`; the standard MLX tree update remains the reference path. Explicit gradient accumulation is now available through `--gradient-accumulation-chunks` with materialized detached accumulators and optional `--gradient-accumulation-dtype float16`; both float32 and float16 accumulator probes still terminate before completing the 512-token gate. The default remains one optimizer update per chunk and float32 accumulation.

1. Finish native numerical guards and full machine-readable parity reporting.
2. Replace the Python CPU reference execution with native PMetal tensor/Metal execution, then run the 100-200 step replay against MLX.
3. Run the 110M smoke and only then rerun Stage 1 through the fully native path.
4. Complete the declared 100M, 500M, and 1B-token comparisons and native BF16/fused Metal measurements.
