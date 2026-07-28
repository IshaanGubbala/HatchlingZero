# HZ-0A History Audit

Date: July 28, 2026

## Scope

This audit is the Phase A0 recovery artifact for the HZ-0A total restart. It treats the archived repository and git history as evidence sources, not as trusted implementation.

## Starting Assumption

The active restart branch intentionally removed the working tree. Historical material is available under `/Users/ishaangubbala/Documents/Training/archive` plus git history. Every legacy result below is classified as `Confirmed`, `Uncertain`, or `Rejected` based on direct artifacts.

## Confirmed

### 1. A real parameter-matched HZ-0A-style 110M baseline existed

Evidence:
- `/Users/ishaangubbala/Documents/Training/archive/configs/hz0a-mac-110m.yaml`
- `/Users/ishaangubbala/Documents/Training/archive/configs/hz0a-mac-110m-tuned.yaml`
- `/Users/ishaangubbala/Documents/Training/archive/outputs/hz0a-mac-110m-fair/config.snapshot.json`
- `/Users/ishaangubbala/Documents/Training/archive/docs/status/audit-step2153.md`

Recovered facts:
- Historical "110M" runs used `d_model=576`, `n_layers=22`, `n_heads=18`, `d_ff=1728`, `attention_every=4`.
- The tuned run used `lr=2e-4` with `grad_accum_steps=4`.
- The paired transformer baseline used the same width/depth/head config shape.
- The audit document explicitly distinguishes a correctly sized PyTorch-era `109,937,664` parameter checkpoint from a later mislabeled MLX checkpoint.

### 2. The later PMetal/MLX "110M" Phase 14 run was actually about 292M params

Evidence:
- `/Users/ishaangubbala/Documents/Training/archive/docs/status/audit-step2153.md`
- commit `c83b1fe` (`Add HZ-0A checkpoint audit script + step-2153 findings doc`)
- `/Users/ishaangubbala/Documents/Training/archive/scripts/audit_checkpoint.py`

Recovered facts:
- The checkpoint audit was created specifically because a `hz0a_110m` safetensors checkpoint was suspiciously large.
- The resulting conclusion was that the run was approximately `292.04M` parameters, mislabeled by about `2.656x`.
- The audit found no optimizer pollution and no duplicate-content anomaly, so the size mismatch was architectural/configuration mismatch, not checkpoint corruption.

### 3. The old repo already knew streaming decode was not trustworthy

Evidence:
- `/Users/ishaangubbala/Documents/Training/archive/src/hz0/model_port/mlx_gdn2_lm.py`
- `/Users/ishaangubbala/Documents/Training/archive/src/hz0/validation/HONEST_STATUS.md`

Recovered facts:
- `decode_step()` in the MLX model was explicitly deprecated and raises at runtime.
- The deprecation text says tokens `0-1` diverged from full-batch behavior while token `2+` converged.
- Therefore the historical MLX decode path is not a valid implementation reference for restart work.

### 4. A readable reference-style GDN-2 path existed and had tests for chunk/state behavior

Evidence:
- `/Users/ishaangubbala/Documents/Training/archive/tests/test_hybrid_lm.py`
- `/Users/ishaangubbala/Documents/Training/archive/docs/architecture.md`

Recovered facts:
- The old test suite included:
  - recurrent state scan equivalence against a manual loop
  - initial-state carry tests
  - chunked vs full-sequence equivalence
  - NumPy vs torch reference comparisons
- The architecture notes describe a `gdn2_ref` backend intended as a dense, auditable reference.
- This is trustworthy as evidence that the project valued a mathematical reference before optimization, even if the exact implementation must be rebuilt.

### 5. Separate decay / erase / write gate parameterization was a stable design target

Evidence:
- `/Users/ishaangubbala/Documents/Training/archive/docs/architecture.md`
- `/Users/ishaangubbala/Documents/Training/archive/src/hz0/validation/HONEST_STATUS.md`
- commit `7130ddf` (`Fix Phase 14: safe GDN-2 gate init + hard finite guards + split-aware data`)
- vendored upstream kernels under `/Users/ishaangubbala/Documents/Training/archive/vendor/GatedDeltaNet-2/lit_gpt/gdn2_ops/`

Recovered facts:
- Legacy HZ-0A differentiated decay, erase, and write gates rather than using a single update gate.
- Gate initialization was numerically important; the Phase 14 fix explicitly changed bias initialization to avoid recurrent-state explosion.
- Any restart spec should preserve explicit gate separation and define initialization deliberately.

### 6. The safest historically observed learning-rate zone was low and conservative

Evidence:
- `/Users/ishaangubbala/Documents/Training/archive/configs/hz0a-mac-110m.yaml`
- `/Users/ishaangubbala/Documents/Training/archive/configs/hz0a-mac-110m-tuned.yaml`
- `/Users/ishaangubbala/Documents/Training/archive/src/hz0/validation/phase1_complete.py`
- commit `7130ddf`

Recovered facts:
- Historical successful configs cluster around `1e-4` to `2e-4`.
- The restart should treat `1e-4` as the safest legacy anchor and `2e-4` as a historically useful but not universally proven tuned value.

### 7. Tokenizer strategy changed over time and is not frozen

Evidence:
- `/Users/ishaangubbala/Documents/Training/archive/src/hz0/experiment_manifest_cli.py`
- `/Users/ishaangubbala/Documents/Training/archive/data/tokenizer/hz_24k.json`
- `/Users/ishaangubbala/Documents/Training/archive/scripts/prepare_tokenizer_corpus.py`
- commit `f220838` (`Update to use 24K BPE tokenizer instead of char-level`)

Recovered facts:
- Older experiments used byte-level tokenization in some paths.
- Later work introduced a `24K` BPE tokenizer and tokenizer-corpus preparation scripts.
- Tokenizer choice was not stable enough to carry forward unexamined.

## Uncertain

### 1. Exact historical equations used in every HZ-0A run

Why uncertain:
- Multiple backends existed: fallback recurrence, `gdn2_ref`, MLX ports, vendored upstream kernels.
- The repository contains plan and status prose that spans more than one implementation generation.

Consequence:
- We should recover the intended recurrence mathematically from the upstream GDN-2 formulation plus the tests, not trust any single old module as canonical.

### 2. The historical 36M and 110M quality claims

Why uncertain:
- The repo contains both optimistic summary docs and later corrections.
- `HONEST_STATUS.md` explicitly walks back several claims, especially around memory and streaming.
- Many reported wins were based on short runs, single seeds, or synthetic tasks.

Consequence:
- Legacy metrics can inform priorities, but not architectural claims.

### 3. Whether PMetal backward was ever training-grade

Why uncertain:
- The restart plan says backward support must be rebuilt explicitly.
- The historical materials include Metal wrappers, kernels, and training claims, but also repeated notes that backward was incomplete, stubbed, or fallback-based in key paths.

Consequence:
- Treat PMetal backward as unproven until re-derived and numerically validated.

### 4. Which old checkpoints, if any, should be reused directly

Why uncertain:
- Some checkpoints are clearly tied to mislabeling or unstable backend generations.
- Some are valid artifacts for comparison, but not necessarily safe warm starts for a clean restart.

Consequence:
- Use old checkpoints as audit/reference objects, not training seeds for the restart.

## Rejected

### 1. "Production-ready" or "streaming validated" status for the old HZ-0A path

Rejected because:
- `/Users/ishaangubbala/Documents/Training/archive/src/hz0/validation/HONEST_STATUS.md`
- `/Users/ishaangubbala/Documents/Training/archive/src/hz0/model_port/mlx_gdn2_lm.py`

The old repo itself documents that this was overstated.

### 2. Any HZ-0A definition that includes HZ-0B scratchpad memory

Rejected because:
- The restart plan explicitly excludes associative memory, session-local fast weights, and adaptive memory features from HZ-0A.
- The legacy memory path was also called out as unvalidated and design-flawed.

### 3. The mislabeled 292M Phase 14 run as evidence for 110M-vs-transformer claims

Rejected because:
- The audit directly disproves the label.
- Any downstream comparison using it as "110M" would be scientifically invalid.

### 4. Streaming decode code as a baseline to optimize

Rejected because:
- It is known-bad and explicitly disabled.
- Restart should rebuild streaming only after full-sequence correctness and backward math are established.

## Historical Commits Worth Referencing

- `c83b1fe` — checkpoint audit and the 110M-vs-292M size correction
- `7130ddf` — safe gate initialization, finite guards, split-aware data
- `f220838` — tokenizer shift toward 24K BPE
- `b46efdc` — tokenizer phase completion marker
- `385b95a` — PMetal restart workspace scaffold

These commits are worth reading during implementation, but not restoring wholesale.

## Obsolete Or Misleading Claims

- "110M" Phase 14 MLX checkpoint label
- "streaming validated"
- "production-ready"
- any HZ-0A claim that depends on HZ-0B memory behavior
- any direct scientific conclusion based on single-seed short-budget memory probes

## A0 Exit Assessment

Phase A0 is satisfied enough to proceed: the intended HZ-0A direction is clear, but the implementation must be rebuilt from a new specification and a tiny mathematical reference, not by reviving the old tree.
