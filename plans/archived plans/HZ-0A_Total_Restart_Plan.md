# HZ-0A Total Restart Plan

## Starting Point

This plan assumes the active branch contains only:

- `README`
- the HATCHLING-ZERO master plan
- `.git` history

It does **not** assume that any working implementation remains.

The following must be treated as unavailable until explicitly recovered or rebuilt:

- MLX model code
- PMetal integration
- tokenizer files
- datasets
- training scripts
- fused Metal kernels
- tests
- checkpoints
- metrics
- evaluation harnesses
- audit scripts

Git history is a reference source, not a trusted implementation. Old code may be inspected and selectively copied only after it is understood, tested, and shown to match the new specification.

---

## Objective

Rebuild HZ-0A from zero as an approximately 300M-parameter recurrent-hybrid language model with:

- GDN-2 recurrent layers
- periodic causal attention
- a clean PMetal training implementation
- explicit, training-grade backward support
- fused Metal inference
- a matched transformer baseline
- reproducible data, training, evaluation, and checkpointing

HZ-0A should answer:

> Does the GDN-2 hybrid provide a useful quality, memory, throughput, or inference-latency tradeoff against a parameter-matched transformer?

HZ-0A must exclude:

- associative memory
- session-local fast weights
- adaptive recurrence
- MoE
- persistent user memory
- tool-use post-training

Those belong to later stages.

---

# Phase A0 — Repository archaeology and specification recovery

## Goal

Recover the actual intended architecture and identify which historical ideas are trustworthy enough to carry forward.

## Tasks

Inspect Git history for:

- the last coherent GDN-2 equations
- model configuration files
- parameter-count logic
- layer-ordering decisions
- tokenizer configuration
- data-mixture assumptions
- learning-rate findings
- gradient-accumulation bug history
- fused-forward kernel design
- known numerical issues
- evaluation scripts
- historical checkpoints and metrics references

Create a recovery report with three categories:

### Confirmed

Ideas supported by code, tests, logs, or repeated experiments.

Examples may include:

- approximate 292M historical parameter count
- `1e-4` as the safest tested learning rate
- pure-MLX training as the historically proven path
- fused Metal forward as an inference-only optimization
- the historical gradient-accumulation bug

### Uncertain

Ideas mentioned but not sufficiently validated.

### Rejected

Dead ends, misleading labels, broken implementations, or unsupported claims.

Do not restore an old branch wholesale.

## Deliverables

- `docs/restart/hz0a_history_audit.md`
- `docs/restart/hz0a_recovered_spec.md`
- list of historical commits worth referencing
- list of obsolete or misleading claims

## Exit gate

The architecture can be described precisely without depending on undocumented old code.

---

# Phase A1 — Freeze the new HZ-0A specification

Create one authoritative model specification.

Initial target:

```text
Name: hz0a_300m
Target size: approximately 300M parameters
Tokenizer: to be rebuilt or retrained
Architecture: GDN-2 recurrent blocks plus periodic causal attention
Context: begin at 1K, validate at 2K–4K
Training backend: PMetal
Numerical reference: simple Python/NumPy or MLX implementation
Inference backend: fused Metal after correctness is established
```

Specify:

- vocabulary size
- hidden size
- layer count
- recurrent-to-attention ratio
- head count
- key and value dimensions
- MLP expansion
- normalization
- residual order
- state shape
- state initialization
- gate parameterization
- output-head tying
- precision policy
- initialization scheme

Write the GDN-2 recurrence mathematically in the repository.

The new launcher must eventually print:

- exact parameter count
- architecture hash
- model dimensions
- tokenizer hash
- dataset-manifest hash
- optimizer settings
- effective batch tokens
- seed
- backend
- precision

## Exit gate

A versioned specification produces one deterministic parameter count near the intended 300M target.

---

# Phase A2 — Build a tiny mathematical reference

Before PMetal, implement the smallest possible readable reference version.

Use:

- Python with NumPy, or
- ordinary MLX operations without custom kernels

Prioritize clarity over speed.

Implement:

1. one GDN-2 step
2. full-sequence GDN-2 recurrence
3. recurrent state initialization
4. state carry across chunks
5. reset behavior
6. one HZ block
7. tiny multi-layer language model
8. loss computation

Reference tests must cover:

- T=1
- short sequences
- non-power-of-two lengths
- chunked versus full-sequence equivalence
- state reset
- repeated-state carry
- finite outputs
- deterministic outputs

## Exit gate

The architecture has a transparent executable definition that does not depend on PMetal or custom Metal code.

---

# Phase A3 — Derive and validate backward mathematics

Do not begin with a fused Metal backward kernel.

First derive the reverse recurrence explicitly.

Backward must produce gradients for:

- Q
- K
- V
- decay and gate terms
- erase and write terms
- initial recurrent state
- any learned projection parameters around the recurrence

Build gradient validation in this order:

1. automatic differentiation through the simple reference
2. finite differences on tiny tensors
3. manual backward using ordinary tensor operations
4. comparison between autodiff and manual backward

Test:

- random small shapes
- multiple sequence lengths
- extreme gate values
- BF16-sensitive cases
- final-state cotangent propagation
- chunk boundaries

## Deliverables

- `docs/math/gdn2_forward.md`
- `docs/math/gdn2_backward.md`
- `tests/reference/test_gdn2_gradients.py`

## Exit gate

Manual backward matches autodiff and finite differences within predefined tolerances.

---

# Phase A4 — Rebuild the tokenizer

Assume the old 24K tokenizer artifact is unavailable.

Recover from Git history only:

- tokenizer type
- normalization rules
- special tokens
- training-data categories
- intended vocabulary size
- code and tool syntax handling

Then retrain it from a versioned tokenizer corpus.

Requirements:

- approximately 24K vocabulary unless the new audit justifies changing it
- deterministic training
- explicit special-token table
- code-friendly tokenization
- JSON and tool-schema coverage
- saved tokenizer hash
- encode/decode round-trip tests
- unknown-token and whitespace tests

## Deliverables

- tokenizer training script
- tokenizer corpus manifest
- tokenizer model files
- tokenizer audit report

## Exit gate

The tokenizer is reproducible from source data and passes round-trip and coverage tests.

---

# Phase A5 — Rebuild the data pipeline

Assume no usable dataset pipeline remains.

Build:

- document ingestion
- provenance tracking
- license metadata
- exact deduplication
- near-duplicate removal
- train/validation/test splitting
- contamination checks
- tokenizer application
- packed-sequence generation
- deterministic shuffling
- resumable iteration
- dataset manifests and hashes

Initial mixture target:

```text
40% general and technical text
35% code
10% documentation and API material
5% JSON and configuration
5% terminal and debugging data
5% mathematical and structured data
```

Use staged datasets:

```text
1M–10M tokens: pipeline smoke tests
10M–100M tokens: optimization and correctness
100M tokens: first serious pilot
500M tokens: architecture pilot
1B–3B tokens: full HZ-0A training target
```

## Exit gate

A 100M-token dataset can be reconstructed exactly from manifests and produces stable packed batches.

---

# Phase A6 — Build the PMetal reference implementation

Reimplement HZ-0A using ordinary PMetal operations before writing fused kernels.

Components:

- embeddings
- RMSNorm
- linear projections
- GDN-2 recurrence
- causal attention
- MLP
- residual routing
- LM head
- recurrent-state API
- loss function

Validate PMetal against the tiny reference on:

- block outputs
- recurrent states
- logits
- loss
- gradients
- one optimizer update

Use float32 for reference checks, then BF16.

## Exit gate

PMetal and the simple reference agree within documented tolerances.

---

# Phase A7 — Rebuild the training harness

The new harness must not inherit old ambiguity around steps and accumulation.

Required counters:

```text
microbatch_count
optimizer_step
tokens_seen
effective_batch_tokens
epoch_or_data_pass
```

Required behavior:

- gradient accumulation independent of `max_steps`
- deterministic seeding
- resumable dataset position
- model checkpoints
- optimizer checkpoints
- scheduler checkpoints
- peak-memory logging
- gradient norms
- parameter-update norms
- NaN/Inf refusal gates
- periodic validation
- checkpoint audit command
- clean interruption and resume
- configuration snapshot per run

For the historical test shape:

```text
batch 2 × sequence 256 × accumulation 4
= 2,048 tokens per optimizer step
```

Add a unit test that proves this accounting.

## Exit gate

A deterministic short run resumes exactly and token accounting is correct.

---

# Phase A8 — PMetal explicit GDN-2 backward

Use PMetal’s explicit forward-cache/backward design.

Target interface:

```text
gdn2_forward(inputs, initial_state)
    -> outputs, final_state, backward_cache

gdn2_backward(d_outputs, d_final_state, backward_cache)
    -> gradients for all inputs and initial_state
```

Use chunked checkpointing:

```text
Forward:
save state every 32–128 tokens

Backward:
recompute within each chunk
walk backward through the chunk
pass state gradient to the prior chunk
```

Implementation sequence:

1. ordinary PMetal forward and manual backward
2. Metal forward plus PMetal backward
3. fully fused Metal forward and backward

Do not proceed to step 3 until step 1 is proven.

## Exit gate

PMetal backward passes gradient checks and a short optimizer replay.

---

# Phase A9 — Deterministic optimizer replay

Recreate the historical scientific-validity test from scratch.

Run:

- same initialization
- same batch order
- same tokenizer
- same data
- same optimizer
- same scheduler
- `lr=1e-4`
- 100–200 optimizer steps

Compare:

- loss by tokens seen
- validation loss
- gradient norms
- update norms
- parameter-update cosine similarity
- recurrent-state norms
- checkpoint integrity
- peak memory
- throughput

## Exit gate

The PMetal training path produces stable, reproducible learning and no unexplained divergence.

---

# Phase A10 — Build the matched transformer

Create a parameter-matched transformer in the same codebase.

Match:

- tokenizer
- dataset
- token budget
- precision
- optimizer
- schedule
- effective batch tokens
- checkpoint cadence
- evaluation cadence

Publish:

- exact HZ-0A parameter count
- exact transformer parameter count
- architecture details
- training compute
- data budget

## Exit gate

Both architectures can run under the same training protocol.

---

# Phase A11 — Training stages

Run in order:

## Stage 1: 10M-token validation

Purpose:

- catch pipeline and numerical failures

## Stage 2: 100M-token pilot

Purpose:

- validate convergence
- validate resume
- compare PMetal and reference behavior
- test data mixture

## Stage 3: 500M-token architecture pilot

Purpose:

- initial HZ versus transformer signal

## Stage 4: 1B–3B-token full comparison

Purpose:

- meaningful architecture evaluation

Do not make capability claims from the smoke-test stages.

---

# Phase A12 — Fused Metal inference

Only after training correctness is stable:

- implement PMetal-native fused GDN-2 forward
- support recurrent state across decode calls
- validate full-sequence versus token-by-token equivalence
- validate chunk boundaries
- add reset and serialization tests
- measure prefill separately from decode
- measure kernel speed separately from end-to-end speed

The historical approximately 9.5× result is a target to reproduce, not a retained result.

## Exit gate

The fused backend matches the reference numerically and improves end-to-end inference.

---

# HZ-0A completion definition

HZ-0A is complete when:

1. The architecture is fully specified.
2. The tokenizer and data pipeline are reproducible.
3. The simple reference implementation passes forward and gradient tests.
4. PMetal forward and backward match the reference.
5. Training accumulation, checkpoints, resume, and token accounting are correct.
6. The approximately 300M model completes a meaningful pretraining run.
7. A parameter-matched transformer completes the same protocol.
8. Quality, memory, training throughput, prefill, and decode results are reported.
9. Fused Metal inference is numerically equivalent and measurably faster.
10. Historical claims are reproduced or explicitly retired.

---

# Recommended branch structure

```text
restart/pmetal-hz0a
├── docs/restart
├── docs/math
├── src/reference
├── src/tokenizer
├── src/data
├── src/pmetal/model
├── src/pmetal/gdn2
├── src/pmetal/training
├── src/pmetal/inference
├── tests/reference
├── tests/pmetal
├── configs
└── scripts
```

---

# First concrete milestone

The first milestone is not a 300M run.

It is:

> A tiny HZ-0A model with a written GDN-2 specification, a readable reference implementation, validated gradients, a rebuilt tokenizer, and one deterministic PMetal optimizer step that matches the reference.
