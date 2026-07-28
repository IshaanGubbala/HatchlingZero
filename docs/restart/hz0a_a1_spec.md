# HZ-0A A1 Specification

Date: July 28, 2026

## Status

This is the authoritative A1 specification for the HZ-0A restart.

Machine-readable source:

- `/Users/ishaangubbala/Documents/Training/specs/hz0a_300m_a1.json`

Parameter-count tool:

- `/Users/ishaangubbala/Documents/Training/scripts/hz0a_param_count.py`

Mathematical forward definition:

- `/Users/ishaangubbala/Documents/Training/docs/math/gdn2_forward.md`

## Model Identity

- Name: `hz0a_300m`
- Version: `a1.v1`
- Architecture hash: `3ebbb729537831bed9b2f21c5ae82178ceca815ad0fbcd83ad7246ef8b6d51b1`
- Target size: `301,178,112` parameters
- Target size (M): `301.178112`

## Locked Dimensions

- Vocabulary size: `24,576`
- Train context length: `1,024`
- Validation context lengths: `1,024`, `2,048`, `4,096`
- Model width `d_model`: `768`
- Total layers: `31`
- Heads: `12`
- Key width per head `d_k`: `64`
- Value width per head `d_v`: `64`
- MLP hidden width `d_ff`: `2,304`

## Layer Schedule

HZ-0A uses `31` residual blocks total:

- recurrent GDN-2 blocks: `25`
- exact causal attention blocks: `6`

The attention layers are fixed at 0-indexed positions:

- `4, 9, 14, 19, 24, 29`

All remaining layers are recurrent GDN-2 blocks.

This keeps attention periodic without making it dense enough to dominate the architecture.

## Residual Block Order

Each recurrent block uses:

1. pre-norm RMSNorm
2. GDN-2 recurrent mixer
3. residual add
4. pre-norm RMSNorm
5. SwiGLU MLP
6. residual add

Each attention block uses the same residual ordering, replacing the recurrent mixer with exact causal attention.

The model ends with:

- final RMSNorm
- tied LM head projection

## State Contract

Each recurrent layer owns a carried state:

- shape: `[batch, 12, 64, 64]`
- semantic layout: `[batch, heads, d_v, d_k]`
- initialization: zeros

State carry is explicit across chunks. Reset returns the state to all zeros.

## Gate Contract

The recurrent mixer uses six projection groups:

- query
- key
- value
- decay logits
- erase logits
- write logits

Gate semantics:

- decay: per-head, per-key-channel retention
- erase: per-head, per-key-channel removal of prior content
- write: per-head, per-value-channel strength of new content insertion

The canonical recurrence is defined in `/Users/ishaangubbala/Documents/Training/docs/math/gdn2_forward.md`.

## Precision Policy

- A2 reference implementation: `fp32`
- Early A3 gradient validation: `fp32`
- Intended training policy: `bf16` activations with `fp32` master weights, optimizer state, reductions, and gradient accumulation
- Recurrent state accumulation remains `fp32` until PMetal correctness is proven

## Initialization Policy

- token embedding: truncated normal, mean `0`, std `0.02`
- linear weights: truncated normal, mean `0`, std `0.02`
- LM head: tied to token embedding
- LM head bias: disabled

Gate biases are intentionally non-neutral:

- decay bias: `+4.59512` so `sigmoid(bias) ~= 0.99`
- erase bias: `-4.59512` so `sigmoid(bias) ~= 0.01`
- write bias: `-4.59512` so `sigmoid(bias) ~= 0.01`

This preserves the legacy lesson that neutral `0.5` gate starts are too unstable for restart work.

## Tokenizer Contract

The tokenizer is not yet built, but the architecture contract is fixed now:

- tokenizer rebuild happens in A4
- tokenizer must land at `24,576` vocabulary items
- the launcher must later print a tokenizer hash and dataset-manifest hash alongside the architecture hash

## Parameter Count

The A1 count is deterministic and script-backed.

Count assumptions:

- tied embedding/output weights
- no LM-head bias
- recurrent in-projection includes `q, k, v, decay, erase, write`
- SwiGLU MLP uses `gate_proj`, `up_proj`, and `down_proj`

Script:

```bash
python3 /Users/ishaangubbala/Documents/Training/scripts/hz0a_param_count.py
```

Expected total:

- `301,178,112` parameters

## Why This Shape

This specification keeps the architecture meaningfully recurrent while staying close to the intended 300M target without inheriting the mislabeled 292M Phase 14 lineage.

The design choices intentionally preserve:

- explicit GDN-2 identity
- periodic causal attention
- clean state semantics
- tied embedding/output head discipline
- room for 1K training and 2K-4K validation

## A1 Exit Assessment

The A1 exit gate is satisfied if:

- the JSON spec remains authoritative
- the parameter-count script reproduces the locked count
- future implementations conform to the math and shape contracts here

As of July 28, 2026, the specification is locked and ready for A2 reference implementation work.
