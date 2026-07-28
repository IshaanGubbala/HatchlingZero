# HZ-0A A10 Matched Transformer Audit

Date: July 28, 2026

## Locked Comparison Configuration

The matched baseline is defined in `configs/hz0a_transformer_matched.json` and uses:

- vocabulary `24,576`
- width `768`
- `12` heads with head dimension `64`
- `29` dense causal-attention layers
- `d_ff=3196`
- tied token embedding/output head
- no LM-head bias
- `fp32` reference and BF16 training target
- AdamW with `lr=1e-4`
- the same tokenizer and source-data manifest paths as HZ-0A

The parameter-count tool reports `301,179,928` parameters versus the HZ-0A target of `301,178,112`, an absolute difference of `1,816` (`0.000603%`).

## Verified Reference

`reference/hz0a_transformer_reference.py` provides a deterministic tiny dense-transformer language model with the same tied embedding, RMSNorm, causal-attention, SwiGLU, and next-token loss conventions used by the HZ-0A reference.

## What This Proves

- exact transformer count is reproducible from a checked-in config
- the baseline is parameter-matched within a documented tolerance
- tiny transformer initialization, logits, and shared LM loss are deterministic

## What This Does Not Yet Prove

- full 301M-parameter transformer execution in PMetal
- matched training launcher/checkpoint cadence
- equal-token pretraining comparison
- quality, memory, throughput, prefill, or decode results

## A10 Assessment

A10 is in progress. The architecture/count/reference sub-gate is satisfied; shared full-scale training protocol and comparison results remain pending.
