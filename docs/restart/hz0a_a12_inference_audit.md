# HZ-0A A12 Inference Audit

Date: July 28, 2026

## Verified Reference Path

`reference/hz0a_inference.py` provides a recurrent-only prefill and tokenwise decode API with explicit state reset and serialization. `scripts/hz0a_inference_benchmark.py` reports prefill and decode timings separately.

The same module now provides an append-only causal-attention KV cache with tokenwise decode and serialization.

Verified command:

```bash
python3 scripts/hz0a_inference_benchmark.py --sequence-length 16
python3 -m pytest -q tests/reference/test_hz0a_inference.py
```

Observed in the current environment:

- batch size: `2`
- sequence length: `16`
- prefill throughput: approximately `27,876.6` tokens/second
- tokenwise decode throughput: approximately `9,735.3` tokens/second
- full-sequence versus tokenwise maximum logit difference: `0.0`

State serialization/resume and reset equivalence are covered. Models containing attention blocks are rejected by this recurrent-only decode API until an attention KV-cache implementation exists.

The shared tiny reference attention path also has a bounded numerical policy; the A7 harness smoke run is warning-free after this guard was added.

## What This Proves

- recurrent full-sequence and tokenwise decode are numerically equivalent
- state carry, serialization, and reset behavior are deterministic
- causal-attention full-sequence and KV-cached tokenwise outputs are numerically equivalent
- prefill and decode are measured separately

## What This Does Not Yet Prove

- PMetal-native or fused Metal execution
- PMetal-native or fused Metal execution
- end-to-end inference speedup over a transformer
- chunked fused-kernel equivalence or device-level memory measurements

## A12 Assessment

A12 reference-correctness sub-gate is satisfied for recurrent decode and causal-attention KV-cache decode. Fused Metal implementation and performance comparison remain incomplete.
