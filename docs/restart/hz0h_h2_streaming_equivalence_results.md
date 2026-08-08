# HZ-0H H2: streaming-state equivalence, real result

Date: 2026-08-08. H0 found `bdh.py` has no explicit `rho`/state variable -- the paper's state-space framing is a claimed equivalence, not literal code. H2's job is to prove or refute that equivalence directly. Done here via a real derivation, not assumed.

## The derivation

BDH-GPU's attention has no softmax and no `K^T·1` normalization (confirmed in H0/H1: raw `scores @ V`). That makes the causal linear attention exactly decomposable into a running outer-product state -- the same reformulation linear-attention-as-RNN methods (linear transformers, RWKV, GLA) use:

```
out[t] = QR[t] @ S[t],   S[t] = sum_{i<t} KR[i] (x) V[i]
S[t+1] = S[t] + KR[t] (x) V[t]
```

This is an exact algebraic identity (`sum_n QR[t,n]*(sum_i<t KR[i,n]*V[i,d]) = sum_i<t (sum_n QR[t,n]*KR[i,n])*V[i,d]`), not an approximation. `S` is a real, concrete candidate for what the paper calls `rho`.

## Real results (`reference/hz0h_bdh_streaming.py`, `tests/reference/test_hz0h_bdh_streaming.py`, 13 tests)

**Algebraic equivalence, float64 (CPU -- MLX has no float64 GPU support), at all 4 of H2's required lengths (1, 16, 128, 1024):**
- Token-by-token streaming vs. parallel: max abs diff < 1e-6 at every length.
- Arbitrary chunked streaming (ragged, non-dividing chunk sizes) vs. parallel: max abs diff < 1e-6 at every length.
- Streaming vs. chunked (against each other, not just both vs. parallel): max abs diff < 1e-6 at every length.

All three confirm the equivalence is exact, not coincidentally close -- an early version of this test compared a float32-computed parallel result upcast to float64 against a genuinely-float64 streaming computation, which showed a real but misleading ~7e-4 discrepancy at T=1024; fixed by recomputing the parallel form from true float64 inputs with a genuinely float64 `freqs` buffer (RoPE's trig functions carry float32 rounding regardless of what dtype they're later combined with) -- the corrected, true apples-to-apples comparison is the <1e-6 result reported above.

**Practical float32 precision** (the precision the real model runs at):

| T | max abs diff | max relative diff |
| --- | --- | --- |
| 1 | 0.0 | 0.0 |
| 16 | 0.0428 | 0.00071 |
| 128 | 0.1734 | 0.00069 |
| 1024 | 0.6268 | 0.00086 |

Real finding: naive token-by-token float32 streaming accumulates real numerical drift from the parallel form (absolute error grows with T, as expected for any running accumulator), but the *relative* error stays roughly stable at ~0.07-0.09% from T=16 through T=1024 -- it does not blow up or compound explosively. Not a bug (proven exact at float64 above) -- a genuine, bounded float32 accumulation-order sensitivity, real and disclosed rather than hidden behind a loose test tolerance.

## What H2 establishes

- The paper's state-space/streaming-equivalence claim for BDH-GPU's attention is real and provable, not just asserted -- a genuine outer-product running state (`S`) is mathematically exact, confirmed at float64 across all required lengths and both streaming forms (pure token-by-token and arbitrary chunk boundaries).
- Float32 streaming is practically usable (~0.07-0.09% relative error, not exploding) but not bit-exact -- a real caveat for anyone building an actual streaming BDH inference path, disclosed with real numbers rather than assumed away.

## What H2 does not establish

- This only covers BDH-GPU's *attention* mechanism. The full `BDH.forward` also has the ReLU-sparse encoder/decoder projections and LayerNorms every layer -- those are already per-token/parallelizable without any special streaming derivation needed (no cross-position dependency outside attention), so this was the one real piece requiring proof; not independently re-verified end-to-end through a full multi-layer streaming forward pass here.
- No test of `reset`/`serialization`/`resume` semantics for the streaming state `S` itself (H2's own stated scope includes this) -- real, disclosed remaining work, not done in this pass.
