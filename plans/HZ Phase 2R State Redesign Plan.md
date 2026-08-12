# HZ Phase 2R — BDH State Redesign

Inserted 2026-08-11 between Phase 2 (exact streaming BDH, done) and the
original Phase 3 (state compression via sparsity/low-rank/quantization)
in `plans/HatchlingZero_Reality_Plan.md`. Direct response to
`docs/restart/hz0h_phase2_streaming_state_size_results.md`'s real,
measured finding: BDH's exact fp32 state is already 1.95x-3.31x the
model's own weight bytes. A 30% reduction (the original Phase 3 exit
gate) does essentially nothing against that — need a structural,
4-10x-class reduction before inference-side optimization (quantization,
sparsity) is worth layering on top.

## Objective

Turn `State_BDH ≈ 1.95-3.31x weights` into `State_HZ ≤ 0.5x weights`
(stretch: `≤ 0.25x`). Needs ~4-7x structural compression to clear 0.5x,
ideally 8-12x, before quantization adds another ~2-4x on top (Phase 3's
already-shipped INT8 result, `docs/restart/hz0h_phase3_state_quantization_results.md`,
0% measured quality loss on passkey retrieval).

## Sub-phases

- **2R-A — Lock the exact-state oracle.** Already satisfied:
  `bdh_stream_chunk`/`init_bdh_states` (H2) is the exact reference every
  compressed variant compares against — real CE, passkey/reassignment
  accuracy, state bytes, decode latency, not just LM loss.
- **2R-B — Value-bottleneck state ("HZ-BDH-VB").** Project the VALUE
  (`V = x`, shape `D`) down to a small `d_state` before it enters the
  state: `S_t = S_{t-1} + K_t^T P(V_t)`, read via `O(Q_t S_t)`, `P: D →
  d_state`, `O: d_state → D`. State becomes `N × d_state` instead of
  `N × D`. NOT mathematically exact BDH anymore (real, explicit
  divergence from the upstream architecture, named accordingly rather
  than presented as a kernel optimization). Test `d_state = D, D/2, D/4,
  D/8`.
- **2R-C — Grouped depth state.** BDH's `encoder`/`encoder_v`/`decoder`
  are already shared/tied across depth, but each layer still needs its
  own historical state — real source of the size blowup. Share ONE
  state across a GROUP of `g` consecutive layers instead of one state
  per layer (`6 layers → 3/2/1 groups`), with small per-layer read/write
  projections so shared layers can still specialize. Called "Grouped
  Synaptic State."
- **2R-D — Combine B+C.** The user's own highest-priority pick:
  **2 depth-state banks (`6→2` groups) + `D/4` value-state width** — a
  real, structural 12x state reduction (3x from grouping, 4x from the
  bottleneck) before any quantization, which would put
  `1.95-3.31x weights` at roughly `0.16-0.28x weights`.
- **2R-E — Quantize the state.** Only after B/C, not before (state is
  more numerically sensitive than static weights, continuously
  updated). INT8 already shipped and validated in isolation on the
  EXACT (uncompressed) state (`docs/restart/hz0h_phase3_state_quantization_results.md`)
  — real next step is applying the same technique on top of a B/C-
  compressed state, not starting over.
- **2R-F — Sparse/block-sparse state.** Only after B-E — sparse systems
  can look good on paper (FLOPs) while losing on real wall-clock
  (`plans/HatchlingZero_Reality_Plan.md`'s own Risk 2 warning, already
  seen once this session: `docs/restart/hz0h_phase1_kv_cache_bdh_results.md`'s
  BDH KV-cache path not consistently beating the O(D²) state despite a
  better asymptotic story). Needs a strictly bounded capacity per block
  or a long enough session makes "sparse" state dense again.
- **2R-G — Teacher-distill from exact BDH.** `L = L_LM + λ1·KL(p_BDH ||
  p_HZ) + λ2·L_hidden`; initialize `P`'s down-projection via PCA/SVD of
  real exact-BDH value activations rather than random init, so the
  bottleneck starts in the subspace BDH actually uses rather than an
  arbitrary one.

## Experiment matrix (25M scale, per the plan; this session starts smaller)

| Variant | Depth-state groups | State width | Precision |
| --- | --- | --- | --- |
| Exact BDH | 6 | D | fp32/bf16 |
| VB-2/VB-4/VB-8 | 6 | D/2, D/4, D/8 | fp32/bf16 |
| GS-3/GS-2/GS-1 | 3/2/1 | D | fp32/bf16 |
| Combined (e.g. GS-2 + VB-4) | 2 | D/4 | fp32/bf16 |
| Best + INT8 | — | — | int8 |

Take the best two per family before combining, per the user's own
"keeps the causal attribution clean" instruction.

## Hard gates

- **Memory**: state ≤ 0.5x weight bytes (min), ≤ 0.25x (preferred).
- **Quality**: LM CE degradation ≤3% at small scale; no catastrophic
  memory-task regression (passkey/reassignment); no recurrent
  instability (NaN/Inf, unbounded state growth).
- **Inference**: must actually improve peak RAM; decode speed should not
  be significantly worse (real risk — INT8's own dequant/requant
  overhead wasn't free either, per the Phase 3 doc's own disclosed gap).
- **KV crossover**: target state-RAM beating a real Transformer KV-cache
  by ~2K context, not the 3K-15K found for the exact fp32 state in
  `docs/restart/hz0h_phase2_streaming_state_size_results.md`.

## Fallback if compression loses too much quality: hybrid recent-KV + compressed BDH

Exact small KV-cache/attention over the last 512-2K tokens, compressed
BDH synaptic state for everything older — bounded long-term memory
without the Transformer's unboundedly-growing KV-cache, and without
committing 100% of memory to BDH's own (currently oversized) state
immediately. Real, credible fallback, not attempted unless 2R-B through
2R-E fail the hard gates above.

## Explicitly deferred until this resolves

Ternary, MoE, adaptive/variable-depth reasoning, alternative BPTT
training laws, additional HZ-0B/HZ-0D-style memory, RL — none of these
address the core blocker (state currently ~2-3.3x the model's own
weight bytes, defeating one of HatchlingZero's stated goals). Per the
user's own framing: right now the architecture is "a small static model
+ an enormous dynamic state," and that has to resolve before anything
else is worth spending compute on.

## Status

2R-A: done (pre-existing). 2R-B: starting now (`reference/hz0h_bdh_vb_torch.py`,
in progress this session). Everything else: not started.
