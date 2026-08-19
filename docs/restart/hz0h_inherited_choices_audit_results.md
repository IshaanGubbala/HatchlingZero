# Auditing BDH's Inherited Choices: Width Is ~2x Oversized, and Softmax+Temperature Beats Upstream Attention

Date: 2026-08-18. Status: real, multi-seed results on a scaled-down
local (MPS) setup. **One confirmed quality win over upstream BDH, one
confirmed ~2x FLOP saving, and three inherited choices vindicated by
real measurement rather than assumption.** Full-scale CUDA confirmation
is dispatched but not yet returned.

## Why this audit happened

Every prior optimization attempt in this project either changed BDH's
*execution* (wide-GEMM, bmm encoder_v -- real wins, same math) or added
a *new mechanism* (routing, factorization -- 3 of 4 lost). Nobody had
questioned the constants BDH itself was built from. The project's
discipline was faithfulness to `github.com/pathwaycom/bdh` -- correct,
and it produced a verified oracle -- but fidelity was never followed by
"is this choice actually optimal?"

A telling detail that motivated the whole audit: upstream's own
`BDHConfig` default is `mlp_internal_dim_multiplier=128`, while this
project has long run `32`. A large, undocumented 4x deviation was
already made. There was no fidelity purity left to protect.

## Part 1: the FLOP accounting that pointed here

`scripts/hz0h_bdh_cost_breakdown_profile.py` (analytic half, runs
anywhere; the CUDA timing half is dispatched and pending):

```text
BDH   per level: 186.83 GFLOP  x8 levels = 1.495 TFLOP/forward
Matched Transformer:            x6 layers = 0.126 TFLOP/forward
BDH / Transformer FLOP ratio: 11.90x
```

Per-level FLOP breakdown:

```text
encoder            27.6%
encoder_v          27.6%
decoder            27.6%   -> three wide projections = 82.8%
attention_scores   13.8%
attention_values    3.4%   -> attention total       = 17.2%
```

Two real consequences:

1. **BDH does 11.90x the Transformer's FLOPs but measures only ~5.3x
   slower in real training** (`hz0h_phase_f_same_gpu_comparison_results.md`),
   implying BDH already achieves BETTER hardware efficiency than the
   Transformer. The remaining gap is real arithmetic -- ARCHITECTURE,
   not execution debt -- so further layout/fusion remaps have little
   headroom. **Honest caveat**: Phase F's BDH trained under the
   2->4->6->8 depth curriculum, so its average training depth was below
   8 and the true training-time FLOP ratio is lower than 11.90x
   (roughly ~7x), making the efficiency edge smaller than a naive
   11.90/5.3 would suggest. Direction holds; magnitude is softer.
2. Attention is NOT the dominant cost despite running at the expanded
   2048 width. The three wide projections are, and they scale LINEARLY
   with `mlp_internal_dim_multiplier` -- which is what Part 2 attacks.

## Part 2: the width is roughly 2x oversized (3 seeds, confirmed)

`scripts/hz0h_bdh_width_flop_frontier_local.py`, MPS, fp32,
`n_embd=256, n_head=4, n_layer=4, seq=256, 5M tokens`, scaled depth
curriculum. Note `n_embd=256/n_head=4` gives `N=2048` per head at
`mult=32` -- the exact same latent width as production, so the swept
quantity is directly comparable.

Seed 7 full frontier:

| arm | params | val loss | vs Transformer | FLOPs vs TFMR | secs |
|---|---:|---:|---:|---:|---:|
| `mult=32` (canonical) | 6.42M | 1.8862 | **-0.0963** | 9.43x | 183 |
| **`mult=16`** | **3.28M** | **1.9124** | **-0.0701** | **4.86x** | 102 |
| `mult=8` | 1.70M | 2.0007 | +0.0182 | 2.57x | 64 |
| `mult=4` | 0.92M | 2.1202 | +0.1377 | 1.43x | 44 |
| matched Transformer | 4.28M | 1.9825 | -- | 1.00x | 37 |

Seed robustness for the decisive 32-vs-16 comparison:

```text
seed 7: mult32=1.8862  mult16=1.9124  gap +0.0262
seed 8: mult32=1.8794  mult16=1.9029  gap +0.0235
seed 9: mult32=1.8876  mult16=1.9016  gap +0.0140
```

**Real conclusion**: halving the latent width costs only ~0.014-0.026
validation loss (consistent in sign across 3 seeds) while cutting FLOPs
~2x, and `mult=16` still beats the matched Transformer by ~0.07-0.08
with FEWER parameters than it (3.28M vs 4.28M). The frontier decays
gently then accelerates (+0.026, then +0.088, then +0.120 per halving);
the knee is at `mult=16`, and BDH crosses below the Transformer at
`mult=8`.

**Real, disclosed limit**: cutting FLOPs 2x did NOT cut wall-clock 2x
(`mult=16` = 102s vs Transformer 37s, still 2.8x slower). Consistent
with Part 1 -- BDH's cost is real arithmetic running efficiently, so the
training-speed gap is narrowed, not closed.

## Part 3: inherited attention primitives (3 seeds, confirmed)

`reference/hz0h_bdh_primitive_ablations_torch.py` +
`scripts/hz0h_bdh_primitive_ablation_sweep.py`. Same harness/config as
Part 2 at `mult=16`. **The baseline arm is proven bit-for-bit identical
to `BDH.forward`** (`torch.equal`, not approximate) by
`tests/reference/test_hz0h_bdh_primitive_ablations_torch.py` -- that
equivalence is what makes every delta below meaningful.

Deltas vs the upstream baseline (negative = better), all 3 seeds:

| arm | s7 | s8 | s9 | mean | verdict |
|---|---:|---:|---:|---:|---|
| **`softmax_scaled`** | -0.0249 | -0.0467 | -0.0165 | **-0.0294** | **WIN, all 3 seeds** |
| `rope_theta_10000` | +0.0013 | -0.0194 | -0.0146 | -0.0109 | inconclusive |
| `scaled_scores_control` | +0.0003 | -0.0014 | +0.0068 | +0.0019 | ~zero (control OK) |
| `rope_theta_1048576` | +0.0153 | +0.0113 | +0.0243 | +0.0170 | worse, all 3 |
| `self_inclusive_mask` | +0.1032 | +0.0863 | +0.0896 | +0.0930 | worse, all 3 |
| `softmax_attention` | +0.1372 | +0.1049 | +0.1196 | +0.1205 | worse, all 3 |
| `standard_attention` | +0.2823 | +0.2611 | +0.2888 | +0.2774 | worse, all 3 |

Baseline absolute losses: 1.9193 / 1.9238 / 1.9071 (seeds 7/8/9).

### The one real win

**Softmax WITH `1/sqrt(d)` temperature beats upstream's raw
unnormalized attention by ~0.029 mean, negative in all 3 seeds.**
Critically, softmax *alone* is much WORSE (+0.1205) -- the temperature
is not a detail, it is the entire effect. Upstream was not wrong to
avoid naive softmax; it simply never paired it with proper scaling.

### Three inherited choices now vindicated by measurement

- **`tril(diagonal=-1)` (a token cannot attend to itself) is genuinely
  right** -- allowing self-attention costs +0.0930, consistent across
  all 3 seeds. This non-standard choice is load-bearing, not an
  oddity.
- **`theta=2**16` is fine** -- standard RoPE 10000 is within noise
  (mean -0.011, sign flips across seeds), and going higher (2^20) is
  reliably worse. Not special, not wrong.
- **Omitting `1/sqrt(d)` is correctly irrelevant WITHOUT softmax.**
  Real reason, verified: `yKV = ln(scores @ x)` applies LayerNorm
  immediately after attention, and `LayerNorm(c*v) == LayerNorm(v)` for
  `c > 0` except for the `eps` inside `sqrt(var + eps)`. The control arm
  measured +0.0019 (~zero) exactly as that math predicts -- which also
  validates that the harness measures what it claims.
- **"Make BDH's attention conventional" is the WORST arm** (+0.2774).
  BDH's attention design is not accidental.

### A real self-caught methodology error, recorded

The first version of the scaling test asserted only that outputs
DIFFER, which passed purely on the `eps` artifact and would have
implied a real effect that does not exist. It was rewritten to assert
the true property (near-invariance without softmax) plus a second test
showing scaling DOES matter under softmax. The `scaled_scores_control`
arm was then kept deliberately as a harness self-check rather than a
candidate.

## Real, disclosed limits on everything above

- Scaled-down local runs: smaller model, 5M tokens (not 25M), fp32 on
  MPS (not bf16), `n_layer=4` (not 8). **Absolute losses here are NOT
  comparable to the CUDA reference numbers** (dense BDH 1.3848, matched
  Transformer 1.5141) and must never be quoted alongside them.
- What transfers is DIRECTION and rough magnitude, which is what a
  3-seed consistent sign is evidence for.
- This project has a documented case of a short probe reversing at full
  scale (FactorizedBDH), which is why the depth curriculum was applied
  throughout here and why CUDA confirmation is still required before
  either finding is treated as settled.

## Concrete next steps

1. **Confirm `softmax_scaled` at full CUDA scale** (25M tokens, bf16,
   `n_layer=8`, `mult=32`). If it holds, it is a free quality gain over
   upstream BDH.
2. **Confirm `mult=16`** at full scale -- a 2x FLOP/parameter cut for
   ~0.02 loss would be the single largest efficiency result in the
   project so far.
3. **Test the two together** (`mult=16` + `softmax_scaled`) -- they are
   independent changes and may compose.
4. Dispatched but NOT yet returned (Windows box was down, then
   recovered): `hz0h_bdh_cost_breakdown_result.json` (per-stage CUDA
   timing vs FLOP share) and `hz0h_bdh_width_frontier_result.json`
   (full-scale width sweep).
5. Still-untested inherited choices from the same audit, in rough
   priority order: **weight tying across depth** (the defining BDH
   choice, never ablated), `V = x` narrow while `Q,K` are wide,
   **ReLU** as the sparsifier (it is what creates the ~5% sparsity),
   and the `ln(x + ln(yMLP))` double-LayerNorm with
   `elementwise_affine=False`.
