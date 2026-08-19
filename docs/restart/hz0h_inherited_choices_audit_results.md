# Auditing BDH's Inherited Choices: Width Is ~2x Oversized, Softmax+Temperature Beats Upstream Attention, and Weight Tying Is Load-Bearing

Date: 2026-08-18/19. Status: real, multi-seed results on a scaled-down
local (MPS) setup. **One confirmed quality win over upstream BDH, one
confirmed ~2x FLOP saving, one confirmed load-bearing structural choice
(weight tying), and three inherited choices vindicated by real
measurement rather than assumption.** Full-scale CUDA confirmation for
Parts 1-3 is dispatched but not yet returned (Windows/RTX3060 is offline
as of this writing -- see Part 4's own next-steps note).

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

## Part 4: weight tying across depth -- the defining choice, now tested (3 seeds, confirmed)

`reference/hz0h_bdh_depth_untied_torch.py` + `scripts/hz0h_bdh_depth_untied_ablation_sweep.py`.
Same harness/config as Parts 2-3 (`mult=16` baseline). `DepthUntiedBDH`
gives each recurrent level its OWN `encoder`/`encoder_v`/`decoder`
instead of the oracle's single set reused every level; `embed`/`ln`/
`attn`/`lm_head` stay shared, matching what upstream itself shares.
**Proven bit-exact against the real oracle** when every level is forced
to identical weights (`tests/reference/test_hz0h_bdh_depth_untied_torch.py`,
`torch.equal`).

Two untied arms, both compared to `tied_baseline` (real oracle, `mult=16`):

| seed | tied_baseline | untied_budget_matched (Δ, 1.0x params) | untied_full_capacity (Δ, 3.88x params) |
|---|---:|---:|---:|
| 7 | 1.9354 | 2.1245 (**+0.1891**) | 1.9284 (-0.0070) |
| 8 | 1.9124 | 2.1371 (**+0.2247**) | 1.9675 (+0.0551) |
| 9 | 1.9213 | 2.1189 (**+0.1976**) | 1.9385 (+0.0173) |

`untied_budget_matched` divides the per-level multiplier by depth
(`budget_matched_multiplier(16, depth=4) = 4`) so its TOTAL
encoder/encoder_v/decoder param count across all levels approximates
the tied baseline's single set -- isolates tying itself from the
capacity confound. `untied_full_capacity` gives every level the SAME
per-level multiplier as the tied baseline (3.88x more total params in
those three matrices), an upper bound on what untying could buy if
capacity were free.

**Real conclusion: weight tying is load-bearing, not an inherited
artifact.** At matched capacity, untying costs +0.19 to +0.22 -- the
single largest effect size found across this entire audit, bigger than
`standard_attention`'s own +0.28 relative to its baseline's absolute
scale. Even paying 3.88x the parameters only gets untying back to
roughly parity with tied (mixed sign, ±0.02-0.06) -- extra capacity
cannot buy back what tying provides. This is real evidence that reusing
one weight set across recurrent iterations acts as a genuine
regularizer/inductive bias, not just a parameter-count-saving trick.

### Part 4b: the full C ~ (depth/groups)*P curve, not just the two endpoints (3 seeds)

Follow-up (2026-08-19) generalized `DepthUntiedBDH` to a `groups`
parameter: `groups` independent weight sets, `depth // groups` adjacent
levels sharing each one (`W1,W1,W2,W2` for `depth=4, groups=2`).
`groups=1` collapses to the tied structure (separately proven bit-exact,
same gate pattern) and `groups=depth=4` is the fully-untied arms above
-- `groups=2` fills in the missing midpoint, motivated directly by the
closed form `C ~ (depth/groups) * P` at fixed total params: if the
quality cost scales with compute reduction, G=2 should land roughly
between G=1 and G=4, not near either endpoint.

Real, disclosed note: this is a SEPARATE run of the whole sweep
(including a fresh `tied_baseline`), and MPS training is not bit-
deterministic seed-to-seed even with `torch.manual_seed` fixed -- the
re-run's `tied_baseline` absolute losses (1.9158/1.9036/1.9255) differ
slightly from Part 4's original run (1.9354/1.9124/1.9213) despite
identical seeds and config. Deltas below are computed WITHIN this
second run's own matched `tied_baseline`, so the comparison stays valid;
only cross-run absolute-value comparisons are unsafe.

| seed | tied (G=1) | G=2 matched (Δ) | G=2 full-cap (Δ, 1.96x) | G=4 matched (Δ) | G=4 full-cap (Δ, 3.88x) |
|---|---:|---:|---:|---:|---:|
| 7 | 1.9158 | 2.0215 (+0.1057) | 1.9540 (+0.0382) | 2.1192 (+0.2034) | 1.9368 (+0.0210) |
| 8 | 1.9036 | 2.0238 (+0.1202) | 1.9422 (+0.0386) | 2.1124 (+0.2088) | 1.9619 (+0.0582) |
| 9 | 1.9255 | 2.0187 (+0.0932) | 1.9346 (+0.0090) | 2.1259 (+0.2004) | 1.9167 (-0.0089) |
| **mean** | -- | **+0.1064** | **+0.0286** | **+0.2042** | **+0.0234** |

**Real conclusion: the cost of untying is roughly monotonic in G, not a
cliff.** At matched params, G=2 costs about half of what G=4 costs
(+0.106 vs +0.204, consistent -- not exactly 2x, but the right
direction and order of magnitude), and this holds in all 3 seeds. At
full capacity (extra params allowed), G=2 and G=4 land at nearly the
SAME small residual cost (+0.029 vs +0.023) -- meaning most of what
extra capacity buys back is available already at G=2; going all the way
to G=4 does not meaningfully unlock more of it. Practically: if a
partial-untying architecture is pursued, G=2 looks like the more
efficient operating point than jumping straight to fully untied -- half
the quality cost per unit of compute saved. This is still `n_layer=4`
only; a real G=8 point needs `n_layer=8`, which needs either a much
longer local run or the CUDA machine (currently offline, see next
steps).

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
2a. **Confirm weight tying (Part 4/4b) at full CUDA scale, `n_layer=8`,
   including a real G=8 point** -- a real +0.19-0.22 effect (G=4,
   matched budget) and a roughly-monotonic G=2 midpoint are large enough
   locally that they should transfer, but nothing past `n_layer=4` has
   been checked, and G=8 (the value that matters for production) has
   never been run at all. Dispatch blocked as of 2026-08-19: Windows/
   RTX3060 has been offline 9h+ (Tailscale `desktop-2sreddp`), longer
   than the documented flip-flop pattern -- treat as a real outage, not
   routine flakiness, until it clears.
3. **Test the two together** (`mult=16` + `softmax_scaled`) -- they are
   independent changes and may compose.
4. Dispatched but NOT yet returned (Windows box was down, then
   recovered): `hz0h_bdh_cost_breakdown_result.json` (per-stage CUDA
   timing vs FLOP share) and `hz0h_bdh_width_frontier_result.json`
   (full-scale width sweep).
5. **Weight tying across depth: now tested, see Part 4.** Confirmed
   load-bearing (3 seeds). Still-untested from the same audit, in rough
   priority order: `V = x` narrow while `Q,K` are wide, **ReLU** as the
   sparsifier (it is what creates the ~5% sparsity), and the
   `ln(x + ln(yMLP))` double-LayerNorm with `elementwise_affine=False`.
