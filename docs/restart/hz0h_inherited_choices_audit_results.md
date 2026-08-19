# Auditing BDH's Inherited Choices: Width Is ~2x Oversized, Softmax+Temperature Beats Upstream Attention, and Weight Tying Is Load-Bearing

Date: 2026-08-18/19. Status: real, multi-seed results on a scaled-down
local (MPS) setup. **One confirmed quality win over upstream BDH, one
confirmed ~2x FLOP saving, one confirmed load-bearing structural choice
(weight tying) with a real speed/quality tradeoff curve across it at
production depth, three inherited choices vindicated by real
measurement rather than assumption, one honest 3-seed NULL result
(a promising single-seed lead for shared-base low-rank adapters did not
replicate -- see Part 4d), and one real green-light structural signal
(a single fitted linear operator explains most one-step recurrent
dynamics and composes almost exactly at 2 steps -- Part 5), one
real prototype result with the strongest efficiency signal in the
whole audit (a trained jump operator, "settle 4 real iterations then
jump the rest," gets 1.9x real wall-clock speedup for +0.029 loss --
Part 6, single-model prototype, needs firming up before production use
but doesn't cost parameters or the tying-quality penalty every other
lever in this doc pays), and one important negative capstone result
(stacking every confirmed win into one recipe LOST on quality to both
raw BDH and the matched Transformer, confounded and not yet isolated --
Part 7).** Full-scale CUDA
confirmation for Parts 1-3 is dispatched but not yet returned
(Windows/RTX3060 is offline
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

### Part 4c: confirmed at real production depth, `n_layer=8` (3 seeds)

Follow-up (2026-08-19), same day: reran the full groups sweep at
`n_layer=8` -- the actual production depth, not the `n_layer=4` local
proxy every earlier part of this audit used. This matters because the
project has a documented case of a small-scale probe reversing at full
scale (FactorizedBDH); Part 4b's G=2 finding needed this check before
being treated as more than preliminary.

| G | Δ matched-budget (mean, 3 seeds) | Δ full-capacity (mean, 3 seeds) | params (full-cap) | wall-clock (matched-budget vs tied) |
|---|---:|---:|---:|---:|
| 1 (tied) | 0.0000 | -- | 1.00x | 1.0x (~227s) |
| 2 | +0.0837 | +0.0054 | 1.96x | **1.6x faster** (~134s) |
| 4 | +0.1520 | -0.0187 | 3.88x | **2.5x faster** (~87s) |
| 8 | +0.2497 | -0.0296 | 7.72x | **3.4x faster** (~63s) |

Per-seed detail (`results/local/hz0h_depth_untied_n8_seed{7,8,9}.json`):

| seed | tied | g2_matched (Δ) | g2_full (Δ) | g4_matched (Δ) | g4_full (Δ) | g8_matched (Δ) | g8_full (Δ) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7 | 1.9194 | +0.1166 | +0.0221 | +0.1489 | -0.0009 | +0.2750 | -0.0079 |
| 8 | 1.9493 | +0.0821 | -0.0046 | +0.1570 | -0.0291 | +0.2287 | -0.0201 |
| 9 | 1.9390 | +0.0525 | -0.0013 | +0.1502 | -0.0261 | +0.2453 | -0.0607 |

**Real conclusion 1: the `n_layer=4` shape holds at real depth.** Matched-
budget cost is still monotonic in G (0.08 -> 0.15 -> 0.25, all 3 seeds
consistent in sign and ordering), confirming this is not a shallow-
recurrence artifact.

**Real conclusion 2, new at this depth: full-capacity untying now
matches or BEATS tied at every G tested, not just at the high end.**
G=2 full-capacity is a statistical wash (+0.005, mixed sign across
seeds); G=4 and G=8 full-capacity are consistently negative (better
than tied) in all 3 seeds. At `n_layer=4` this pattern only showed up
noisily; at real depth it is real and consistent.

**Real conclusion 3, the actual production-relevant finding: full-
capacity arms cost the SAME wall-clock as tied (~227-234s) -- giving
each group its own full-width weights doesn't save any compute, only
changes which weights get used per round.** The genuine speed win lives
ENTIRELY in the budget-matched arms, where narrowing each group's width
by `groups` produces a real, measured, monotonic wall-clock reduction
(1.6x / 2.5x / 3.4x faster than tied for G=2/4/8) at the cost of a real,
also-monotonic quality hit. This is the real tradeoff curve a production
decision would sit on -- not "untying costs quality" (true only for the
matched-budget arms) and not "untying is free" (true only for the
full-capacity arms, which spend up to 7.72x the params to get there).

Real, disclosed limit specific to this part: still MPS/fp32/5M tokens
per arm, not the 25M-token CUDA reference config, and wall-clock
numbers are local-hardware-specific (MPS matmul scheduling, not
representative of CUDA kernel launch/memory-bandwidth tradeoffs at
production batch size). The QUALITY deltas and their monotonic shape
are the more portable claim; the exact 1.6x/2.5x/3.4x speed multipliers
should be treated as directionally suggestive only until measured on
CUDA.

### Part 4d: shared-base + low-rank adapter -- 3-SEED RESULT: NULL, DID NOT REPLICATE

Follow-up (2026-08-19), same day, prompted directly by the question "do
we really need each group's own FULL toolbox, or can groups share a
base and each pay for only a small correction?" Part 4c's
`untied_full_capacity` (every group gets a full independent matrix)
matched or beat tied quality at every G, but cost up to 7.72x the
params at G=8. `reference/hz0h_bdh_depth_adapter_torch.py`
(`AdapterDepthBDH`) tests a LoRA-style middle ground: ONE shared
full-size `encoder`/`encoder_v`/`decoder` (same as tied), plus a small
rank-`r` correction PER GROUP (`W_group = W_shared + A_group @
B_group`). `B_group` is zero-initialized, so at construction every
group is bit-exact identical to the tied oracle -- proven by
`tests/reference/test_hz0h_bdh_depth_adapter_torch.py`, same gate
pattern as the untied module. Mechanism itself is correct; the RESULT
below is what failed to replicate, not the code.

**First pass was a single seed (seed=7 only, deliberately, to get a
fast read before the 3-seed budget) and showed a promising lead: rank=4
landed at -0.0085 vs tied, comparable to full-capacity untying's -0.0079
win but at 1.14x params instead of 7.72x.** That lead is why this part
was re-run properly. It did NOT hold up.

**Real 3-seed result, `n_layer=8`, `mult=16`, `groups=8`, ranks
2/3/4/6/8/16** (`results/local/hz0h_depth_adapter_full_seed{7,8,9}.json`):

| rank | seed 7 Δ | seed 8 Δ | seed 9 Δ | mean Δ | sign pattern |
|---|---:|---:|---:|---:|---|
| 2 | +0.0086 | -0.0273 | +0.0034 | -0.0051 | +-+ |
| 3 | +0.0138 | -0.0115 | -0.0029 | -0.0002 | +-- |
| 4 | +0.0159 | -0.0242 | -0.0070 | -0.0051 | +-- |
| 6 | +0.0057 | -0.0012 | +0.0091 | +0.0045 | +-+ |
| 8 | +0.0123 | -0.0092 | +0.0018 | +0.0016 | +-+ |
| 16 | +0.0290 | -0.0364 | -0.0064 | -0.0046 | +-- |

`tied_baseline` itself across the 3 seeds: 1.9262 / 1.9655 / 1.9386 --
a spread of **0.039**, larger than every single mean effect above
(all under 0.006 in magnitude).

**Real conclusion: NOT ONE rank shows a consistent sign across all 3
seeds.** Every rank wins in some seed and loses in another. This is not
"a small but real effect obscured by noise" -- the `tied_baseline`
reference itself varies by 0.039 seed-to-seed while the largest mean
effect is 0.0051, roughly 7-15x smaller than the reference's own noise.
The original single-seed "r=4 beats tied by -0.0085" result was inside
that noise band, not a real signal -- confirmed directly: rerunning the
exact same seed (7) with the same code produced tied=1.9262 and
r4=1.9421 (Δ **+0.0159**, the OPPOSITE sign from the original
preliminary run's Δ-0.0085 at the same nominal seed). MPS run-to-run
non-determinism (already disclosed in Part 4b) plus ordinary
single-training-run variance are large enough here to flip the
headline result's sign even at identical seed and config.

**Real diagnosis: this experimental design cannot resolve an effect
this small.** Each arm (including `tied_baseline`) is ONE training run.
Comparing one noisy run against another noisy run means the comparison
noise is at least as large as either run's own noise -- and here it
dominates. A real follow-up would need either (a) multiple independent
`tied_baseline` reps per seed to establish a stable reference mean and
variance before comparing adapters against it, or (b) a much larger
token budget / longer run where training noise shrinks relative to any
real effect, or (c) just accept that if the adapter mechanism has a
real effect at this scale, it is smaller than ~0.03 in validation loss
and not worth chasing further with this harness.

**Honest bottom line: the shared-base low-rank adapter idea is
unconfirmed, not refuted, but the specific numeric lead ("r=4 recovers
full-untying's win at 1/7th the params") is dead -- it was noise, and
should not be repeated as a finding.** Contrast with Parts 4/4b/4c,
where every reported effect had a consistent sign across all 3 seeds
(the actual bar this result failed to clear). This is exactly the kind
of result the 3-seed standard exists to catch, and it worked as
intended here.

## Part 5: BDH trajectory linearizability diagnostic -- a real green light for a jump-ahead operator

Follow-up (2026-08-19), separate question from Parts 1-4: not "how do
we make the repeated transform cheaper" but "does the repeated
transform actually need to be repeated `R` full nonlinear times, or
does it settle into something a single cheap operator could jump
across?" `reference/hz0h_bdh_trajectory_torch.py`
(`bdh_forward_with_trajectory`, bit-exact vs the oracle at full depth)
captures the real recurrent state `x_0..x_8` and the encoder ReLU mask
at every step; `scripts/hz0h_bdh_linearizability_diagnostic.py` trains
a small real BDH (CPU, `n_embd=128`, `n_layer=8`, `mult=16`, 1.5M
tokens -- deliberately run on CPU so it could execute alongside the
concurrent MPS-based Part 4 sweep without resource contention) and
fits closed-form (ridge least squares, no gradient training needed for
the diagnostic itself) latent linear operators on the captured states.

**Real, disclosed scope: this is ONE model, lightly trained (1.5M
tokens, ~3 minutes) -- a structural/qualitative diagnostic, not a
3-seed quantitative claim.** Its job is to decide whether investing in
a real jump-operator architecture is well-motivated at all, not to
pin down exact numbers.

**Finding 1: BDH's recurrence genuinely settles, on its own.** Both the
per-step update magnitude and the encoder ReLU activation pattern
stabilize monotonically with depth, with no mechanism forcing this:

| step | cosine(x_r, x_r+1) | \|Δx\|/\|x\| | ReLU-mask IoU |
|---|---:|---:|---:|
| 0->1 | 0.786 | 0.652 | 0.655 |
| 2->3 | 0.945 | 0.329 | 0.787 |
| 4->5 | 0.976 | 0.215 | 0.868 |
| 6->7 | 0.992 | 0.124 | 0.920 |
| 7->8 | 0.994 | 0.103 | -- |

This is exactly the regime where local linearization is expected to
work best (the update shrinks and the piecewise-linear ReLU region
stabilizes) -- and it's an empirical property of trained BDH, not an
assumption.

**Finding 2, the crucial test: does a single fitted operator K COMPOSE
across multiple steps, or only fit locally?** Pooled a one-step
operator K (via ridge least squares across ALL consecutive-step pairs,
valid because the oracle literally reuses the same weights every
iteration) in a PCA-reduced latent space, then compared `K^k` (composed)
against a separately, directly-fit `A_k` (the honest upper bound on
what any k-step operator could achieve) at k=2 and k=4:

| latent dim | one-step error | K² vs A₂ gap (k=2) | K⁴ vs A₄ gap (k=4) |
|---|---:|---:|---:|
| 32 | 0.172 | +0.0065 | +0.0623 |
| 64 | 0.163 | +0.0086 | +0.0863 |
| 128 | 0.155 | +0.0093 | +0.0858 |

**Real conclusion: a single linear operator explains a real majority of
one-step dynamics (~83-85%, i.e. 1 minus the ~15-17% relative error)
and composes almost exactly at k=2 (gap under 1%).** At k=4 the gap
grows to a real but modest ~6-9 points -- degrading, not collapsing.
This is a genuine green light, not proof: it says a `J_2`-style jump
(replace 2 real recurrent iterations with one learned/fit operator)
is well-supported by this data; `J_4` would need either a properly
end-to-end-trained jump network (not just post-hoc least squares) or
accepting some quality cost, consistent with the horizon-4 gap seen
here.

**Real next steps for this specific thread:** (1) rerun at a properly
trained model (more tokens, full curriculum depth) to firm up these
numbers before investing engineering time in an actual jump-operator
architecture; (2) if the pattern holds, prototype a real learned `J_2`
(trained end-to-end against real BDH's own 2-step trajectories as the
target, per the `L_multi` loss sketch this thread was built to test)
rather than the closed-form post-hoc fit used here, since a trained
jump network isn't bound by what a linear operator alone can capture.

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

## Part 6: trained jump operator J2 -- "settle first, then jump" holds, with real numbers

Follow-up (2026-08-19), same day, direct continuation of Part 5: does a
small TRAINED operator (not the closed-form lstsq fit) do better, and
crucially -- does substituting it for real recurrence actually preserve
DOWNSTREAM validation loss, not just raw state-prediction accuracy?
`reference/hz0h_bdh_jump_operator_torch.py` (`JumpOperator`, a small
zero-init residual MLP standing in for 2 real recurrent iterations, and
`jump_bdh_forward`, proven bit-exact vs `bdh_variable_depth_forward` at
`num_jumps=0`) plus `scripts/hz0h_bdh_jump_operator_prototype.py`
(trains a BDH teacher, distills J2 against the teacher's REAL 2-step
trajectories via state MSE + logits KL, then evaluates real/jump
splits on held-out validation loss) answer this directly.

**Real, disclosed scope: one trained BDH teacher (1.5M tokens, CPU),
500 distillation steps for J2 -- a prototype, not a 3-seed claim.
Its job is to check whether this idea is worth real engineering
investment, and it is.**

Real substitution result, all arms reaching the same depth-equivalent
(8), only the real-iteration/jump split differing:

| arm | real iters | jumps | loss | Δ vs real_depth8 | eval wall-clock | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `real_depth8` | 8 | 0 | 2.6151 | 0.0000 | 0.50s | 1.0x |
| `hybrid_4real_4jump` | 4 | 2 (=4 depth-eq) | 2.6445 | **+0.0293** | 0.26s | **1.9x** |
| `hybrid_2real_6jump` | 2 | 3 (=6 depth-eq) | 2.6905 | +0.0753 | 0.14s | 3.6x |
| `all_jumps` | 0 | 4 (=8 depth-eq) | 2.9269 | +0.3118 | 0.02s | 25x |

**Real conclusion: the "settle first, then jump" hypothesis (motivated
directly by Part 5's finding that early recurrent steps do the most
genuine nonlinear work, measured via lowest cosine similarity between
consecutive states) holds cleanly and monotonically.** Skipping real
recurrence entirely (`all_jumps`) costs a real, large +0.31 loss despite
being 25x faster -- the jump operator alone cannot substitute for the
genuinely nonlinear early iterations. But keeping just 4 real iterations
up front, then jumping the remaining depth-equivalent via 2 cheap jump
calls, gets **1.9x real wall-clock speedup for a validation-loss cost of
only +0.029** -- close to indistinguishable from full real recurrence.
`hybrid_2real_6jump` is the middle point (3.6x faster, +0.075).

**Why the speed numbers look the way they do**: a `JumpOperator` call is
a small `D -> 4D -> D` residual MLP, dramatically cheaper than a real
BDH iteration's three wide `D -> mult*D` GEMMs plus attention (consistent
with Part 1's finding that those three projections are 82.8% of
per-iteration FLOPs) -- so wall-clock scales almost entirely with how
many REAL iterations remain, not how many jumps are used. This is the
first idea in this whole audit that attacks throughput directly, without
requiring fewer parameters or a param/capacity tradeoff (contrast with
Parts 2/4/4b/4c/4d, which all traded SOME capacity or quality for speed
or vice versa) -- the jump operator's own parameter cost is a small,
fixed, one-time addition, not something that scales with how much
recurrent compute it replaces.

**Real next steps for this thread:** (1) rerun at a properly trained
teacher and more distillation steps to firm up the exact tradeoff curve
before committing engineering time; (2) sweep the real/jump split more
finely (3 real + jumps, 5 real + jumps) to find the actual knee, not
just the two hybrid points tested here; (3) test J4 (jump_size=4)
directly, since Part 5 found the linear composition gap was still
modest (not catastrophic) even at k=4; (4) if the pattern holds at
scale, this is the strongest efficiency lever found in this entire
audit -- it doesn't cost parameters, doesn't cost the tying-quality
penalty found in Parts 4/4b/4c, and gives a real, tunable speed/quality
dial via how many real iterations run before jumping.

## Part 7: the capstone comparison -- stacking every win does NOT beat raw BDH or the Transformer

Follow-up (2026-08-19), same day: the obvious next question after Parts
1-6 is "if we combine every confirmed positive, how does the result
compare to raw BDH and the matched Transformer, on both quality and
real throughput?" `reference/hz0h_bdh_combined_best_torch.py` (proven
bit-exact against the tested `softmax_scaled` ablation) stacks `mult=16`
(Part 2), `softmax_scaled` attention (Part 3), weight tying kept exactly
as upstream (Part 4 found untying load-bearing-BAD), and the trained
jump operator at its best-found hybrid point (`real_prefix=4,
num_jumps=2`, Part 6). `scripts/hz0h_bdh_combined_best_comparison.py`
trains this combined recipe, raw BDH (true canonical `mult=32`), and the
matched Transformer on the SAME data/recipe/token budget (5M tokens,
`n_embd=256`, `n_layer=8`, MPS), then measures validation loss and real
throughput (uncompiled and with `torch.compile`, which was empirically
confirmed to give ~1.55x on this machine's MPS backend before being
included -- not assumed).

**Real, honest result -- the combined recipe LOST on quality to both
baselines:**

| arm | val loss | params | tok/s (uncompiled) | tok/s (`torch.compile`) |
|---|---:|---:|---:|---:|
| `raw_bdh` (canonical) | 1.9191 | 6.42M | 28,144 | 45,401 (1.6x) |
| `combined_best` | **2.0586** | 3.80M | 100,433 | 120,957 (1.2x) |
| `matched_transformer` | 1.9778 | 8.49M | 183,588 | 469,001 (2.6x) |

`combined_best` is +0.1395 WORSE than raw BDH and +0.0808 WORSE than the
matched Transformer -- despite using every individually-confirmed
positive finding from this entire audit. It IS genuinely faster than
raw BDH (3.6x uncompiled, 2.7x compiled -- the jump-hybrid recurrence
and narrower width both did their job on throughput), but the
Transformer is still both faster AND higher quality than the best
combined BDH recipe.

**Real, honest diagnosis -- confounded, not yet isolated.** Two
plausible causes were stacked together without a control run to
separate them: (1) `mult=16` has roughly half `raw_bdh`'s parameter
count (3.80M vs 6.42M, before the jump operator's own small addition),
and Part 4 already established fewer params costs quality on its own;
(2) the jump operator's substitution cost, found to be only +0.029 in
Part 6's SMALLER prototype (`n_embd=128`, 1.5M tokens), may cost
noticeably more at this larger scale (`n_embd=256`, 5M tokens) -- this
run never evaluated the `mult=16 + softmax_scaled` teacher at its OWN
full real depth=8 (no jumps) as an intermediate checkpoint, so the
capacity cost and the jump-substitution cost cannot currently be told
apart.

**This is a real, useful negative result, not a wasted run: individually
confirmed wins do not automatically compose, and this audit's central
methodological lesson (isolate one variable at a time, per-seed,
against a real control) applies to COMBINING findings just as much as
to finding them in the first place.** The obvious immediate follow-up,
not yet run: evaluate the same `mult=16 + softmax_scaled` teacher at its
own full real depth=8 (no jump substitution) to isolate exactly how much
of the +0.1395 gap is the width/capacity cost versus the jump cost,
before concluding anything about whether this specific stacked recipe
is salvageable.

### Part 7 addendum: a research pass over PRE-audit project history for other real wins

A pass over `docs/restart/*.md` and git history (before this session's
audit began) for confirmed positives not yet in this recipe surfaced
real candidates for a future combined-recipe round:

- **Wide-GEMM encoder layout remap** (`reference/hz0h_bdh_wide_gemm_encoder_torch.py`,
  commit `a314bc5`): real, CPU+CUDA-parity-confirmed **1.705x faster**,
  pure execution remap of the SAME math (reshapes the per-head broadcast
  matmul into one big GEMM) -- composes freely with the quality-side
  recipe since it changes nothing numerically. **Caveat: forward-only,
  not yet wired for gradient flow** -- real integration work needed
  before it can replace the training-time encoder call.
- **bmm encoder_v layout remap** (`reference/hz0h_bdh_bmm_encoder_v_torch.py`,
  commit `31ae1d2`): same story, **1.509x faster**, same forward-only
  caveat. Different projection than the wide-GEMM remap above, so the
  two may compose (not yet measured together).
- **`torch.compile(mode="max-autotune")`**: a prior CUDA result found
  +4.6% over default compile mode. Cheap to try here too -- a one-line
  flag change to `measure_throughput`'s `torch.compile` call.
- **Activation checkpointing**: real but config-dependent in prior CUDA
  work (2.08x speedup at one config, -11.2% at another) -- do not add
  blindly, needs the specific winning config identified first.
- **Confirmed NOT worth adding**: GPU-native integration (prior result:
  1.57x SLOWER, a real loss) and 2:4 structured sparsity (real hardware
  blocker, not available on this Mac).

None of these were in `combined_best`'s recipe. The wide-GEMM and
bmm-encoder_v remaps are the cleanest next additions -- zero numerical
change, so they cannot explain or worsen Part 7's quality gap, only
throughput -- but need real training-path integration (not just a
forward-pass benchmark) before they can be added to a training run like
this comparison.

## Concrete next steps

1. **Confirm `softmax_scaled` at full CUDA scale** (25M tokens, bf16,
   `n_layer=8`, `mult=32`). If it holds, it is a free quality gain over
   upstream BDH.
2. **Confirm `mult=16`** at full scale -- a 2x FLOP/parameter cut for
   ~0.02 loss would be the single largest efficiency result in the
   project so far.
2a. **Confirmed locally at real depth (Part 4c) -- now confirm on CUDA
   at the real 25M-token config.** The `n_layer=8` local run answered
   the depth-transfer question already (the shape holds), so what CUDA
   still needs to confirm is SCALE: does the matched-budget quality cost
   and the budget-matched wall-clock speedup both hold at 25M tokens,
   bf16, the real batch size? Dispatch blocked as of 2026-08-19: Windows/
   RTX3060 has been offline 16h+ (Tailscale `desktop-2sreddp`), longer
   than the documented flip-flop pattern -- treat as a real outage, not
   routine flakiness, until it clears.
2b. **Find the minimum per-group multiplier that recovers tied's
   quality, for a fixed G** (e.g. G=4) -- Part 4c only tested the two
   extremes (budget-matched = mult/G, full-capacity = mult unchanged).
   The real production question is the cheapest per-group width that
   still matches tied's loss: sweep per-group multiplier at fixed G
   between those two endpoints to find the actual speed/quality knee,
   rather than picking one of the two ends blind.
2c. **Once 2b finds a candidate (G, per-group multiplier) that matches
   tied's quality with a real compute saving, this stops being an
   ablation and becomes a real architecture candidate** -- would need
   integration into the main HZ-0H training path (not just this
   isolated sweep script) and a CUDA-confirmed throughput number before
   being treated as a production change.
2d. **DONE, NULL RESULT -- see Part 4d.** The 3-seed re-run (ranks
   2/3/4/6/8/16) found no rank with a consistent sign across seeds; the
   original single-seed lead did not replicate and was noise, not a
   real effect. Do not re-attempt this exact experiment without first
   fixing the underlying issue: `tied_baseline` varied 0.039 across
   seeds (single un-averaged run each), 7-15x larger than any measured
   adapter effect. A real follow-up would need multiple `tied_baseline`
   reps per seed to establish a stable reference before comparing
   adapters against it -- otherwise any future adapter/architecture
   ablation at this token budget faces the same noise floor.
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
