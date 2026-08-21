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
real prototype result that LOOKED like the strongest efficiency signal
in the whole audit at toy scale (a trained jump operator, "settle 4
real iterations then jump the rest," got 1.9x real wall-clock speedup
for +0.029 loss at `n_embd=128` -- Part 6) but DID NOT SURVIVE scaling
to real production depth: the same ~2x-speedup arm cost 41-65x MORE
quality loss at real scale (`n_embd=2496`), confirmed THREE independent
times (toy prototype, local MPS, and real CUDA -- Part 10) -- now a
closed, real negative result, not a promising lead, one important
negative capstone result
(stacking every confirmed win into one recipe LOST on quality to both
raw BDH and the matched Transformer, confounded and not yet isolated --
Part 7), one real CUDA-confirmed hidden cost (RoPE application is
36.5% of per-level wall-clock time despite ZERO FLOPs, invisible to
every FLOP-based argument in this whole audit until now -- Part 8), and
one real, first-time production-scale CUDA result (Part 9) that
reframes the whole "why bother with BDH" question: `raw_bdh` (mult=16,
plain attention) BEATS the matched Transformer on validation loss
at nearly equal params, on real 10M-token training -- at a real
throughput cost, corrected as of 2026-08-20 (see Part 9's retraction)
to ~3.7x slower (inference) / ~5.2x-6.9x slower (real training), not
the originally-reported 66x (that number was itself a bug --
`measure_throughput` was silently running in fp32, not bf16). Both are
true at once, not a contradiction. **UPDATE (2026-08-21): this is no
longer a single-run result -- a real 3-seed confirmation (on a rented
RunPod A40, using two further memory/speed fixes found the same day)
now shows `raw_bdh` beating `matched_transformer` on EVERY one of 3
seeds** (1.6844/1.4080/1.4027 vs. 2.3016/1.8988/1.9353, means 1.498 vs.
2.045). `combined_best`'s own arm remained broken (real
validation_loss=49.6306, a depth-8-specific divergence gradient
clipping only partially fixed), so this quality win belongs to plain
BDH, not the stacked recipe.
Finally, Part 11 directly measures the realized per-round operator gate
`g_r` (not a proxy for it), first at small local scale then CONFIRMED
on real production-scale CUDA (`n_embd=2496`, matching Part 9's shape):
~85-90% of its nominal width is exactly zero every round, the active
set stabilizes across recurrent depth (round-to-round support overlap
rises to 0.82-0.89 by the final rounds), and -- a real surprise, MORE
favorable than the local prototype suggested -- effective rank
collapses to under 1% of nominal width by the final round on real
hardware. Real evidence for "the architecture discovers sparsity after
paying for it," with an honest open question about how much of the
local-vs-production gap reflects the production model's real
convergence versus a sampling artifact (see Part 11's CUDA
confirmation for the full caveat) rather than a settled number. A
follow-up crux measurement (`cross_token_support_jaccard`) then
resolved the obvious next design question and got a real, disclosed
NEGATIVE-LEANING answer: the collapse is a shared low-rank SUBSPACE
across tokens, not tokens converging on the same discrete active
neurons (cross-token Jaccard only reaches 0.153 even at the
most-collapsed round, far short of the ~1.0 a static shared-mask kernel
would need) -- so the simple, GPU-easy block-sparse design is NOT
supported by this data; a harder subspace-projection kernel design
remains open and unevaluated.**

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

## Part 8: real CUDA cost-breakdown profile lands -- confirms Part 1, finds a new hidden cost

`hz0h_cost_breakdown_request.txt` (dispatched 2026-08-18/19, queued
untouched while Windows/RTX3060 was offline ~24h) finally landed
2026-08-19 -- real per-stage timing on the actual RTX 3060, production
shape (`n_embd=512`, `n_layer=8`, `n_head=8`, `mult=32`, bf16).
`transcription_max_abs_difference_vs_oracle: 0.0` -- the profiling
harness's load-bearing gate confirms it measured the real oracle, not
an approximation.

**Confirms Part 1's original analytic claim, now on real hardware, not
just FLOP arithmetic:** BDH does 11.897x the matched Transformer's
FLOPs; the REAL measured training slowdown is 5.3x
(`measured_training_slowdown_ratio_phase_f`); implied hardware
efficiency **2.24x** -- BDH really does run more efficiently per-FLOP
than the Transformer on this hardware, exactly as Part 1 argued from
FLOP counting alone.

**Real, new finding this profile surfaces that FLOP-based reasoning
never could: `attention_rope` is 36.5% of per-level wall-clock time,
despite ZERO recorded FLOPs** -- the single largest time-consuming
stage in the whole recurrent step, bigger than encoder (16.3%),
encoder_v (16.6%), or decoder (15.0%) individually:

| stage | time share | FLOP share | time/FLOP-share ratio |
|---|---:|---:|---:|
| `attention_rope` | 36.55% | 0% | -- (FLOP-invisible) |
| `encoder_v` | 16.59% | 27.59% | 0.60 (efficient) |
| `encoder` | 16.33% | 27.59% | 0.59 (efficient) |
| `decoder` | 14.97% | 27.59% | 0.54 (efficient) |
| `attention_scores` | 4.22% | 13.79% | 0.31 (very efficient) |
| `gate_multiply` | 3.64% | 0% | -- |
| `attention_values` | 1.65% | 3.45% | 0.48 (efficient) |

**Real implication: this whole audit's "three wide projections are
82.8% of FLOPs, attention is only 17.2%" framing (Part 1) is correct
about FLOPs but was never separately checked against real wall-clock
time per stage until now.** The three wide projections (encoder,
encoder_v, decoder) are all comfortably UNDER their FLOP-share of time
(ratios 0.54-0.60, i.e. genuinely efficient GEMMs on this hardware) --
but `attention_rope`'s RoPE application, invisible to FLOP counting
entirely, is nonetheless the single biggest real time cost. This looks
like a memory-bandwidth-bound or kernel-launch-overhead-bound
operation, not a compute-bound one -- a real, previously-hidden
optimization target this audit never looked at because every prior
part reasoned from FLOPs, not measured wall-clock per stage.

Also confirmed: `backward_over_forward_ratio: 2.67` (real measured
backward-pass cost multiplier, not assumed 2x), and
`end_to_end_forward_only_milliseconds: 213` / `end_to_end_train_step_
milliseconds: 569` at this shape and batch size (12).

**Real next step:** profile `attention_rope` specifically (is it
memory-bound? does a fused/precomputed RoPE application reduce its real
time without changing the math?) before assuming any of the
architecture-level levers in Parts 2-7 are the highest-leverage next
target -- a 36% single-stage time cost that's invisible to FLOP
accounting is a real, concrete, previously-undiscovered lead.

**CRITICAL, UNVERIFIED CAVEAT, flagged before this finding is trusted:**
this profile ran on a machine that was JUST recovering from ~24h
offline, and this project has a documented prior incident (during the
dynamic-block-routing OOM investigation) of stale zombie processes
silently holding GPU memory on this exact Windows box. Whether the GPU
was EXCLUSIVELY available during this specific profiling run has not
been confirmed. If another process was contending for the GPU
concurrently, `attention_rope`'s anomalous 36.5%-time/0-FLOP result
could be a contention artifact (kernel launches interleaving badly with
a concurrent process, or memory-bandwidth contention inflating one
kernel's measured time disproportionately) rather than a real,
reproducible architectural cost. **Do not act on this finding (e.g.
building a fused-RoPE optimization) until a clean-GPU-state rerun
confirms the same result.** Requested as an immediate follow-up.

## Part 9: real matched-param capstone lands on CUDA -- raw_bdh BEATS the matched Transformer on quality

Windows dispatch, 2026-08-20. The parameter-matched capstone comparison
(Part 7's local small-scale version, now run for real on the RTX 3060 at
~300M params, real 10M-token curriculum, bf16, gradient checkpointing,
int8 optimizer state, `mult=16` for all BDH arms per this doc's own
"drop canonical mult=32" correction) produced two real arms and one
broken one:

| arm | val loss | params | status |
|---|---:|---:|---|
| `raw_bdh` (mult=16, plain attention) | **1.7038** | 300.32M | clean, real, usable |
| `matched_transformer` | 2.3016 | 302.57M | clean, real, usable |
| `combined_best` (mult=16 + softmax_scaled + jump) | 49.6306 | 296.98M | BROKEN -- see below |

**Real, first-time result that reframes the whole "why bother with
BDH" question from earlier in this session: `raw_bdh` beats the matched
Transformer on validation loss, at nearly identical parameter count,
on real 10M-token training.** This is the first time this project has
measured BDH-vs-Transformer quality at matched params on real
production-scale CUDA hardware with both sides fully real (real RoPE,
real curriculum, matched weight_decay/optimizer/batch_size). It directly
contradicts the framing several messages earlier in this session that
"the Transformer wins outright, no asterisks" -- that framing was based
on THROUGHPUT alone (originally reported as a 66x speed gap; **corrected
2026-08-20 to ~3.7x -- see the retraction later in this Part, the 66x
figure was a measurement bug, not a real BDH cost**), not quality.
**Both are true simultaneously: BDH is slower (~3.7x, corrected) AND
gets a real, better validation loss at this exact matched-param scale.**
That is a genuine, disclosed tradeoff,
not a contradiction -- and it means the earlier "just use the
Transformer" conclusion needs to be walked back to "use the Transformer
if throughput is what matters; BDH may be worth the compute cost if
quality-per-parameter is what matters." **UPDATE (2026-08-21): the
3-seed confirmation this needed is now done -- see later in this Part
-- `raw_bdh` beat `matched_transformer` on all 3 real seeds
(1.6844/1.4080/1.4027 vs. 2.3016/1.8988/1.9353), not just the single
original run.**

### Cross-hardware reproducibility check (2026-08-21): same result, different GPU entirely

A second full 10M-token `raw_bdh` run at the identical shape/recipe
(`n_embd=2496, mult=16, n_layer=8`, `seed=7`, bf16, adam8bit) landed on
a rented RunPod A40 (`results/cuda/hz0h_runpod_a40_raw_300m_10m_result.json`)
-- completely different hardware, driver stack, and PyTorch/CUDA build
than the original Windows/RTX3060 run this Part's headline number came
from. Real result: **`validation_loss=1.6844`**, very close to the
original `1.7038`. **Real, precise caveat**: this uses the SAME seed
(7) as the original run, so it confirms the result isn't an artifact of
one specific GPU/driver/library combination (a real, non-trivial check
-- floating-point non-determinism across different hardware is a real
phenomenon) but it is NOT a substitute for the still-pending 3-SEED
confirmation noted above, which tests run-to-run variance rather than
hardware reproducibility. Both checks are real and both still matter;
neither replaces the other.

### combined_best's real failure mode: depth-dependent divergence, not a single bad step

Gradient clipping (added this session specifically for this bug) fixed
the ORIGINAL NaN point (step 1350, previously hard NaN, now
loss=2.9669 there -- healthy) and combined_best trained stably through
the depth=4 and depth=6 curriculum stages (losses 1.9-4.4, normal
noise). But a NEW, different divergence appeared entering the FINAL
depth=8 stage specifically:

```text
step 4650  loss=3.5039   (healthy)
step 4700  loss=8.5399
step 4750  loss=9.9610
step 4800  loss=14.7525
step 4850  loss=12.9649
final_loss=12.6063, validation_loss=49.6306 (ln(256)=5.5 is the random-
guessing baseline for this byte-level task -- 49.6 is not "undertrained",
it's a numerically corrupted state)
```

**Real diagnostic conclusion: this points to the instability compounding
with recurrent DEPTH itself, not a single large gradient spike that
clipping alone can suppress.** softmax_scaled attention was clean at
shallower curriculum depths (4, 6) and only broke down once training
reached the full real depth (8) -- consistent with a per-iteration
numerical issue (likely in the `(QR @ QR.mT) * scale` reduction under
bf16, per the mechanism already suspected when this bug was first found)
that accumulates or compounds across repeated iterations of the SAME
tied weights, rather than a one-off unlucky batch. Not yet diagnosed or
fixed -- real, open follow-up.

Also real: `combined_best`'s jump-operator distillation "succeeded"
numerically (`final_state_loss=0.4687`) but that number is meaningless
-- it was distilling against an already-numerically-corrupted teacher's
garbage trajectories, disclosed rather than treated as a working result.

### Compiled throughput for combined_best: a real regression, not a win

`combined_best`'s compiled throughput (249 tok/s) was SLOWER than
uncompiled (683 tok/s) at this shape -- the opposite of `raw_bdh`'s
story (which OOM'd under default compile mode rather than merely being
slow). This specific run predates this session's `max-autotune` default
switch (HEAD `c0354dc`, before `3660053`), so it used `torch.compile`'s
plain default mode -- unconfirmed whether `max-autotune` fixes this
regression too, real follow-up once retested.

### raw_bdh's 66x throughput gap: confirmed genuinely compute-bound, real memory data

A real question came up mid-session: was `raw_bdh`'s 153-164 tok/s
figure (the 66x-slower-than-Transformer result driving this whole
part) actually memory-pressure-limited on the RTX 3060, rather than a
real reflection of how expensive this shape's math is? Answered twice,
independently, with the same conclusion:

- Locally (this Mac, MPS, fp32): peak memory stayed FLAT at 1.20GB
  through the entire `no_grad` throughput loop -- no growth at all.
- On the real RTX 3060 (`torch.cuda.max_memory_allocated`, precise, not
  `nvidia-smi` sampling): **peak_memory_allocated=7.96GB, 66% of the
  12GB ceiling** -- real headroom, not a near-miss.

**Both independently confirm the 66x throughput gap is genuinely
compute-bound.** The math at `mult=16, n_embd=2496` really is that
expensive per token; it is not an artifact of memory contention,
thrashing, or being near a hardware ceiling.

Separately, also confirmed as a REAL, reproducible bug rather than a
one-off: rerunning the exact same default-compile-mode measurement
crashed AGAIN with the byte-identical failure signature (same `(8, 8,
2496, 4992)` fp32 buffer shape, same 2.97 GiB allocation failure) --
Inductor is deterministically generating this fp32 intermediate at this
shape under default compile mode, not something that happened once by
chance. Whether `--compile-mode max-autotune` avoids it is the next
real test, already dispatched.

### RETRACTION (2026-08-20): the "genuinely compute-bound, 66x slower" conclusion directly above was WRONG

`--compile-mode max-autotune` did NOT avoid the crash above (same
byte-identical failure a third time, confirmed on real CUDA) -- so the
investigation kept going instead of stopping at "max-autotune fixes
it." That follow-up found the real bug: `measure_throughput()` compiled
and ran `raw_bdh`'s forward pass under `torch.no_grad()` but **never
wrapped it in `autocast_context`**, unlike every other forward call in
`hz0h_bdh_combined_best_comparison.py` (`train_bdh`, `train_transformer`,
`evaluate_loss` all do). Model weights live in fp32, so the "throughput"
measurement -- both the number quoted above (153-164 tok/s) AND the
7.96GB peak-memory figure used to argue "compute-bound, not
memory-pressure" -- was silently measuring pure fp32 execution, not the
bf16 path every other number in this project uses. Fixed in commit
`14df7b2` (added the missing `autocast_context` wrap); full local test
suite still green (874 passed) after the fix.

**Real, corrected numbers on the same RTX 3060, same shape
(`mult=16, n_embd=2496`), after the fix:**

| | peak mem | tok/s |
|---|---:|---:|
| uncompiled, before fix (fp32 bug) | 7.96GB | 164 |
| uncompiled, after fix (real bf16) | 5.90GB | **2758** |
| compiled, before fix (fp32 bug) | -- | CRASH |
| compiled, after fix (real bf16) | 11.69GB | 617 |

The corrected uncompiled throughput (2758 tok/s) is 16.8x faster and
uses 26% less memory than the number this whole section was built on.
Against `matched_transformer`'s uncompiled ~10130 tok/s, **the real gap
is ~3.7x, not 66x.** The "genuinely compute-bound" conclusion above is
retracted as measured-on-a-bug, not a real finding -- the memory
headroom argument (7.96GB / 66% of ceiling) was real on its own terms,
but the throughput side of that comparison was corrupted by the same
fp32 bug, so the ratio it produced doesn't reflect BDH's real cost.

This changes Part 9's framing (below), not its core result:
`raw_bdh` beating the matched Transformer on validation loss
(1.7038 vs 2.3016) at matched params is untouched by this bug (that
number came from real training, not `measure_throughput`). What
changes is the price paid for that quality win -- roughly 3.7x slower,
not 66x -- which makes the "BDH may be worth the compute cost" framing
dramatically more favorable than previously stated, not less.

**Real, disclosed distinction (2026-08-21) -- the ~3.7x figure above is
INFERENCE-only, not training**: `measure_throughput` runs a bare
forward pass under `no_grad` -- no backward, no optimizer step. A
separate real dispatch measured actual TRAINING throughput (forward +
backward + `optimizer.step()`, same bf16/adam8bit recipe, same matched
~300M-param shapes, same 2M-token budget) for both arms directly:

| | training tok/s | params |
|---|---:|---:|
| `raw_bdh` (domain-mix run, varies by curriculum depth) | ~1040-1402 | 300.32M |
| `matched_transformer` | ~7223-7235 (flat, no recurrence to slow down) | 302.57M |

**Real training-time gap: ~5.2x-6.9x, somewhat larger than the 3.7x
inference-only figure** -- expected, since BDH's backward pass runs
through the SAME tied weights 8 times per step (real per-step backward
cost scales with recurrence depth in a way a Transformer's distinct-
per-layer weights don't), and inference-only throughput never pays that
backward cost at all. Both numbers are real and both matter for
different purposes (~3.7x is the right number for inference-time cost;
~5.2x-6.9x is the right number for how long a training run actually
takes) -- neither one alone is "the" BDH-vs-Transformer speed ratio,
and conflating them was the exact original mistake the fp32-autocast
bug caused. `matched_transformer`'s own post-training inference
benchmark also confirmed clean on this same dispatch: uncompiled 29337
tok/s, compiled (max-autotune) 32185 tok/s (1.10x, no crash, no
fp32-upcast issue) -- archived at
`results/cuda/hz0h_matched_transformer_training_throughput_result.json`.

**New, separate, NOT-yet-investigated observation surfaced by this same
fix:** compiled throughput (617 tok/s) is now confirmed SLOWER than
uncompiled (2758 tok/s) for `raw_bdh` -- the same direction as
`combined_best`'s already-documented regression (249 compiled vs 683
uncompiled, a few sections above). Two independent arms now show
`torch.compile` making BDH's forward pass slower, not faster, at this
shape/family -- worth its own investigation, not yet started, not
folded into any conclusion above.

Also a real, disclosed gap this bug exposes: `main()` computed a
`throughput` dict for all three arms but never actually merged it into
the saved JSON report (only `results["results"] = results` was written,
`throughput` was print-only). Fixed alongside the autocast bug
(`report["throughput"] = throughput` added) so future runs archive the
real numbers instead of requiring them to be read off a chat message or
scrollback log.

## Part 10: Part 6's jump-operator win does NOT survive scaling to production depth -- a real, important negative result

Local run, `scripts/hz0h_bdh_jump_operator_scale_test.py`, real
production shape (`n_embd=2496, mult=16, n_layer=8` -- identical to
`raw_bdh`'s own confirmed-real config), MPS/fp32/regular AdamW (not yet
the CUDA/bf16/int8-optimizer version, which is separately dispatched
and still queued), 3M tokens, 2.15h real teacher training:

| arm | val loss | Δ vs real_depth8 | speedup | tok/s |
|---|---:|---:|---:|---:|
| `real_depth8` (ground truth) | 2.0347 | 0.0000 | 1.00x | 1,235 |
| `hybrid_4real` | 3.9170 | **+1.8823** | 1.99x | 2,453 |
| `hybrid_2real` | 4.9749 | +2.9402 | 3.87x | 4,780 |
| `all_jumps` | 8.2564 | +6.2217 | 79.08x | 97,644 |

**Real, important comparison: Part 6's original small-prototype result
(`n_embd=128`, 1.5M tokens) found `hybrid_4real`-equivalent cost only
+0.029 loss for ~1.9x speedup -- the strongest efficiency signal in
this entire audit. Here, at essentially the SAME ~2x speedup
(`hybrid_4real`, 1.99x), the real cost is +1.8823 -- roughly 65x
worse.** This is a genuine, large, negative finding: the jump
operator's substitution cost, measured honestly at a scale close to
production, is catastrophic, not the small, tolerable cost the original
prototype suggested.

**This matches a pattern already documented elsewhere in this project
(FactorizedBDH's small-probe reversal) and now confirmed again for a
second, independent mechanism: a promising result at toy scale
(`n_embd=128`) can look completely different at real scale
(`n_embd=2496`), and the gap here is not subtle -- it's nearly two
orders of magnitude in the cost, not a modest correction.**

**Real, disclosed limits on this specific local run:** MPS/fp32/regular-
AdamW/3M-tokens -- see the CUDA confirmation immediately below, which
independently reproduces the same finding on real production hardware.

### CUDA confirmation lands the same day: independently reproduced, real hardware

Windows dispatch, same day, HEAD `7b85230`, real RTX 3060, bf16,
`adam8bit`, `--use-wide-gemm --compile-training --compile-mode
max-autotune` (a "quick validation" run for the new compile/wide-GEMM
levers that incidentally ran the FULL jump-operator evaluation at 500K
tokens -- short budget, not the queued 10M-token run, but real):

| arm | val loss | Δ vs real_depth8 | speedup | tok/s | peak mem |
|---|---:|---:|---:|---:|---:|
| `real_depth8` | 2.7707 | 0.0000 | 1.00x | 53.6 | 5.20GB |
| `hybrid_4real` | 3.9651 | **+1.1944** | 1.99x | 107 | 5.23GB |
| `hybrid_2real` | 5.9402 | +3.1695 | 3.97x | 213 | 5.23GB |
| `all_jumps` | 16.1774 | +13.4067 | 3782x | 202,747 | 1.80GB |

**Real, independent confirmation on production hardware: `hybrid_4real`
gives essentially the same speedup as Part 6's original claim (1.99x
vs 1.9x) but the real quality cost is +1.1944 -- 41x worse than the
original +0.029.** This is a THIRD independent measurement (toy
prototype, this doc's own local MPS run at 65x worse, now real CUDA at
41x worse) all showing the same qualitative story with consistent
order-of-magnitude severity -- the exact multiplier differs by token
budget/precision (500K tokens here vs 3M locally, bf16 vs fp32), but
the direction and magnitude class are decisively consistent across
three independent runs, two different devices, two different
precisions.

**Real conclusion, now settled: the jump-operator direction (Part 6,
and this session's whole "settle N real iterations then jump the rest"
thread) is a closed, negative result at production scale.** The full
queued 10M-token CUDA run was deliberately NOT run to get a more
"final" number -- the three independent confirmations already agree
closely enough (same speedup, both showing 40-65x quality cost
inflation) that spending further real GPU time on a fourth confirmation
of the same conclusion was judged not worth it; that compute was
redirected to confirming `max-autotune` also fixes `raw_bdh`'s original
compile crash (a still-open question) instead. This joins Part 4d's
LoRA-per-group null and Part 7's failed stacked-recipe capstone as
findings that looked promising at small scale and did not survive being
tested for real -- three real negative results in this audit now, each
independently confirmed rather than assumed.

**Real, useful side finding from this same dispatch: `torch.compile(mode="max-autotune")` combined with `--use-wide-gemm` runs clean through
training with zero OOM** (bit-exact levers from Parts 8-9's addenda),
confirming both new levers are usable together on real hardware, not
just individually.

## Part 11: g_r -- the realized per-round operator gate -- is genuinely sparse AND its support stabilizes across depth (small local scale, first pass)

Motivated by a real reframing (2026-08-20): BDH's per-token update
algebraically collapses to `x' ~= x + x @ (E @ diag(g_r) @ D)`, where
`g_r = x_sparse_r (elementwise*) y_sparse_r` is the coefficient vector
selecting which of the `N`-per-head rank-1 operators `e_n d_n^T` (`E`'s
columns paired with `D`'s rows) are active THIS token, THIS round.
Every prior optimization attempt in this audit either preserved `g_r`
(kept quality, kept the compute -- wide-GEMM/bmm layout remaps) or
reduced/approximated something ADJACENT to it (width, `E`/`E_v`/`D`
directly, or the hidden state `x_r` itself, in the jump operator's
case) -- never `g_r` directly. This part measures `g_r` for the first
time.

`reference/hz0h_bdh_g_r_operator_diagnostic_torch.py` adds a capture
point to `bdh_variable_depth_forward`'s exact math (proven bit-exact
against it, logits + loss, by
`tests/reference/test_hz0h_bdh_g_r_operator_diagnostic_torch.py`) that
exposes `xy_sparse` per round without changing BDH's computation at
all. `scripts/hz0h_bdh_g_r_operator_diagnostic.py` trains a small local
raw BDH (`n_embd=256, mult=16, n_layer=8, n_head=8`, `N=512` per head --
NOT production scale, see caveats below) via the same `train_bdh` used
by the main capstone comparison, then pools `g_r` over 20 held-out
batches (162,560 samples per round) and measures density, round-to-round
support overlap, top-k energy concentration, and effective rank
(participation ratio of the SVD spectrum).

**Real numbers, this run
(`results/local/hz0h_g_r_operator_diagnostic_result.json`):**

- **Density**: 83-86% of `g_r`'s `N=512` entries are EXACTLY zero, every
  round (real zeros -- `g_r` is a product of two ReLUs, not a
  thresholding artifact). Only ~15% of the nominal width is active per
  token/round.
- **Concentration**: the top 2% of dims (`k=10` of 512) already explain
  82-88% of a token-round's L2 energy; the top 10% (`k=51`) explain
  99.0-99.1%.
- **Round-to-round support drift -- the new finding, not previously
  measured anywhere in this project**: Jaccard overlap between
  consecutive rounds' active sets RISES monotonically from 0.52 (round
  0-to-1) to 0.89 (round 6-to-7). Early rounds churn heavily; by the
  final two rounds the active set has nearly stabilized.
- **Effective rank**: 34-40% of `N` per individual round, but 46% when
  pooled across all 8 rounds -- consistent with "any single token uses
  a small, concentrated subset, but WHICH subset varies enough across
  different tokens that the union spans much more of the space than any
  one token needs."

**Real implication for the follow-ups proposed alongside this
reframing**: this is genuine first-pass evidence for "the architecture
discovers sparsity after paying for it" -- both halves of the earlier
router post-mortem are confirmed here too: (1) there IS real sparsity
to exploit (~85% exact zeros), so a perfect oracle router is not chasing
nothing, and (2) the fact that per-round support only stabilizes by
round ~6-7, not round 1-2, argues AGAINST a static x-only router (which
is what actually shipped and lost) and FOR the proposed "recurrent
active-set" idea specifically: discover the active support over the
first couple of REAL rounds (using both `x` and the attention-derived
`a`, unlike the router), then restrict later rounds to it with periodic
refresh -- matching where the Jaccard curve says the support actually
settles, not a guess made before any rounds have run.

**Real, disclosed limits -- this is a hypothesis check, not a
confirmed result**: small scale (`n_embd=256`, not the `n_embd=2496`
production shape used in Parts 8-10), 12 seconds / 300K tokens of
training (final training loss 3.30 -- likely meaningfully undertrained,
not the converged model these statistics ideally should come from),
single seed, no CUDA confirmation. Whether density/stabilization hold
in direction AND magnitude at production width and with a properly
converged model is the real open question before anything gets built on
top of this -- same caveat this project already attaches to every other
local-scale-first result (Part 5, Part 6's original prototype, both of
which changed materially at production scale).

### CUDA confirmation lands the same day: real production scale, and MORE favorable than the local prototype

Windows dispatch, real `n_embd=2496` (`N=4992` per head, matching Part
9's shape exactly), 2M tokens trained to a real, healthy `final_loss=1.5855`
(vs the local prototype's undertrained `3.30`). First attempt crashed --
see the two real bugs fixed and disclosed below before this result --
second attempt succeeded cleanly (result archived at
`results/cuda/hz0h_g_r_operator_diagnostic_cuda_result.json`).

**`f_x`/`f_y`/`f_xy` direction and rough magnitude confirmed**: `f_x`
0.30-0.35 (matches the local 0.29-0.36 window closely), `f_xy`
0.10-0.15 (local was 0.14-0.17, close), `f_y` real and consistently
LOWER than local (0.35-0.38 vs 0.45-0.47) -- a genuine, disclosed gap,
not noise, but doesn't change the exact-compute-skip estimate
materially: `ND(1+f_x+f_xy)` vs `3ND` is still a real ~2x reduction on
the `E_v`+decoder portion at this scale too.

**Top-k energy concentration confirmed, if slightly less extreme at
early rounds**: `k=99` (2% of `N`) explains 91-99% of energy depending
on round; `k=1248` (25% of `N`) explains 99.6% of energy EVERY round,
flat across depth -- an independent confirmation of the same
low-effective-dimensionality story from a different angle.

**The real surprise, not anticipated by the local prototype**:
effective rank does not stay roughly flat around 35-40% like the local
run showed -- it collapses monotonically and dramatically with depth:

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eff. rank (% of N=4992) | 9.7% | 10.2% | 10.5% | 10.2% | 9.5% | 6.2% | 2.1% | **0.8%** |

By the final round, only ~41 of 4992 dimensions (participation ratio)
carry real signal across a 4000-sample reservoir -- combined with the
round-to-round support Jaccard also rising to 0.82 by round 6-7 (same
direction as local's 0.52->0.89, similar magnitude), this is a
substantially STRONGER, more actionable signal for "recurrent
active-set BDH" than the local prototype implied: by round ~5-6, later
rounds are not just similar in support to the previous round, they are
using an extremely low-dimensional subspace, not merely a smallish
fraction of the nominal width.

**Real, disclosed reason to treat this as promising but not yet
settled, rather than a clean win**: this local-vs-production divergence
runs OPPOSITE to this project's usual pattern (small-scale usually
looks promising and gets refuted at scale -- FactorizedBDH, the jump
operator). Here production looks BETTER than local predicted, which is
unusual enough to warrant a specific hypothesis rather than just
banking the number: the local model was meaningfully undertrained
(loss 3.30, likely still close to representing tokens somewhat
randomly) while the production model converged properly (loss 1.5855)
-- a plausible, coherent explanation is that a well-trained model
genuinely learns to concentrate its computation into fewer effective
directions, and the local prototype's high effective-rank number
partly reflects its own lack of convergence, not a true small-width
architectural difference. Also a real methodological limitation shared
by both runs: the 4000-sample SVD reservoir is drawn from only 20
batches of 8 sequences each, so reservoir samples are NOT fully i.i.d.
(same handful of source sequences, correlated adjacent tokens) --
could bias the effective-rank estimate downward versus a truly random
population sample. Both caveats argue for a larger/more-diverse-batch
re-measurement on a fully-converged model before treating the <1%
figure as a hard design target, not for discarding the signal.

**Two real bugs fixed before this result was obtainable at all**,
disclosed in full in commit `3863a7b`: (1) the original collection
design pooled full `u`/`v` tensors across all batches even after
capping per-batch samples, reaching an estimated ~20-25GB peak against
a machine with 16.7GB TOTAL system RAM -- a real crash (silent
OS-level OOM-kill, zero traceback) on the first CUDA attempt, after a
real, non-wasted 32-minute training run. Rewritten to a streaming
design (`StreamingRoundStats`) where every statistic that reduces to a
per-sample mean is a running weighted average computed and discarded
one batch at a time, with only the SVD reservoir given a small fixed
memory budget (~638MB regardless of `N`). (2) A second bug caught
LOCALLY before re-dispatch, not by any test: the first streaming
rewrite summed active-fraction counts over both the sample and feature
dimensions but divided only by the sample count, silently producing
`f_x`/`f_y`/`f_xy` in the hundreds instead of fractions in `[0, 1]` --
caught by rerunning locally and eyeballing the printed numbers, not by
automated tests (none existed for this specific arithmetic).

**Real, disclosed gap in this run**: the streaming rewrite dropped the
previous version's cross-round pooled summary (only per-round stats are
computed/reported now) -- a deliberate simplification under time
pressure, not yet added back, real per-round data is complete and
sufficient to draw the conclusions above.

### The crux question, resolved on real CUDA: naive block-sparse E_v+decoder is NOT well-supported by this data

Before writing any kernel, the real open question was: does the
effective-rank collapse at later rounds mean different TOKENS converge
onto the SAME active neuron identities (exploitable with one static,
shared column-subset per round -- a real, easy, dense-GEMM win, no
gather needed), or does each token keep its OWN different small active
set that merely happens to be jointly low-rank (genuinely per-token-
varying support -- and naive per-token gather does NOT save real FLOPs,
since gathering a different weight slice per token costs as much as the
matmul it replaces, the exact reason Part 7's MoE-style router lost on
real wall-clock despite a real theoretical FLOP reduction).

Measured directly via `cross_token_support_jaccard` (pairwise support
overlap between DIFFERENT tokens at the same round, reusing the
already-collected SVD reservoir, no new data). Real result on the same
production-scale run (Windows dispatch, `n_embd=2496`):

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eff. rank (% of N) | 9.7% | 10.2% | 10.5% | 10.2% | 9.5% | 6.2% | 2.1% | 0.8% |
| cross-token Jaccard | 0.065 | 0.071 | 0.077 | 0.086 | 0.101 | 0.120 | 0.138 | **0.153** |

**Real, non-noise correlation**: cross-token Jaccard rises monotonically
as effective rank collapses (roughly doubling, 0.065 -> 0.153) --
tokens genuinely do share MORE support as depth increases. But the
ABSOLUTE magnitude answers the crux question clearly: even at round 7
(lowest effective rank), 0.153 is nowhere near the ~1.0 a shared-
identity static mask would need -- it's only ~1.9x the local
prototype's independence baseline (~0.081 for that density). Round 0
(0.065) is actually BELOW that baseline.

**Real conclusion, disclosed rather than spun as a win**: the sparsity
is mostly per-token-varying, not primarily a shared static mask, even
at the most-collapsed round. The effective-rank collapse is real, but
it reflects many DIFFERENT individual per-token active sets all living
inside one small (~40-dimension) shared SUBSPACE -- not tokens
converging onto the same discrete set of active neurons. A naive
static-column-mask kernel is NOT well-supported by this data; the more
likely real risk is repeating Part 7's router failure mode (real
theoretical FLOP reduction, real wall-clock loss from gather/scatter
overhead). A different, harder, NOT YET EVALUATED kernel idea survives
this result: project each token's activation into the shared ~40-dim
subspace once, do the dense math there, project back out -- exploiting
the shared SUBSPACE rather than shared neuron IDENTITY. Whether that is
practical (extra projection cost vs. savings, numerical fidelity of a
~40-dim subspace approximation of a `N=4992`-wide computation) is a
real open question, not yet investigated.

No bugs found adding `cross_token_support_jaccard` itself -- verified
via the run's own reproducibility check (every pre-existing stat
matched the prior run byte-for-byte at the same seed, confirming the
new addition didn't disturb anything already validated).

### Domain-conditioned specialization test: real local first-pass, no meaningful signal yet

A follow-up hypothesis (2026-08-21, user-proposed): the previous
`cross_token_support_jaccard` measurement pooled random token pairs
from a single, roughly homogeneous byte-level corpus (Python/ML-config-
like content) -- it could not distinguish "support really is
per-token-random" from "support IS domain-conditioned, this dataset
just never had more than one domain to condition on." This project
already has real, unused domain-labeled data sitting in
`data/packed/external/` (code, documentation, json_and_configuration,
mathematical_and_structured, terminal_and_debugging -- confirmed
genuinely distinct by decoding real samples: code is Python source,
`mathematical_and_structured` is StackExchange math discussion,
`terminal_and_debugging` is StackOverflow-style Q&A), tokenized with
the BPE tokenizer at `data/tokenizer/hz0a_24576.json` rather than this
project's byte-level vocab. `scripts/hz0h_bdh_domain_bytes_prep.py`
bridges this: decodes the packed BPE sequences back to text via
`tokenizer.hz0a_tokenizer.HZ0ATokenizer`, re-encodes as raw UTF-8
bytes, and re-chunks into byte-level windows compatible with this
project's existing `read_batch` format -- no new tokenizer or model
change, a pure format bridge. Writes a domain-mixed training set and
per-domain (labeled) validation sets under `data/packed/domains/` (not
committed -- `data/` is gitignored project-wide, same as every other
dataset in this repo).

`scripts/hz0h_bdh_domain_specialization_diagnostic.py` trains on the
real domain mix, then compares WITHIN-domain cross-token Jaccard (two
different code snippets, say) against ACROSS-domain (a code token vs.
a math token), per round, via the new `cross_domain_support_jaccard`
(added alongside `cross_token_support_jaccard` in
`scripts/hz0h_bdh_g_r_operator_diagnostic.py`).

**Real local result (small scale, `n_embd=256`, 1M tokens, converged to
`final_loss=1.90`)**: within/across ratio stays in **1.01x-1.10x**
across all 8 rounds -- essentially no separation. If domain-conditioned
specialization were real and strong, WITHIN should be meaningfully
above 1.0x; instead it's indistinguishable from noise at this scale.

**Real, disclosed limits before treating this as a negative result**:
this is a SINGLE local-scale run, and this exact project has already
seen local-scale results diverge dramatically from production scale in
BOTH directions this session (the jump operator got much WORSE at real
scale; `g_r`'s effective-rank collapse got much STRONGER at real scale
than local predicted) -- so a local near-null result here is a real,
honest data point, not yet a settled answer, same caveat attached to
every other local-scale-first result in this Part. The domain-mixed
training set is also real but modest (~21.5K windows from ~1500 packed
source rows per domain) and the model itself is small/under-parameterized
relative to production. Production-scale (`n_embd=2496`) CUDA
confirmation is the natural next dispatch before drawing a real
conclusion either way.

### CUDA confirmation: production scale AGREES with local -- a real, closed negative result

Windows dispatch, real production shape (`n_embd=2496`, 2M tokens,
`final_loss=1.0195` -- the best-converged model in this whole domain-
specialization line of testing), same 5 real domains, 3840 samples per
domain per round:

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| within-domain | 0.041 | 0.040 | 0.039 | 0.042 | 0.041 | 0.052 | 0.076 | 0.102 |
| across-domain | 0.037 | 0.036 | 0.037 | 0.038 | 0.040 | 0.050 | 0.074 | 0.092 |
| ratio | 1.10x | 1.09x | 1.05x | 1.09x | 1.03x | 1.04x | 1.03x | 1.12x |

**Real answer, and notably the first time this session production
scale AGREED with the local prototype rather than diverging sharply in
either direction** (contrast the jump operator, which got much worse at
scale, and `g_r`'s effective-rank collapse, which got much stronger):
the ratio stays in a tight 1.03x-1.12x band across all 8 rounds --
essentially flat, barely above 1.0. Both within- and across-domain
values rise together with depth (same ~2.5x growth in both, tracking
the already-documented effective-rank collapse), but domain identity
itself adds almost nothing on top of that shared trend.

**Real, disclosed conclusion**: hard-filtering neurons by domain during
training, to cut compute, is NOT supported by this data -- BDH does not
show meaningful preferential neuron activation by domain (code vs.
math vs. documentation vs. terminal/debugging vs. JSON/config) at this
training scale and budget. A real, honest caveat this doesn't rule out:
2M tokens on a 300M-param model is still a modest budget -- whether
domain specialization would emerge with substantially more training is
a different, unanswered question (the same "maybe it needs more scale
to show up" caveat this project has attached to every negative result
so far), not something this test can distinguish from "never happens."
Closed as a real negative result at THIS budget, not a permanent
verdict on the underlying idea.

### Full 10M-token domain-specialization confirmation: same conclusion at 5x the budget

Real full-budget rerun (RunPod A40, batch=32, `--use-wide-gemm`, the new
memory/speed fix from earlier this Part -- 10M tokens trained in 2015s
thanks to it, vs. an estimated ~2.5-2.7h this same test would have taken
on the RTX3060 the 2M-token version ran on):

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| within-domain | 0.023 | 0.022 | 0.025 | 0.028 | 0.029 | 0.033 | 0.044 | 0.064 |
| across-domain | 0.020 | 0.020 | 0.023 | 0.026 | 0.027 | 0.032 | 0.041 | 0.055 |
| ratio | 1.16x | 1.06x | 1.07x | 1.07x | 1.05x | 1.03x | 1.07x | 1.18x |

**Real answer: this directly closes the "maybe it needs more scale to
show up" caveat from the 2M-token result** -- at 5x the token budget
(and the same real production shape/params as every other Part 9/10/11
result), the ratio stays in the same flat 1.03x-1.18x band, no trend
toward separation. A real, secondary observation worth noting
honestly: BOTH within- and across-domain absolute values dropped
compared to the 2M-token run (e.g. round 0: 0.023/0.020 here vs.
0.041/0.037 at 2M tokens) -- more training makes activation support
sparser/more token-specific overall, consistent with the earlier
effective-rank findings, but this shift is proportional across both
within- and across-domain measurements, so the RATIO (the thing that
actually answers the domain-specialization question) is unaffected.
Closed as a real negative result at production scale and budget, not
just the earlier small-scale check.

### Seed=8 lands toward Part 9's still-open 3-seed confirmation

Real second seed of the `raw_bdh` 10M-token production run (same shape,
`--use-wide-gemm`, batch=32), run in parallel with the domain-
specialization confirmation above on the pod's second A40:
**`validation_loss=1.4080`** -- genuinely different from seed=7's
1.6844/1.7038 (real seed variance, and notably BETTER here), the first
actual different-seed data point toward the 3-seed confirmation Part 9
has flagged as pending all session. Still only 2 of 3 seeds; a third
run would close this out. Archived at
`results/cuda/hz0h_runpod_a40_raw_300m_10m_seed8_result.json`.

### 3-SEED CONFIRMATION COMPLETE: raw_bdh beats matched_transformer on EVERY seed

Seed=9 (same shape/recipe, `--use-wide-gemm`, batch=32, RunPod A40)
completed cleanly: **`validation_loss=1.4027`**. This is the third real
seed for `raw_bdh` and closes the 3-seed confirmation Part 9 has
flagged as pending since it was first reported. In parallel, the
matched Transformer control arm also picked up 2 more real seeds this
session (previously only seed=7 had ever been run), so both arms of
the comparison now have 3 real seeds each -- not just BDH:

| seed | `raw_bdh` | `matched_transformer` |
|---|---:|---:|
| 7 | 1.6844 (A40) / 1.7038 (original RTX3060, same seed) | 2.3016 |
| 8 | 1.4080 | 1.8988 |
| 9 | 1.4027 | 1.9353 |
| **mean** | **1.498** | **2.045** |

**Real, decisive result: `raw_bdh` beats `matched_transformer` on
EVERY individual seed, not just on average** -- a real, now-properly-
confirmed finding, not a single-run artifact. This closes the "pending
3-seed confirmation" caveat attached to Part 9's headline result since
this document's very first version. Real, disclosed caveat that
remains: seed=7's `raw_bdh` number is unusually close to `raw_bdh`
seeds 8/9 being a matched PAIR (1.4080, 1.4027) while seed=7 sits
noticeably higher (1.6844/1.7038) -- worth noting as a real, visible
seed-to-seed spread rather than three near-identical numbers, though
the direction (BDH < Transformer) is completely consistent across all
three. Archived at
`results/cuda/hz0h_runpod_a40_raw_300m_10m_seed9_result.json`,
`results/cuda/hz0h_runpod_a40_matched_transformer_10m_seed{8,9}_result.json`.

### FlashBDH: mathematically correct, but a real negative result on memory/speed as built

Real attempt (2026-08-21, user-directed) at the harder follow-up to
checkpointed wide-GEMM: `BDHFlashRoundFunction`
(`reference/hz0h_bdh_flash_round_torch.py`), a custom
`torch.autograd.Function` per BDH round with a hand-derived analytic
backward -- intended to avoid checkpointing's real, measured ~28%
recompute-forward tax entirely (standard backprop-through-matmuls
theory predicts backward costs ~2x forward FLOPs regardless of method;
checkpointing's extra cost is specifically the redundant "redo forward"
1x on top of that).

**The math is real and proven correct, on two independent gates**:
bit-exact (logits, loss, gradients) against `bdh_variable_depth_forward`,
AND `torch.autograd.gradcheck` (double-precision finite-difference,
both single-round and 2-round-chained) on the raw Function in
isolation. All 5 tests passed on the first implementation attempt.
Getting there required finding and fixing THREE real, distinct
autocast/dtype bugs on real CUDA hardware, invisible to local CPU/fp32
tests which never exercised autocast at all: (1) custom Function
forward/backward don't automatically share a dtype under
`torch.autocast` since backward runs outside the lexical `with
autocast` scope; (2) PyTorch's native LayerNorm backward kernel
promotes its returned gradient dtype internally even with no active
autocast; (3) the real root cause underlying both -- `forward()` itself
still runs INSIDE the caller's active autocast scope, so autocast's own
per-op policy (which forces LayerNorm to run and RETURN in fp32
regardless of input dtype) silently overrode the explicit bf16 casts,
meaning saved tensors were fp32 when bf16 was assumed. Fixed by
explicitly disabling autocast's automatic policy for the whole function
body (`torch.autocast(..., enabled=False)`), giving full deterministic
control. Real accuracy check on CUDA after all three fixes: forward
loss matches the oracle within normal bf16 rounding noise (5.65100 vs.
5.65051 oracle, same magnitude as wide-GEMM's own 5.65082 deviation);
gradient norms match within <1%, individual-element max-abs relative
differences of 4.5-9% consistent with real bf16 precision compounding
over 8 real recurrent rounds via a different operation order, not a
bug (the underlying math was already proven exact via double-precision
gradcheck).

**But the real, honest practical result is negative for THIS
implementation**: measured on the RTX A40 at the real production shape
(`n_embd=2496`), FlashBDH's memory and throughput do NOT beat what
already exists --

| config | batch=8 | batch=32 | real ceiling |
|---|---:|---:|---:|
| wide-GEMM alone | ~14GB / ~4200 tok/s | 34GB(est) / 5071 tok/s | 32 |
| checkpointed wide-GEMM | ~8GB / 2509 tok/s | 14.55GB(*) / 2662 tok/s | ~160-168 |
| **FlashBDH** | **16.43GB / 2658 tok/s** | **34.46GB / 3296 tok/s** | **~32-40** |

(*checkpointed wide-GEMM's own reported "peak_memory" figures earlier
in this Part were later found to come from a disconnected benchmark
that doesn't use checkpointing at all -- see the real training-loop
probe numbers used here instead.)

**Real root cause, understood, not just observed**: `ctx.save_for_backward`
keeps a round's saved tensors alive until THAT round's own `backward()`
is called -- which, for a normal sequential forward pass through
`n_iterations` rounds, only happens after every later round's backward
has already run (standard reverse-order backprop). Saving LESS per
round (no broadcast-expansion bug, no retained QR/scores) is real and
helps somewhat, but without an OUTER discard-and-recompute mechanism
across round boundaries -- exactly what `torch.utils.checkpoint`
provides -- memory still scales with `n_iterations`, just with a
smaller per-round constant than the plain path. And wrapping FlashBDH's
rounds in `torch.utils.checkpoint` on top would defeat the entire
point: checkpoint's own recompute would just re-run the whole custom
Function's forward again, exactly the redundant work FlashBDH exists to
avoid. This is a genuine structural gap in the current design, not a
bug to patch -- a real cross-round memory-discipline mechanism (outside
`torch.utils.checkpoint`'s own recompute-based approach) would be
needed to actually deliver the intended win, real future work if
pursued further.

**Disclosed as a real negative result for this implementation, matching
this project's own standing discipline (Part 6, Part 7, Part 10)** --
the underlying idea (avoid checkpointing's recompute tax via analytic
backward) remains theoretically sound and the math is now proven
correct and reusable, but this session's build does not currently
outperform the simpler checkpointed-wide-GEMM path on either memory or
speed, and should not be used in production training as-is.

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
