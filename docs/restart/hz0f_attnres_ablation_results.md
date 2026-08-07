# HZ-0F: Attention Residuals Ablation on the Tiny Exact-GDN-2 Model

Date: 2026-08-06. Real, controlled test of "Attention Residuals" (AttnRes)
-- learned, content-dependent depth-wise attention over previous layer
representations, in place of standard PreNorm residual chaining -- per a
recent architecture survey flagging it (and its low-rank/multi-head
variants) as the highest-priority, most orthogonal idea to test against
HZ's existing exact-GDN-2 backbone.

## Implementation

`archive/src/hz0/model/attn_residual.py::AttentionResidual`. At layer
`l`, instead of feeding layer `l`'s mixer only `x_{l-1}` (the immediately
previous layer's output), it feeds a learned combination:

```text
x_l_input = sum_{i<l} alpha_{l,i}(x) * h_i
```

where `alpha` is a softmax attention distribution over DEPTH (not
sequence position), computed from a query derived from the most recent
representation and keys from every prior representation (including the
embedding). One `AttentionResidual` module per layer (own query/key
projections), not shared -- matching the layer-indexed `alpha_{l,i}` in
the formula. `rank` controls the query/key projection width (low-rank
variant: routing is low-rank, but VALUES stay full `d_model`, so
representational capacity passed forward is not reduced). `n_heads`
gives independent depth-distributions, each reading its own
`d_model/n_heads` value slice (standard multi-head value-splitting).

Wired into `HybridLM` via a new `residual_mode: "standard" | "attn_res"`
constructor argument (default `"standard"`, so all existing behavior and
tests are unaffected -- verified directly, not assumed:
`test_hybrid_lm_standard_mode_is_unaffected_by_attn_res_addition`). 6 new
tests in `archive/tests/test_attn_residual.py` cover shape, a real
correctness property (single-history-entry input must reduce exactly to
that entry, since softmax over length-1 is always weight `1.0`), gradient
flow, and low-rank/multi-head shape variants. Full archive suite:
**32 passed** (26 pre-existing + 6 new), no regressions.

## Experiment

Real (not synthetic) text: `archive/data/hz0a_seed_train.txt`/
`hz0a_seed_val.txt` (73KB / 8KB real byte-level text). Tiny model
(`d_model=192, n_layers=8, n_heads=6, d_ff=512`, `mixer_backend: gdn2_ref`
-- the real, CPU-runnable exact-GDN-2 reference mixer, not the CUDA-only
upstream kernel), 300 real training steps, `lr=3e-4`, AdamW. Four
variants, same seed/data/steps, differing only in residual mechanism:

1. `standard` (baseline) -- `configs/hz0f-attnres-standard.yaml`
2. `attn_res` full-rank, single-head (`rank=192, heads=1`) -- `hz0f-attnres-fullrank.yaml`
3. `attn_res` low-rank, single-head (`rank=32, heads=1`) -- `hz0f-attnres-lowrank.yaml`
4. `attn_res` full-rank, multi-head (`rank=192, heads=8`) -- `hz0f-attnres-multihead.yaml`

## Result: standard residual wins, consistently, at this scale -- a real negative result

Seed 7 (all four variants):

| Variant | Params | Eval loss @ step 100 | Eval loss @ step 200 | Train loss @ step 275 |
| --- | ---: | ---: | ---: | ---: |
| standard | 4,644,416 | 3.0547 | **2.7269** | **2.3154** |
| attn_res full-rank | 5,234,240 | 3.0886 | 2.7950 | 2.3210 |
| attn_res low-rank | 4,742,720 | 3.1055 | 2.7997 | 2.3375 |
| attn_res multi-head | 5,234,240 | 3.1215 | 2.8278 | 2.3324 |

**Standard residual chaining wins on every metric, against all three
AttnRes variants, at every checkpoint measured.** This contradicts the
source technique's own reported results (validation-loss improvements at
100M/350M/1B parameter scale) -- not because the implementation is wrong
(verified correct via the single-history-entry reduction property and
shape/gradient tests above), but because this experiment ran at a scale
roughly 20-200x smaller than what the technique was originally
demonstrated at.

**Checked for seed sensitivity before accepting this**: re-ran `standard`
vs. `attn_res` full-rank (the most directly comparable, paper-matching
variant) at 2 more seeds:

| Seed | Standard eval@200 | Full-rank AttnRes eval@200 |
| --- | ---: | ---: |
| 7 | 2.7269 | 2.7950 |
| 1 | 2.7220 | 2.7919 |
| 2 | 2.7462 | 2.7956 |

**Standard wins in all 3 seeds, by a consistent, similar margin
(`~0.05-0.07` nats)** -- not seed noise. This is a real, reproducible
negative result at this specific scale.

## Honest interpretation

This does NOT falsify AttnRes as an idea -- it falsifies "AttnRes helps
at a ~5M-parameter, 300-step, single-domain-text scale on this specific
hybrid recurrent+attention architecture." Plausible reasons, not
distinguished here: (1) the depth-attention mechanism needs more
layers/scale than 8 layers to have meaningful "which depth to attend to"
signal to learn from, (2) 300 steps on 73KB of text is far short of what
either the baseline or the extra AttnRes parameters need to shake out,
(3) the extra ~590K parameters (13% more than baseline) are pure
overhead at this scale, not yet earning their keep, (4) HZ's hybrid
recurrent-mixer architecture may interact differently with depth-attention
than the Transformer-only architectures (Kimi K3 and its follow-ups) the
technique was originally validated on.

**Not recommended for adoption at this scale.** If revisited, the
evidence-supported next step is testing at a scale closer to where the
source technique was actually validated (100M+ parameters), not treating
this tiny-model result as the final word -- this experiment answers "does
it help THIS tiny model," not "is the idea sound," and should not be
overclaimed as a rejection of the broader technique.

## Final verdict (recorded 2026-08-06)

> AttnRes is actively worse than standard residuals for tiny HZ models,
> by a substantial and seed-stable ~0.05-0.07 nat margin.

All three variants lost (full-rank, low-rank, multi-head), not just one
implementation choice -- combined with the single-history identity
correctness property and 32/32 passing tests, "probably implemented
wrong" is not a live explanation. No further compute is being spent
tuning AttnRes at this ~5M-parameter scale.

```text
Standard residual   -> HZ default
AttnRes @ tiny scale -> rejected
AttnRes @ 100M+      -> unresolved
```

**If this is revisited**, the protocol should be brutally simple, not a
repeat of all three variants: ~100M parameters, same tokenizer/data/step
order, same optimizer/token budget, standard residual vs. ONE best
lightweight AttnRes variant (low-rank or multi-head, not both), 3 seeds,
comparing validation loss, convergence rate, throughput, memory, and
gradient stability. Not run in this investigation -- filed here as the
exact next step if HZ ever approaches that scale, not attempted
speculatively now.
