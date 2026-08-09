# HZ-0H T0: ternary training design memo

Date: 2026-08-08. Written on the RTX 3060/Windows/PyTorch side. Covers T0's
own exit bar: "quantization contract and success metrics documented." Scope
is the T0-T4 ternary lane as defined in
`plans/HZ-0H_BDH_Reconciliation_Plan.md` and
`plans/HZ-0H_Progress_Tracker.md` -- **not** a restatement of BitNet, and not
a claim that ternary training changes any architecture conclusion. Per the
tracker's own ternary guardrail:

> No result from ternary/1.58-bit training changes the HZ-0H architecture
> conclusion unless the corresponding full-precision control is already
> known. Treat ternary as an efficiency qualification layer on top of
> established architecture evidence, not as the evidence itself.

Everything below is written to respect that: T0-T4 answer "does ternary
preserve what the full-precision comparison already found," never "which
architecture is better," which stays H3-H7's job.

## What already exists (reuse, don't reinvent)

A BitNet b1.58-style ternary quantization scheme already exists in this
project, predating HZ-0H, built and validated for a different reason (the
HZ-0A 300M->1.5-3B scale-up motivation) -- see
`docs/rtx3060_windows_setup.md` section 5e and
`reference/hz0a_torch_model.py`'s `_ste_round_clip`/`BitLinear`/`_make_linear`:

- **Quantization formula:** absmean ternary -- `gamma = mean(|w|)` (one
  scalar per weight tensor), `w_q = round(w/gamma).clamp(-1,1) * gamma`.
- **Gradient:** straight-through estimator, `w + (w_q - w).detach()` --
  identity gradient w.r.t. the full-precision weight, verified exactly 1.0
  (unclipped) via a hand-computed unit check before being trusted.
- **What's quantized:** every `nn.Linear` in the model body (SwiGLU/mixer/
  attention projections) when `--bitnet` is passed. Embedding/LM-head and
  norm layers stay full precision -- standard BitNet b1.58 practice.
- **What's NOT quantized (yet):** activations. This is the weights-only
  "W1.58" half of BitNet's full "W1.58A8" scheme -- deliberately, to isolate
  the weight-quantization effect on its own first (this project's own
  one-variable-at-a-time discipline, applied consistently).
- **Already measured, real numbers:** no training-speed benefit on this
  hardware (matmul shape/FLOPs unchanged; only the weight *values* differ) --
  `--bitnet` gets the same ~4400 tok/s as the equivalent non-`--bitnet`
  config. The case for ternary is entirely about deployment-time footprint
  and a future low-bit-tensor-core hardware target, not making today's
  training run faster. A real-data ~16K-token training-trajectory comparison
  against unmodified `nn.Linear` showed noise-level divergence (9.380 vs
  9.346 final loss), not a real regression.

T0-T4 build on this rather than re-deriving it. The open work is: (a) apply
it to the *other* HZ-0H-relevant architectures (BDH-GPU, whose `encoder`/
`encoder_v`/`decoder`/`lm_head` are raw `nn.Parameter` matrices, not
`nn.Linear` modules, so the existing `BitLinear` class doesn't drop in
as-is -- needs a functional quantize-in-forward wrapper instead), and (b)
formalize the comparison contract so T1-T4's results are comparable to each
other and to the full-precision H3 baselines once those exist.

## Quantization contract

**In scope for ternary (T1-T4):**

| Architecture | Quantized | Excluded | Notes |
| --- | --- | --- | --- |
| HZ-0A hybrid (`gdn2`/`gdn3`/`gdn2_fix`/Transformer) | Every mixer/attention/SwiGLU `nn.Linear` | Embedding, LM-head, all `RMSNorm`/`LayerNorm` | Already implemented (`--bitnet`); no new code needed |
| BDH-GPU (`reference/hz0h_bdh_torch.py`) | `encoder`, `encoder_v`, `decoder` (the three shared low-rank matrices) | `embed`, `lm_head`, `ln` | Raw `nn.Parameter`, not `nn.Linear` -- apply `_ste_round_clip` functionally at each use site inside `forward`/`bdh_stream_chunk`, same formula, no new derivation |
| Matched Transformer (`reference/hz0a_matched_transformer.py`) | `qkv`, `attn_out`, `gate`, `up`, `down` | `embedding`, `final_norm` | **Built 2026-08-08** (was aspirational when this row was first written) -- `_make_linear` imported directly from `reference/hz0a_torch_model.py`, gated by `MatchedTransformerConfig(use_bitlinear=True)`; `--bitnet` now also reaches `--architecture transformer` in the runner (previously silently ignored there). See `docs/restart/hz0h_t2_matched_transformer_fp_vs_ternary.md` for T1/T2 results. |

**Consistent across all three:** absmean ternary, STE, weights-only
(activations stay at the run's `--dtype`), one scalar scale per weight
tensor (not per-row/per-channel -- matches the existing validated
implementation; a finer-grained scale is a real, separate future
experiment, not assumed better without its own A/B).

**Explicitly out of scope for T0-T4:** activation quantization (BitNet's
"A8" half), any non-ternary bit-width (this project's low-bit work so far
is 1.58-bit specifically, not a general N-bit sweep), and any ternary
variant of HZ-0B/HZ-0D/HZ-0E/HZ-0F/HZ-0G components not already covered by
H3's own architecture list.

## Success metrics

Each metric below states what "success" means for a *ternary* result --
i.e., what counts as "ternary preserved the full-precision picture," per
the guardrail. None of these are claims about which architecture is best;
that comparison lives entirely in the paired full-precision control.

1. **Training stability** (T1's bar): loss decreases below the
   random-baseline floor (`ln(vocab_size)` for a from-scratch LM) within a
   small, fixed token budget, with no NaN/Inf at any point
   (`assert_finite`-style check, matching this project's existing runner
   discipline) and no divergence over the run. This is a pass/fail gate,
   not a comparison -- it only asks "does ternary training work at all for
   this architecture," before any FP-vs-ternary comparison is meaningful.
2. **Convergence gap vs. the matched full-precision control** (T2's bar):
   same architecture, same data, same token budget, same seed, ternary vs.
   unmodified -- report final loss/perplexity delta as a real number
   (already have a template: the existing `--bitnet` HZ-0A comparison found
   ~noise-level, 9.380 vs 9.346). A gap judged "acceptable" is a judgment
   call to make explicitly in each T2 report, not a fixed universal
   threshold -- state the number and let the reader decide, don't just say
   "close enough."
3. **Memory footprint** (T2/T4): report actual bytes -- ternary's honest
   footprint claim is 1.58 bits/weight at *deployment* (packed ternary
   representation), not during this project's training runs (where the
   underlying `nn.Parameter` stays full-precision the whole time, per
   `BitLinear`'s own docstring -- don't conflate "trains with ternary-valued
   forward matmuls" with "uses less VRAM while training," those are
   different claims and only the first is what this repo's `--bitnet`
   currently does).
4. **Throughput** (T2/T4): report actual tok/s, training and (if a decode
   path exists) inference. Already known for HZ-0A hybrid: no training
   speedup on this hardware (see above) -- T2/T4 should re-measure per
   architecture rather than assume this transfers, since BDH-GPU's compute
   pattern (dense small matmuls via shared low-rank projections) differs
   from HZ-0A's per-layer projections.
5. **Ranking preservation** (T3's bar, the guardrail's central question):
   given the H3 full-precision ranking across BDH-GPU/GDN-2/Transformer,
   does the *same* ranking hold when all three are trained ternary under
   matched conditions? Report per-architecture convergence deltas (metric
   2) side by side -- ranking preservation is "the ordering of those deltas
   doesn't flip the H3 ordering," stated explicitly, not inferred.
6. **Graft qualification** (T4's bar): for any component that survived
   H7's full-precision graft evaluation, confirm the same graft still wins
   under ternary by the same predeclared H7 metric. A graft that only wins
   full-precision is a real, reportable outcome (not a failure of T4 to
   properly test it).

## Sequencing and dependencies

- **T0 (this memo): done.**
- **T1 (ternary sandbox, simple baselines):** can start immediately --
  doesn't need H3's full-precision numbers, only needs "does it train
  stably." Natural first target: HZ-0A hybrid (already has `--bitnet`,
  already validated) as the reference case, plus a first BDH-GPU ternary
  pass (new work, needs the functional wrapper described above) as the
  actual new T1 contribution.
- **T2 (same-architecture FP vs. ternary):** needs a same-architecture,
  same-conditions full-precision control run to pair against -- for HZ-0A
  hybrid this already exists (the `--bitnet` comparison cited above already
  *is* a T2-shaped result, just not labeled as part of this plan); for
  BDH-GPU, needs a fresh full-precision BDH-GPU run at matched
  scale/tokens/seed first.
- **T3 (post-H3 ternary replay):** hard-blocked on H3 (needs the real
  full-precision BDH-GPU vs. GDN-2 vs. Transformer ranking to exist before
  "does ternary preserve it" is answerable).
- **T4 (ternary graft qualification):** hard-blocked on H7.

## What "done" looks like for this memo

T0's own artifact requirement (`docs/restart/hz0h_ternary_training_design.md`)
is this file. Its job is done when T1 can start without re-deriving the
contract above -- it does not require T1-T4 to have run yet.
