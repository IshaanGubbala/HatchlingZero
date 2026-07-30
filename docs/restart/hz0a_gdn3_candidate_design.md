# Candidate "GDN-3" Recurrence: What Kimi K3/Linear Suggest HZ-0A Is Missing

Date: 2026-07-30. **This is a proposal, not a plan.** Nothing here is
implemented, tested, or scheduled. HZ-0A's architecture is deliberately
frozen (`plans/HZ-0A_Progress_Tracker.md`, "Both Stage 2 architectures are
now complete") -- Stage 2 finished, was full-holdout evaluated, and B6/B7/B8
all have real integration work built against that exact checkpoint. Nothing
below touches any of that. This document exists so the idea isn't lost, in
case a future retrain is ever undertaken.

## 1. What prompted this

Moonshot AI published the Kimi K3 technical report on 2026-07-27 (a 2.8T-
parameter MoE, built on their earlier "Kimi Linear" architecture,
[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)). Its recurrent mixer,
**Kimi Delta Attention (KDA)**, is in the same family as HZ-0A's own GDN-2
("Gated DeltaNet")-named mixer -- close enough that a direct comparison was
worth doing.

## 2. What HZ-0A's GDN-2 actually computes today

From `reference/hz0a_mlx_model.py`'s `GDN2.__call__` (the Python/MLX
fallback path; the native Metal kernel implements the same math):

```python
state = decay[:, t, :, None, :] * (1 - erase[:, t, :, None, :]) * state \
        + write[:, t, :, :, None] * v[:, t, :, :, None] * k[:, t, :, None, :]
output_t = sum(state * q[:, t, :, None, :], axis=-1)
```

In matrix form, per head, per step: `S_t = decay_t ⊙ (1 - erase_t) ⊙
S_{t-1} + write_t · (v_t ⊗ k_t)`, where `decay_t`/`erase_t`/`write_t` are
learned, per-channel (indexed along the key dimension) sigmoid gates. This
is a **gated linear attention (GLA)-style update**: state decays
elementwise, then a new outer-product term is added. `reference/hz0a_torch_model.py`'s
`GDN2Mixer` implements the identical equation.

## 3. What the actual delta rule computes (Kimi Linear / KDA)

From the Kimi Linear paper: `S_t = (I - β_t k_t k_t^T) · Diag(α_t) ·
S_{t-1} + β_t k_t v_t^T`.

The difference is the `(I - β_t k_t k_t^T)` term -- a rank-1 projection
that removes exactly the component of `S_{t-1}` that would produce the OLD
value when queried with `k_t`, before writing the new `v_t`. This is a real
associative **overwrite**: "whatever this key used to map to, forget
precisely that, then write the new mapping." KDA's own stated motivation
for extending Gated DeltaNet is finer-grained (per-channel, via `Diag(α_t)`)
gating on top of this same delta-rule projection.

**HZ-0A's GDN-2 has no `k_t k_t^T` projection term anywhere.** Its `erase`
gate is a plain elementwise multiply, uninformed by the key's own
direction -- it can dampen the whole state (or specific channels)
uniformly, but it cannot selectively "forget what this specific key used
to represent" the way the true delta rule can. Despite the "DeltaNet"
naming, the recurrence as implemented is closer to plain GLA than to
delta-net. This is a real architectural gap, verified by reading the exact
equations, not assumed from the name.

## 4. Candidate GDN-3 recurrence

```
S_t = (I - β_t k_t k_t^T) · Diag(α_t) · S_{t-1} + β_t k_t v_t^T
```

directly adopting KDA's formulation, with:

- `α_t` (per-channel decay, `Diag`): HZ-0A's current `decay` gate is
  already indexed per key-channel (see section 2), so this part may already
  be structurally close -- worth confirming precisely against KDA's exact
  gating parameterization before assuming a change is needed here at all.
- `β_t` (a new, single learned scalar-per-head "write strength" gate,
  distinct from the current `write` gate): needs a new small projection
  (`in_proj` would grow from 6 heads-worth of projections to 7, or `β_t`
  could be derived from the existing `write` gate's own logit -- an open
  design choice, not resolved here).
- The `k_t k_t^T` projection itself requires a per-step matrix-vector
  product against the FULL `[d_v, d_k]` state (`S_{t-1} @ k_t` then
  outer-producted with `k_t` again) -- more FLOPs per step than the current
  elementwise update, though delta-net's own literature (and KDA's
  "specialized DPLR chunkwise algorithm... substantially reduces
  computation compared to the general DPLR formulation") suggests this is
  made tractable via a chunked/blocked scan formulation, not a naive
  per-token loop -- the same category of optimization HZ-0A's own native
  Metal kernel already does for its current (simpler) recurrence.

## 5. Two smaller, more separable ideas from the same K2/K3 report

- **QK-Clip / MuonClip** (Kimi K2): a hard cap on max attention logit
  during training (Muon-style orthogonalized updates + a post-update clip
  on attention Q/K projections), credited with "zero training instability"
  at 1T-param scale. This is separable from the recurrence question above
  -- it targets HZ-0A's 6 causal-attention layers specifically, not the 25
  GDN-2 layers, and could in principle be trialed independently of any
  recurrence change. Not itself investigated further here.
- **Attention Residuals (AttnRes)** (K3): cross-layer residual routing
  weighted by an attention score rather than uniform accumulation,
  reported at ~4%/2% train/inference overhead. The public sources
  available when this was researched (2026-07-30) were thin on the exact
  mechanism -- worth revisiting once Moonshot's full 47-page K3 technical
  report is more thoroughly indexed/explained, rather than acting on the
  vague description available now.

## 6. Controlled benchmark result (2026-07-30)

The scientific gate proposed in earlier review of this document has been
run at small, controlled scale -- see
`docs/restart/hz0a_gdn3_overwrite_benchmark_results.md` for full numbers.
Summary: current HZ-0A's recurrence has a real, measured tradeoff where
clean key overwrite requires an erase strength that also destroys 90-99%
of unrelated stored content's magnitude (a blanket, key-blind mechanism);
the delta-rule projection achieves clean overwrite with zero (or, with
mild added decay, near-zero) collateral damage to unrelated keys, exactly
as its equations predict. This is real evidence at controlled scale, not
yet evidence at language-modeling scale -- see that doc's section 5 for
what is and isn't justified by it. The systems gate (real kernel cost at
301M scale) remains unanswered.

## 7. Honest scope of this document

- Nothing here has been implemented, benchmarked, or gradient-checked.
- Adopting section 4 would be a genuine architecture change requiring: a
  new native Metal kernel (the existing one implements the current,
  simpler recurrence), a new PyTorch reference implementation, re-deriving
  and re-verifying backward passes (the same rigor HZ-0A's own A3/A8
  phases went through for the current recurrence), and a full retrain --
  not a patch to the frozen Stage 2 checkpoint.
- This is real, credible evidence (from a shipped, evaluated frontier
  model's own technical report) that a specific, identifiable piece of
  HZ-0A's recurrence differs from what its own naming implies, and that
  the missing piece is plausibly load-bearing for expressiveness. It is
  not evidence that fixing it would clearly improve HZ-0A's own results at
  HZ-0A's much smaller scale (~301M vs. K3's 2.8T params) -- that would
  need to be measured, not assumed.
