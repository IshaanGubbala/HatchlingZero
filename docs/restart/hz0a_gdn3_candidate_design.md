# Candidate "GDN-3" Recurrence: What Kimi K3/Linear Suggest HZ-0A Is Missing

Date: 2026-07-30. **Status: fully investigated, NOT recommended -- a
single-seed positive result did not replicate across 4 seeds.** See
section 8 for the final verdict. Nothing here is implemented in or
scheduled for the real HZ-0A. HZ-0A's architecture is
deliberately frozen (`plans/HZ-0A_Progress_Tracker.md`, "Both Stage 2
architectures are now complete") -- Stage 2 finished, was full-holdout
evaluated, and B6/B7/B8 all have real integration work built against that
exact checkpoint. Nothing below touches any of that. This document exists
so the idea and the real, honest evidence gathered about it aren't lost,
in case it's worth revisiting later with the confounds section 8 names
addressed.

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

## 6b. Real language-modeling comparison (2026-07-30, same date)

`docs/restart/hz0a_gdn3_tiny_lm_comparison_results.md`: a tiny (dim=64,
4-layer), same-data, same-seed, real-LM-loss comparison. Found and fixed
a real bug first (unnormalized learned keys blow up the delta-rule
projection -- the synthetic benchmark's hand-set unit-norm keys had
hidden this). After the fix: **statistically tied final validation loss**
(candidate -0.0087, within noise) -- the synthetic overwrite/interference
advantage does not show up as a general perplexity win at this scale.
Does not rule out the hypothesis (could be scale- or task-dependent, see
that doc's recommendation for the more targeted next experiment), but
meaningfully weakens the case for "obviously retrain HZ-0A with this."

## 6c. The decisive test (2026-07-30, same date): associative recall with overwrite

`docs/restart/hz0a_gdn3_associative_recall_results.md`: the direct,
on-point test -- a multi-query associative-recall-with-reassignment task
(the standard benchmark family delta-net papers themselves use), trained
from scratch, same fair-comparison discipline. **First attempt found the
candidate slightly behind (32.4% vs 30.5%) -- but that mixer had fewer
parameters than GDN2 and only 800 training steps, both real, disclosed
confounds. Corrected (parameter-matched, 3000 steps): candidate 32.4% vs.
current 29.7%, a +2.73 point win for the candidate**, both far above the
12.5% chance floor. The correction reversed the finding -- a real example
of a confounded test giving the wrong answer, caught by naming and then
actually fixing the confounds rather than trusting the first result. See
section 8 for the overall, updated verdict.

## 8. Final verdict (2026-07-30, updated after multi-seed replication, same day)

Four real tests were run, escalating in rigor -- the associative-recall
test specifically went through 3 rounds (confounded, corrected
single-seed, then multi-seed) because each earlier round's result did not
survive the next level of scrutiny:

1. **Isolated mechanism** (no training, synthetic keys/values): real,
   substantial, clearly demonstrated advantage for the delta projection.
   This part holds -- never contradicted by anything that followed.
2. **Real generic language-modeling loss** (trained, real corpus, both
   mixer versions, MLX and torch): tied every time -- no downside on
   general text, consistently.
3. **Real associative-recall-with-overwrite task, single seed**
   (confounded attempt: candidate behind; corrected attempt, parameter-
   matched: candidate ahead by +2.73 points).
4. **Same task, 3 additional seeds via the torch port**
   (`docs/restart/hz0a_gdn3_associative_recall_results.md`'s multi-seed
   section): candidate behind on average (-1.30 points), winning only 1
   of 3 seeds. Combined across all 4 seeds run: 2 losses, 1 near-tie, 1
   win -- statistically indistinguishable from no effect.

**Final recommendation: do not pursue an HZ-0A retrain.** The mechanism-
level advantage (1) is real and stands on its own -- a genuine, correctly
identified architectural gap relative to true delta-net. It simply was
not shown, after real and repeated attempts including catching two of
this investigation's own testing mistakes, to matter for trained model
capability at the scale tested. That is the honest, complete answer, not
an abandoned or rushed one -- the single favorable seed was investigated
rather than trusted, and turned out to be noise.

## 9. Honest scope of this document

- Nothing here has been implemented in the real HZ-0A, benchmarked at
  HZ-0A's real scale, or run through a real kernel.
- Adopting section 4 for real would require: a new native Metal kernel
  (the existing one implements the current, simpler recurrence), a new
  PyTorch reference implementation, re-deriving and re-verifying backward
  passes (the same rigor HZ-0A's own A3/A8 phases went through for the
  current recurrence), and a full retrain -- not a patch to the frozen
  Stage 2 checkpoint. Given section 8's verdict, none of this is currently
  recommended.
- This document's value, as it stands, is the trail of real evidence: a
  credible architectural gap was found (section 2-3), tested rigorously
  at small scale in three escalating ways (sections 6, 6b, 6c), and the
  honest conclusion is that it doesn't currently justify the cost of
  pursuing further -- a complete, disclosed negative result, not an
  abandoned thread.
