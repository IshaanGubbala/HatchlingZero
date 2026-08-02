# HZ-0C C1: Frozen Scaled Topology and Three Controlled Models

Date: 2026-08-02. Per the user's explicit decision (asked directly,
since `docs/restart/hz0c_recovered_requirements.md` flagged this as
an open call this session couldn't make unilaterally): **HZ-0C scales
from the CURRENT 301M topology now**, not a rescaled backbone -- the
GDN-2 fix / 1.5-3B scale-up decision (`plans/GDN-2_Fix.md`) is
tracked separately and does not block C1.

## Frozen topology parameters

`vocab_size=24576, dim=768, layers=31, heads=12, d_ff=2304` -- the
exact topology of the frozen HZ-0A checkpoint used throughout HZ-0B's
B11 evaluation
(`outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout`).
Context length and compute budget: not yet fixed to a specific number
-- inherited from whatever HZ-0A/B11 protocol is reused for the
isolated simulator (C3), real values to be set when C3's synthetic
tasks are built, since the plan explicitly allows the isolated
simulator to proceed before this is pinned down further.

Inherited HZ-0B memory placement: HZ-0B's memory read/write happens
per-position, independent of the attention/recurrent choice at each
layer (`reference/hz0b_b8_latent_write.py`'s `sequential_latent_write_and_read`
operates on the FINAL hidden state after all blocks, not inside any
one block) -- HZ-0C's anchor mechanism is a pre-final-layer
architectural choice and does not need to change HZ-0B's own
integration point. Real verification of this deferred to C6.

## The three controlled models -- built and audited

`reference/hz0c_surprise_trigger.py`, `tests/reference/test_hz0c_surprise_trigger.py`
(11/11 pass).

| Model | Description | Real params (audited) | Built how |
| --- | --- | --- | --- |
| 1 | Scaled recurrence, no anchors | **311,808,768** | `HZ0AMlxModel(..., attention_indices=())` -- already-existing infra, empty tuple |
| 2 | Scaled recurrence, fixed periodic anchors | **301,178,112** | `HZ0AMlxModel(..., attention_indices=(4,9,14,19,24,29))` -- ALREADY EXISTS, this IS the frozen HZ-0A checkpoint's own architecture (count cross-validated exactly against `plans/GDN-2_Fix.md`'s independently-cited `301,178,112`) |
| 3 | Scaled recurrence, surprise-triggered anchors | **325,982,988** | NEW: `HZ0CSurpriseTriggeredModel(..., anchor_indices=(4,9,14,19,24,29))` -- same 6 layer positions as model 2, each layer now doing BOTH recurrence AND conditional bounded attention |

All three produce finite output on real forward passes
(`test_c1_three_models_forward_pass_on_same_real_tokens`).

**Honest note on parameter matching**: model 3 is NOT parameter-matched
to model 1 or 2 (+14.2M vs. model 1, +24.8M vs. model 2) -- each
anchor-capable layer pays for a full recurrent GDN2 mixer AND a full
attention module, since the "conditional" part of "conditional anchor
attention" is a runtime data-dependent choice per position, not a
parameter-level choice per layer. This is not a gap to fix: the plan's
own Hard Constraints already anticipated this exactly --
"Claims must be compared at matched attention FLOPs, not only
parameter count" -- so C4/C9's real fairness axis for comparing model
3 against models 1/2 is attention FLOPs and average trigger rate, not
raw parameter count. Recorded here so no future comparison
accidentally treats these three as parameter-matched when the plan
itself says not to.

Also real, and interesting on its own: model 1 (pure recurrence) has
MORE parameters than model 2 (with 6 attention layers) at this
dim/heads config -- GDN2's `in_proj: dim -> 6*dim` (~3.5M params/layer)
costs more than `CausalAttention`'s `qkv: dim -> 3*dim` (~1.8M
params/layer) at heads=12, dim=768. Recurrent layers are not
inherently cheaper than attention layers here; the "expensive" part of
attention this whole project cares about is compute (O(seq^2)), not
parameter count.

## C2's surprise signal, implemented as part of this same module

`surprise_score()`: hidden-state delta norm per position
(`||x_t - x_{t-1}||`), C2's simplest candidate signal, chosen because
it needs zero additional learned parameters beyond a tiny calibration
scale/bias (`trigger_decision()`), works at real inference time (no
teacher-forced next-token access needed, unlike the token-loss-proxy
candidate), and needs no HZ-0B integration (unlike the memory-read-
uncertainty candidate, deferred to C6). `trigger_decision()` supports
both a soft (sigmoid) and hard (STE, matching `reference/hz0b_b8_latent_write.py`'s
established pattern) trigger, real and tested (position 0 always
scores 0, a constant hidden state scores ~0, a genuinely changing one
scores meaningfully positive).

`masked_anchor_attention()`: full causal attention restricted to
triggered query/key positions via masking -- correctness-first
reference (matches this project's established "slow reference before
optimized kernel" discipline, same as HZ-0B's B2 simulator before
PMetal). Real, tested: a triggered query's output is verified to be
COMPLETELY UNCHANGED when a non-triggered position's value is
perturbed by +100 (proves the masking genuinely excludes non-triggered
keys, not just down-weights them), and non-triggered query positions
are verified to produce exactly zero output.

## C1 exit gate: met

"All models have audited parameter counts and comparable protocols" --
real counts for all three models above; comparable protocols (same
vocab/dim/layers/heads/d_ff, same anchor-layer positions for models
2/3) established, with the one real, disclosed parameter-count
asymmetry (model 3) explained rather than hidden.

## Not yet done (explicitly C2-C9's work, not this pass's scope)

- C2's own exit gate ("surprise correlates with controlled novelty or
  difficulty") -- `surprise_score()` exists and is unit-tested for
  basic sanity (zero on no change, positive on change) but has NOT
  been validated against actual novelty/difficulty-labeled data yet.
- C3's isolated trigger simulator (novelty points, topic shifts, etc.)
  -- not built.
- C4's fair-baseline comparisons (no/fixed/random/oracle anchors, full
  attention, equal-compute transformer) at matched attention FLOPs --
  not run.
- The bounded/optimized (non-full-O(seq^2)) anchor-attention kernel --
  C8's job, this pass's `masked_anchor_attention` is the reference it
  must match, not the final implementation.
