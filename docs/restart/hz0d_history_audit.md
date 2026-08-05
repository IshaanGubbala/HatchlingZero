# HZ-0D D0: History Audit

Date: 2026-08-04. Real `git log --all` sweep (`-i --grep="fast.weight\|fast weight\|HZ-0D\|hz0d\|low.rank adapt\|snapshot.*rollback"`)
for any prior fast-weight implementation, matching C0's own discipline
(never assume a clean slate without checking).

## Real prior work found: "Phase 16" (July 27, 2026, old `src/hz0/` tree)

Six commits (`4966807`, `9a52a59`, `2e6ecbe`, `b5daab1`, `4d93933`,
`fee890b`) implemented what was THEN called "HZ-0C session-local fast
weights" -- the OLD naming convention, before this project's restart
relocated "fast weights" to HZ-0D and redefined HZ-0C as "surprise-
triggered anchor attention"
(`docs/restart/hz0c_history_audit.md` already flagged this same
relocation from the other direction). Deleted from the active tree in
the `47b79a1` "Restart repo layout" commit; preserved under
`archive/src/hz0/fast_weights/` (6 files, ~1,100 lines), read in full
for this audit, not just the commit messages.

## What was real and reusable

- **`FastWeightLinear`**: `y = x @ (W_base + W_fast) + (b_base +
  b_fast)`, `W_base` frozen, `W_fast` session-local, `reset_fast_weights()`
  zeros it. The additive-delta STRUCTURE is directly compatible with
  HZ-0D's own plan (`W_effective = W_base + A_fast @ B_fast`).
- **Snapshot/rollback/session design**: `ProductionFastWeightSession`'s
  `checkpoint(name)`/`rollback(name)` (dict-of-array snapshots, restore
  by name) and `start_session()`/`end_session()` (reset fast state to
  zero) are a real, reasonable, directly reusable DESIGN PATTERN --
  independent of whether the update mechanism underneath them worked.
- **Production safeguards attempted**: gradient-norm clipping, fast-
  weight-norm clipping, NaN/inf detection. Reasonable safety mechanisms
  in principle, real code, with unit tests that pass -- see the honest
  caveat below on what those tests actually validated.
- **Placement**: fast weights were applied to attention Q/K/V
  projections (`FastWeightAttentionBlock`, wrapping the old model's
  `AttentionBlock`) -- consistent with D1's own "narrow locations"
  guidance (upper MLP, controllers, anchor-attention output
  projections), though the OLD version replaced the QKV projection with
  a FULL dense `[dim, 3*dim]` delta, not a low-rank one (see below).

## The critical, disclosure-worthy defect: the "gradient-based" update was never gradient descent

`meta_learner.py::GradientBasedMetaLearner.adapt_batch` (the ONLY
adaptation mechanism actually exercised by Phase 16a/b/c) does this,
verbatim:

```python
for layer in fast_weight_layers:
    if hasattr(layer, 'fast_weight'):
        eps = 1e-4
        perturbation = mx.random.normal(layer.fast_weight.shape) * eps
        # Two-point gradient estimate
        logits_pert = model(context_inputs)
        loss_pert = loss_fn(logits_pert, context_targets)
        # Update in direction that reduced loss
        layer.fast_weight = layer.fast_weight - self.learning_rate * perturbation * loss_val
```

`loss_pert` is computed and then **never used**. The update is a FRESH
random direction (`perturbation`) scaled by the PRE-perturbation loss
magnitude (`loss_val`), not by any measured effect of that direction on
the loss. This is not gradient descent, not a valid finite-difference/
SPSA estimator (which would need to compare `loss_pert` against
`loss_val` and use that DIFFERENCE, or at minimum reuse the same
perturbation whose effect was measured), and not a Hebbian rule either
-- it is unbiased random noise injection, with zero expected loss-
reduction property. `FastWeightLinear.update_fast_weights` (the
correctly-shaped, clip-aware update API) is NEVER called from this path
at all; it is bypassed entirely in favor of the inline broken update
above.

**This was self-admitted, in the same phase's own commit message**,
Phase 16c: "Current simple gradient approximation insufficient for
learning (proper backprop needed for real gains)." That note is
accurate -- more accurate than it might have first read, since the
mechanism was not merely a WEAK gradient approximation, it computed no
gradient information at all. Phase 16a's claimed "1.6% loss improvement"
and "2.1% accuracy improvement" were produced by this mechanism and
should not be trusted as evidence that fast weights learned anything;
they are far more likely measurement noise or a placebo effect from
adding any zero-mean perturbation to a nearly-linear local loss surface.

**Phase 16d's "production hardening" tested this defect's SYMPTOMS, not
its cause, and never closed the loop.** Its own test suite
(`test_production_hardening`) validates gradient clipping and weight-
norm clipping using HAND-CONSTRUCTED fake gradient dicts (e.g.
`{"fast_weight": mx.ones((64, 64)) * 100.0}`), passed directly to
`update_fast_weights` -- never once through the actual (broken)
`adapt_batch` path that Phase 16a-c used end to end. So "Phase 16
complete... production-ready... 100% test coverage" (`fee890b`'s own
words) tested infrastructure robustness in isolation, decoupled from
whether the adaptation mechanism it was guarding ever did anything
real. It did not.

## A real design difference to disclose, not just a defect

The old implementation used a FULL dense per-session weight delta
(`W_fast` shaped `[dim, 3*dim]` for a QKV projection at `dim=768` --
1.77M session-local floats per layer). HZ-0D's own plan specifies a
LOW-RANK delta (`A_fast @ B_fast`) specifically to bound session-state
memory and compute -- a real, deliberate departure from the old design,
not an oversight to reconcile. Worth stating explicitly so a future
reader does not assume the old dense-delta code is a drop-in reference.

## What this means for D1 (the contract)

1. **The update mechanism must be a REAL, verified gradient computation**
   (`mx.grad`/`mx.value_and_grad` against an actual loss, or a
   well-specified, honestly-validated Hebbian/analytic rule) -- not
   assumed correct because a plausible-sounding name and clipping
   infrastructure surround it. Any update mechanism chosen at D3 must be
   checked with a finite-difference or synthetic-signal sanity test
   BEFORE being trusted on a real task, matching this project's own
   standing discipline elsewhere (e.g. the CPU/GPU kernel parity tests,
   the ranking-loss sanity check in C7).
2. **Snapshot/rollback/session-reset infrastructure and clipping/NaN
   safeguards are legitimate, reusable design patterns** -- carry the
   PATTERN forward (checkpoint-by-name, reset-to-zero on session
   boundary, gradient- and weight-norm clipping, NaN/inf health checks),
   not the old code verbatim, and validate each piece against a REAL
   update mechanism this time, end to end, not against synthetic
   gradient dicts in isolation.
3. **Low-rank, not dense, deltas** -- per HZ-0D's own plan, a deliberate
   change from the old design, disclosed here so it reads as a decision,
   not a gap.
4. Any reused claim of "improvement" from the old Phase 16 numbers
   (1.6% loss, 2.1% accuracy) must be treated as unverified and
   re-measured from scratch against a real gradient-based mechanism --
   not cited as prior evidence that fast weights help.
