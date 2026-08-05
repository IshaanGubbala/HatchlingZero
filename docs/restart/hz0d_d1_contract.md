# HZ-0D D1: The Fast-Weight Contract

Date: 2026-08-04. Per the plan's own D1 text ("Specify which layers
receive fast weights, low-rank dimensions, update frequency and budget,
decay, clipping, normalization, snapshot, rollback, reset,
serialization, gradient flow, and maximum fast-state memory") and exit
gate ("every state tensor and lifecycle operation is documented"). Every
decision below is paired with real code
(`reference/hz0d_fast_weights.py`) and real tests
(`tests/reference/test_hz0d_fast_weights.py`), not left as prose alone
-- matching this project's own standing discipline, and directly
required by D0's finding that a prior "gradient-based" mechanism was
never checked for real correctness before being called production-ready.

## 1. Which layers receive fast weights

The anchor-attention OUTPUT projection (`out` in
`reference/hz0a_mlx_model.py::CausalAttention`, `nn.Linear(dim, dim)`)
at the SAME 6 layers HZ-0C already uses (`ATTENTION_INDICES = (4, 9, 14,
19, 24, 29)`). Reasoning:

- The plan's own D6 text names "upper MLP blocks, memory controllers, or
  anchor-attention output projections" as narrow starting locations, and
  explicitly says "avoid modifying the core GDN-2 update first."
- The old archived implementation (`docs/restart/hz0d_history_audit.md`)
  independently chose attention Q/K/V projections at the same kind of
  layer -- convergent evidence this is a reasonable place to start, even
  though that implementation's update mechanism was broken.
- The OUTPUT projection specifically (not Q/K/V) is chosen over the old
  implementation's choice because it is the single point where every
  attended value already converges before leaving the layer -- one
  `[dim, dim]` matrix to adapt per layer, not three, for a first,
  narrow, low-risk target. Q/K/V fast weights are real, disclosed future
  work, not attempted here.

## 2. Low-rank dimensions

`W_effective = W_base + A_fast @ B_fast`, `A_fast: [dim, rank]`,
`B_fast: [rank, dim]`, `rank = 16` (default, a config field, not
hardcoded) against `dim = 768`. `rank=16` is a conventional, clearly-
bounded LoRA-style starting rank (real precedent: typical LoRA
deployments use rank 4-64 at this hidden size); not tuned yet -- D2's
own isolated-simulator work is where a real adaptation-quality-vs-rank
tradeoff would be measured, not assumed here.

**Init is deliberately ASYMMETRIC, and this was caught by testing, not
designed correctly up front**: the first version of `init_fast_weights`
zero-initialized BOTH `A_fast` and `B_fast`. `test_real_gradient_step_strictly_reduces_loss`
failed immediately -- for `delta = A @ B`, `dL/dA = upstream @ B.T` and
`dL/dB = A.T @ upstream`, so if BOTH factors start at zero, BOTH
gradients are exactly zero (each one's formula multiplies by the OTHER,
still-zero factor), a dead saddle point no gradient step can ever leave.
Fixed to the standard LoRA convention -- `b_fast` zero, `a_fast` small
random noise (`init_scale=0.02`, `init_seed` for determinism) -- so the
PRODUCT is still exactly zero at session start (the inactive-reproduces-
base invariant holds, unchanged), but the gradient with respect to
`b_fast` is nonzero from the very first step. Left in this document as a
real example of D0's own lesson applied successfully: the update
mechanism was checked with a real test before being trusted, the test
caught a real defect, and the defect was fixed -- the opposite of the
prior implementation's history.

## 3. Update frequency and budget

`max_updates_per_session: int = 50` (config field). No update happens
automatically; a caller must explicitly invoke `update_fast_weights`.
This is a hard cap enforced by the config value existing and being
checked by callers (D2's simulator), not by the state/lifecycle module
itself refusing calls -- the module's own job is correctness of each
individual operation, not session-level policy.

## 4. Decay

`decay_fast_weights(state, decay_rate)`: `A_fast *= decay_rate`,
`B_fast *= decay_rate` (decaying both factors, not just one, so the
EFFECTIVE delta `A@B` decays roughly as `decay_rate^2` per call --
documented explicitly since it is easy to miscount). `decay_rate = 1.0`
is an exact no-op (tested). Reuses this project's own established
`forget_or_decay` pattern from HZ-0B (`reference/hz0b_memory_simulator.py`)
in spirit -- multiplicative confidence/state erosion -- not its code,
since the state shape is entirely different (dense low-rank factors, not
a slotted memory bank).

## 5. Clipping

Applied to the EFFECTIVE delta's Frobenius norm (`||A_fast @ B_fast||`),
not to `A_fast`/`B_fast` individually -- clipping the factors separately
does not bound the product's norm in an easy-to-reason-about way (the
product can still grow if the factors are clipped independently but
their product's alignment changes), while clipping the realized delta
directly gives an exact, verifiable bound on how much the effective
weight can move from the base weight. `max_delta_norm` is a config
field. Applied inside `update_fast_weights` after the gradient step,
before returning the new state (tested: the effective delta's norm never
exceeds the bound regardless of input gradient magnitude, matching D0's
requirement that clipping be validated against the REAL update path, not
a synthetic gradient dict in isolation).

## 6. Normalization

None applied beyond clipping, deliberately, for this first pass -- kept
simple until D2's simulator shows a real need. Documented here so it
reads as a decision, not an oversight.

## 7. Snapshot / rollback / reset

- `snapshot(state) -> dict`: named checkpoint (a plain dict of `mx.array`
  copies -- the caller owns naming/storage, matching the archived
  implementation's `checkpoint(name)`/`rollback(name)` pattern, carried
  forward as a design pattern per D0, not its code).
- `rollback(checkpoint) -> FastWeightState`: exact restore. Tested for
  BIT-IDENTICAL equality (`mx.array_equal`), not approximate closeness.
- `reset_fast_weights(config) -> FastWeightState`: returns to the exact
  same zero-initialized state `init_fast_weights` produces. Tested for
  bit-identical equality against a fresh `init_fast_weights` call.

## 8. Serialization

`serialize(state) -> dict` (JSON-safe: `.tolist()` on every array, plus
the update count) / `deserialize(data, config) -> FastWeightState`
(rebuilds `mx.array`s from the lists). Round-trip tested for exact
equality. Matches this project's own established checkpoint-dict
convention (`payload["arrays"]`-style, used throughout HZ-0A/B/C) rather
than a new, bespoke format.

## 9. Gradient flow

`A_fast`/`B_fast` must be differentiable via `mx.grad`; `W_base` is
NEVER part of the fast-weight gradient computation (the function
`apply_fast_linear` takes `base_weight` as a plain, non-parameter input
array -- there is no code path by which `mx.grad` over `(A_fast,
B_fast)` could touch it). Verified three ways, per D0's explicit
requirement that this NOT be assumed correct the way the old
implementation wrongly was:

1. A synthetic toy task where the analytic gradient's SIGN and
   magnitude are checked against what a real reduction in loss requires.
2. A finite-difference check: the analytic `mx.grad` gradient is
   compared directly against a numerical two-point finite-difference
   estimate of the SAME loss, and must agree to a small tolerance --
   the exact check the old mechanism never had and that would have
   caught its defect immediately.
3. An explicit real-gradient-step-reduces-loss test: one real update
   step, using the analytic gradient, is checked to strictly reduce a
   real loss value on a controlled example -- not assumed from the
   gradient's existence alone.

## 10. Maximum fast-state memory

Audited, not estimated: at `dim=768`, `rank=16`, 6 layers,
`A_fast` is `[6, 768, 16]` and `B_fast` is `[6, 16, 768]` -- `2 * 6 * 768
* 16 = 147,456` float32 values, `589,824` bytes (576 KiB) per session.
`fast_state_memory_bytes(config)` computes this directly from the config
fields (tested against this exact hand-computed number) so the bound is
never silently out of sync with the real state shapes.

## Exit gate check

"Every state tensor and lifecycle operation is documented": the
`FastWeightState`/`FastWeightConfig` dataclasses
(`reference/hz0d_fast_weights.py`) carry field-level docstrings for
every tensor, and `init`/`apply`/`update`/`decay`/`snapshot`/`rollback`/
`reset`/`serialize`/`deserialize` are each documented and independently
tested (`tests/reference/test_hz0d_fast_weights.py`). Met.
