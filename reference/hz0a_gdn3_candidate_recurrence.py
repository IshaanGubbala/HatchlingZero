"""Candidate GDN-3 recurrence variants, for the controlled overwrite
benchmark in `scripts/hz0a_gdn3_overwrite_benchmark.py`.

Companion to `docs/restart/hz0a_gdn3_candidate_design.md`. NOT wired into
HZ-0A anywhere -- `reference/hz0a_mlx_model.py` is untouched. This module
exists to let 4 recurrence variants be compared head-to-head on a small,
synthetic, controlled task before anything about the frozen HZ-0A
architecture changes.

State orientation (verified directly against `reference/hz0a_mlx_model.py`'s
`GDN2.__call__`, per-head): `state` has shape `[Dv, Dk]` -- rows are
value-channels, columns are key-channels. Current HZ-0A's own decay/erase
gates are indexed along Dk (broadcast across all Dv rows uniformly); its
write gate is indexed along Dv (broadcast across all Dk columns) and
multiplies `v` elementwise before the outer product with `k`. All four
variants below share this exact orientation so comparisons are apples to
apples.

Variant 1, "current": HZ-0A's actual recurrence, verbatim.
  S_t = decay_t[None,:] * (1 - erase_t[None,:]) * S_{t-1}
        + outer(write_t * v_t, k_t)

Variant 2, "current_strong_erase": same shape of update, erase gate's
  pre-sigmoid logit boosted by a constant so it saturates toward 1 (full
  channel-wise forgetting every step) -- tests whether just forgetting
  MORE, without any key-targeted projection, can approximate what the
  delta rule buys structurally.

Variant 3, "delta_projection": the true delta rule (Kimi Linear/KDA's
  formulation, arXiv:2510.26692), no separate decay beyond what the
  projection itself provides (alpha_t == 1) -- isolates the projection
  term's own effect against the baseline.
  old_retrieved_t = S_{t-1} @ k_t                      # [Dv]
  S_t = S_{t-1} + beta_t * outer(v_t - old_retrieved_t, k_t)

Variant 4, "delta_projection_plus_decay": variant 3 plus HZ-0A-style
  per-channel decay applied first (the full GDN-3 candidate from the
  design doc).
  S_t = decay_t[None,:] * S_{t-1}
  old_retrieved_t = S_t @ k_t
  S_t = S_t + beta_t * outer(v_t - old_retrieved_t, k_t)

All four read AFTER updating state at the same step (matches HZ-0A's own
convention: `state = ...; outputs.append(sum(state*q,...))`), for a fair
comparison.
"""
from __future__ import annotations

import mlx.core as mx


def step_current(state: mx.array, q: mx.array, k: mx.array, v: mx.array, decay: mx.array, erase: mx.array, write: mx.array) -> tuple[mx.array, mx.array]:
    """HZ-0A's actual recurrence, verbatim (per-head, single step).
    All of q/k/v/decay/erase/write are [Dk] or [Dv] (equal-sized here,
    matching HZ-0A's single head_dim). state is [Dv, Dk]."""
    new_state = decay[None, :] * (1 - erase[None, :]) * state + mx.outer(write * v, k)
    output = new_state @ q
    return output, new_state


def step_current_strong_erase(state: mx.array, q: mx.array, k: mx.array, v: mx.array, decay: mx.array, erase_logit: mx.array, write: mx.array, *, boost: float = 4.0) -> tuple[mx.array, mx.array]:
    """Same update as `step_current`, but the erase gate's pre-sigmoid
    logit is boosted toward saturation -- an ablation, not a claim this
    is a good design, per the reviewer's own point: rule out "just forget
    harder" before crediting the projection term specifically."""
    erase = mx.sigmoid(erase_logit + boost)
    new_state = decay[None, :] * (1 - erase[None, :]) * state + mx.outer(write * v, k)
    output = new_state @ q
    return output, new_state


def _normalize_key(k: mx.array) -> mx.array:
    """The (I - beta*k*k^T) projection is only non-expansive when k is
    unit-norm (k*k^T then has eigenvalue exactly 1 along k's own
    direction) -- an unnormalized k can make (1 - beta*||k||^2) go
    negative and the recurrence unstable. This benchmark's own hand-set
    keys (Part A) are already unit/one-hot, so this is a no-op there, but
    a real learned k (as in `reference/hz0a_gdn3_candidate_mixer.py`) is
    not naturally unit-norm -- verified directly: omitting this caused
    immediate NaN in the tiny-LM comparison. Normalizing inside these
    functions makes them safe by default rather than relying on every
    caller to remember."""
    return k / (mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True)) + 1e-6)


def step_delta_projection(state: mx.array, q: mx.array, k: mx.array, v: mx.array, beta: mx.array) -> tuple[mx.array, mx.array]:
    """True delta rule, no separate decay (alpha == 1). `beta` is a
    scalar (per-head) write-strength gate, distinct from HZ-0A's
    per-channel `write` gate."""
    k = _normalize_key(k)
    old_retrieved = state @ k
    new_state = state + beta * mx.outer(v - old_retrieved, k)
    output = new_state @ q
    return output, new_state


def step_delta_projection_plus_decay(state: mx.array, q: mx.array, k: mx.array, v: mx.array, decay: mx.array, beta: mx.array) -> tuple[mx.array, mx.array]:
    """The full GDN-3 candidate: HZ-0A-style per-channel decay, THEN the
    delta-rule projection/correction."""
    k = _normalize_key(k)
    decayed = decay[None, :] * state
    old_retrieved = decayed @ k
    new_state = decayed + beta * mx.outer(v - old_retrieved, k)
    output = new_state @ q
    return output, new_state
