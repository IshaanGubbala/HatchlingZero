"""GDN-3 candidate mixer: an `nn.Module` matching `reference/hz0a_mlx_model.py`'s
`GDN2` interface exactly (same `__call__(x, state) -> (output, next_state)`
signature, same `[bsz, heads, head_dim, head_dim]` state shape), but using
the delta-rule projection recurrence
(`reference/hz0a_gdn3_candidate_recurrence.py`'s `step_delta_projection_plus_decay`)
instead of GDN2's plain gated update. Built so a tiny model can swap this
in for GDN2 and be compared head-to-head on real language-modeling loss --
`reference/hz0a_mlx_model.py` itself is not touched.

Needs only 2 learned gates (decay `alpha`, write-strength `beta`) instead
of GDN2's 3 (decay, erase, write) -- the delta rule's `(I - beta*k*k^T)`
projection does the work GDN2's separate erase gate approximates.

`in_proj` is `dim -> 6*dim` here, matching GDN2's own parameter count
exactly -- one slot (`_unused_padding`) is computed but never used in the
recurrence, kept ONLY for parameter-count parity so a comparison against
GDN2 isn't confounded by one model simply having more learnable capacity
than the other (an earlier version used `dim -> 5*dim`, a real, disclosed
handicap flagged in `docs/restart/hz0a_gdn3_associative_recall_results.md`
as a possible confound in that result -- this fixes it explicitly rather
than leaving it as a footnote).
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class GDN3CandidateMixer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.in_proj = nn.Linear(dim, 6 * dim)
        self.out = nn.Linear(dim, dim)
        # decay(alpha) biased toward ~1 (retain by default) and beta biased
        # toward a small write strength (~0.01) -- both match GDN2's own
        # init convention of starting conservative (mostly retain, barely
        # write) rather than aggressive, which is what GDN2's own decay/
        # erase/write bias init already does. The padding slot's bias
        # doesn't matter since its output is never used.
        self.in_proj.bias = mx.concatenate([mx.zeros((3 * dim,)), mx.full((dim,), 4.59512), mx.full((dim,), -4.59512), mx.zeros((dim,))])

    def __call__(self, x, state=None):
        bsz, steps, _ = x.shape
        q, k, v, decay_logit, beta_logit, _unused_padding = mx.split(self.in_proj(x).reshape(bsz, steps, 6, self.heads, self.head_dim), 6, axis=2)
        q, k, v, decay_logit, beta_logit = (mx.squeeze(item, axis=2) for item in (q, k, v, decay_logit, beta_logit))
        if state is None:
            state = mx.zeros((bsz, self.heads, self.head_dim, self.head_dim), dtype=x.dtype)
        decay = mx.sigmoid(decay_logit)
        beta = mx.sigmoid(beta_logit)
        # The (I - beta*k*k^T) projection is only a proper (non-expansive)
        # projection when k is unit-norm -- k*k^T then has eigenvalue
        # exactly 1 along k's own direction. An UNNORMALIZED learned k
        # (unlike the synthetic benchmark's hand-set unit/one-hot keys)
        # can have ||k||^2 >> 1, making (1 - beta*||k||^2) go negative and
        # the recurrence unstable (verified: caused immediate NaN before
        # this normalization was added). Real delta-net implementations
        # normalize k for exactly this reason.
        k = k / (mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True)) + 1e-6)
        outputs = []
        for t in range(steps):
            decayed = decay[:, t, :, None, :] * state
            # old_retrieved: what querying k_t against the (already-decayed)
            # state currently returns -- [bsz, heads, head_dim]
            old_retrieved = mx.sum(decayed * k[:, t, :, None, :], axis=-1)
            correction = beta[:, t, :, :, None] * (v[:, t] - old_retrieved)[:, :, :, None] * k[:, t, :, None, :]
            state = decayed + correction
            outputs.append(mx.sum(state * q[:, t, :, None, :], axis=-1))
        return self.out(mx.stack(outputs, axis=1).reshape(bsz, steps, self.dim)), state
