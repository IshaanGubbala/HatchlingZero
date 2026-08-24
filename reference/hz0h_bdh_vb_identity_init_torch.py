"""BDHVB with P/O initialized near-identity instead of random noise.

Direct test of the hypothesis from the VB width frontier's discriminating
experiment (d_state=2496 landing at 2.0065, nearly identical to the most
compressed variant, nowhere near exact BDH's 1.8585): the quality cliff
is an artifact of P/O starting as random N(0, 0.02) noise and not
converging in 5M tokens, not information loss through the bottleneck.

At d_state == n_embd, P and O can be initialized to the exact identity
matrix -- O(P(v)) = v exactly at step 0, so this configuration starts
IDENTICAL to exact BDH and only diverges as training teaches the model
something else is better. If quality still lands near the VB cluster
(~2.0) instead of near exact BDH (~1.86), that would rule out the
init-artifact hypothesis too and point back toward something more
structural about having a separate P/O pathway at all.

For d_state < n_embd, true identity isn't possible (non-square) -- uses
a truncated-identity init instead (P selects the first d_state
coordinates, O scatters back to those same coordinates, zero elsewhere).
This is real information loss at init (a coordinate projection, not a
random one), but zero LEARNED distortion -- a principled starting point,
not a fix for compression itself.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig


class BDHVBIdentityInit(BDHVB):
    def __init__(self, config: BDHVBConfig):
        super().__init__(config)
        D = config.n_embd
        d_state = config.d_state
        with torch.no_grad():
            if d_state == D:
                self.P.copy_(torch.eye(D, dtype=self.P.dtype))
                self.O.copy_(torch.eye(D, dtype=self.O.dtype))
            else:
                k = min(d_state, D)
                p_init = torch.zeros(D, d_state, dtype=self.P.dtype)
                p_init[:k, :k] = torch.eye(k, dtype=self.P.dtype)
                o_init = torch.zeros(d_state, D, dtype=self.O.dtype)
                o_init[:k, :k] = torch.eye(k, dtype=self.O.dtype)
                self.P.copy_(p_init)
                self.O.copy_(o_init)
