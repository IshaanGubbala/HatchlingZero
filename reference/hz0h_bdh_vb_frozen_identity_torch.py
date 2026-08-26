"""BDHVB with P/O frozen at (truncated) identity throughout training --
the "crux" test from the VB identity-mystery investigation, plus Tier 1
Phase VB-A of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md
("does a stable fixed compressed basis suffice, or must it adapt?").

At d_state == n_embd, P=O=I exactly and NEVER updated
(requires_grad=False) makes the model's forward math IDENTICAL to exact
BDH at every training step, not just at initialization (unlike
BDHVBIdentityInit, where P/O are trainable and can drift). Real result,
2026-08-24: val_loss=1.8412 vs exact BDH's 1.8585 -- ruled out a bug or
capacity limit, isolating VB's quality cliff to trainable P/O's own
gradient dynamics.

At d_state < n_embd, true identity isn't possible (non-square) -- uses
the same truncated-identity init as BDHVBIdentityInit (P selects the
first d_state coordinates, O scatters back, zero elsewhere), but frozen
rather than trainable. This IS real, permanent information loss (a fixed
coordinate projection), unlike the d_state==n_embd case -- comparing this
against the warm-start result (freeze early, then unfreeze) at the same
width is Phase VB-A's gate: if permanently frozen lands within ~0.02-0.03
of warm-start's val_loss, a stable fixed compressed basis is sufficient
and the simpler frozen-forever recipe wins; if materially worse, the
basis must eventually adapt.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig


class BDHVBFrozenIdentity(BDHVB):
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
        self.P.requires_grad_(False)
        self.O.requires_grad_(False)
