"""BDHVB with P/O frozen at exact identity throughout training -- the
"crux" test from the VB identity-mystery investigation.

If P=O=I and NEVER updated (requires_grad=False), the model's forward
math is IDENTICAL to exact BDH at every single training step, not just
at initialization (unlike BDHVBIdentityInit, where P/O are trainable and
can drift). If this frozen-identity model's final validation loss still
differs meaningfully from exact BDH's, that reveals a real bug or
implementation mismatch somewhere in BDHVB's forward, not an
optimization-difficulty story -- the two computations should be
mathematically identical throughout the entire run.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig


class BDHVBFrozenIdentity(BDHVB):
    def __init__(self, config: BDHVBConfig):
        super().__init__(config)
        D = config.n_embd
        d_state = config.d_state
        if d_state != D:
            raise ValueError("BDHVBFrozenIdentity requires d_state == n_embd for exact identity")
        with torch.no_grad():
            self.P.copy_(torch.eye(D, dtype=self.P.dtype))
            self.O.copy_(torch.eye(D, dtype=self.O.dtype))
        self.P.requires_grad_(False)
        self.O.requires_grad_(False)
