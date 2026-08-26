"""BDHVB with P/O initialized at (truncated) identity, frozen for the
first `freeze_steps` training steps, then unfrozen to train normally.

Separates two competing explanations for VB's trainable-random-init
quality cliff. The frozen-identity crux experiment (this session) showed
val_loss=1.8412 with PERMANENTLY-frozen identity P/O at d_state==n_embd,
vs 1.8585 exact BDH baseline vs 2.0065 for trainable-random-init at the
same width -- ruling out capacity limits and implementation bugs, but
leaving open WHY the trainable variant collapses:

  1. "bad initial representation" -- random init starts far from a good
     solution and optimization never finds its way back. Predicts warm
     start (good init) STILL collapses once gradients start flowing,
     since it's an optimization-landscape problem independent of start.
  2. "gradient dynamics destabilize a good representation" -- identity
     is reachable/stable at init, but training signal actively pushes
     AWAY from it. Predicts warm start HOLDS close to exact-BDH quality
     even after unfreezing, since there's no gradient pressure to leave
     a solution that's already correct.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_vb_identity_init_torch import BDHVBIdentityInit
from reference.hz0h_bdh_vb_torch import BDHVBConfig


class BDHVBWarmStart(BDHVBIdentityInit):
    def __init__(self, config: BDHVBConfig, freeze_steps: int = 0):
        super().__init__(config)
        self.freeze_steps = freeze_steps
        self._unfrozen = False
        if freeze_steps > 0:
            self.P.requires_grad_(False)
            self.O.requires_grad_(False)

    def maybe_unfreeze(self, step: int) -> bool:
        """Call once per training step (0-indexed). Returns True the step it unfreezes."""
        if not self._unfrozen and step >= self.freeze_steps:
            self.P.requires_grad_(True)
            self.O.requires_grad_(True)
            self._unfrozen = True
            return True
        return False
