"""HZ mixed-depth efficiency pilot: cheap token-local blocks (C) mixed with
BDH's expensive full memory/attention rounds (B), per
plans/HZ_Mixed_Depth_Pilot_2026-09-15.md.

Reuses reference/hz0h_bdh_vb_subspace_decoder_gated_residual_torch.py's
math for B exactly (single-stream variant: the compound model's one
validated real win, per plans/HatchlingZero_Internal_Computation_Phase_2026-08-29.md's
"Compound BDH + single residual gate + exact addressing + depth curriculum").
Does not modify the faithful BDH oracle (reference/hz0h_bdh_torch.py) or
BDHVBSubspaceDecoder itself.

Scope note (real, not a placeholder): this file implements the FULL-SEQUENCE
(teacher-forced) forward path for all four arms -- everything needed for
plan section 2's tests 1/2/4 (logits/loss/gradient equivalence, one training
update, gradient isolation between B and C). It does NOT yet implement
chunked-streaming or token-by-token decode equivalence (test 3) or the "two
independent temporal state slots" requirement for D's two B occurrences --
those require reusing hz0h_bdh_torch.py's own streaming-state machinery,
which this file does not yet touch. Section 3's decode-speed benchmarking
depends on that being done first; it is real, separate, not-yet-started work,
not something to fake here.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_subspace_decoder_gated_residual_torch import (
    _gated_residual_checkpoint_iteration,
    add_gated_residual_stream,
)
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig


class CBlock(nn.Module):
    """Cheap token-local block, defined exactly per plan section 1:

        u = LayerNorm(x)
        x_next = x + sigmoid(g) * (silu(u @ W_up) @ W_down)

    No attention, synaptic state, convolution, or token-history access --
    deliberately cannot perform B's temporal/memory role. Non-affine
    LayerNorm (eps=1e-5), both matrices Normal(0, 0.02), no biases,
    g = logit(0.01) so sigmoid(g) = 0.01 at init (matches B's own g1/g2
    near-identity-at-init convention from the gated-residual reference).
    """

    def __init__(self, width: int, bottleneck: int = 128):
        super().__init__()
        self.ln = nn.LayerNorm(width, elementwise_affine=False, eps=1e-5)
        self.W_up = nn.Parameter(torch.zeros((width, bottleneck)).normal_(std=0.02))
        self.W_down = nn.Parameter(torch.zeros((bottleneck, width)).normal_(std=0.02))
        self.g = nn.Parameter(torch.tensor(math.log(0.01 / (1 - 0.01)), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.ln(x)
        return x + torch.sigmoid(self.g) * (F.silu(u @ self.W_up) @ self.W_down)


def _b_round(x, model, B, T, D, nh, N):
    return _gated_residual_checkpoint_iteration(x, model, B, T, D, nh, N)


class MixedDepthModel(nn.Module):
    """One model per schedule string ('B'/'C' tokens, spaces ignored --
    e.g. 'BBBBBBBB', 'BB', 'CCCB CCCB'). All B occurrences within one
    model instance share the SAME BDHVBSubspaceDecoder + single-stream
    gated-residual parameters (model.g1, decoder_up/down, etc.); all C
    occurrences share one CBlock instance. No per-position weights are
    created, matching the plan's explicit prohibition.
    """

    def __init__(self, config: BDHVBSubspaceDecoderConfig, schedule: str, checkpoint_b: bool = True,
                g1_init: float = 1.0, g1_fixed: bool = False):
        super().__init__()
        self.config = config
        self.schedule = schedule.replace(" ", "")
        self.checkpoint_b = checkpoint_b
        if not self.schedule or any(t not in "BC" for t in self.schedule):
            raise ValueError(f"schedule must be non-empty B/C characters, got {schedule!r}")
        self.decoder = BDHVBSubspaceDecoder(config)
        add_gated_residual_stream(self.decoder, single_stream=True, g1_init=g1_init, g1_fixed=g1_fixed)
        self.cblock = CBlock(config.n_embd) if "C" in self.schedule else None

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        C = self.config
        model = self.decoder
        Bsz, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = model.embed(idx).unsqueeze(1)
        x = model.ln(x)
        for token in self.schedule:
            if token == "B":
                if self.checkpoint_b:
                    x = torch.utils.checkpoint.checkpoint(
                        _b_round, x, model, Bsz, T, D, nh, N, use_reentrant=False)
                else:
                    x = _b_round(x, model, Bsz, T, D, nh, N)
            else:  # "C"
                x = self.cblock(x)

        logits = x.view(Bsz, T, D) @ model.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def param_count(self):
        """Trainable and frozen counts, separated -- plan section 1 requires
        reporting D's added parameter count rather than calling arms
        parameter-matched."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen}


# Arm definitions, exactly per plan section 1's table.
ARM_SCHEDULES = {
    "A": "BBBBBBBB",  # existing eight-round quality/cost control
    "B": "BBBBBBBB",  # same schedule as A; "exact execution simplifications" are a separate,
                       # not-yet-implemented systems change (frozen-identity P/O slicing, plan section 2) --
                       # this schedule alone does NOT constitute arm B yet, see module docstring.
    "C": "BB",         # simply removing rounds
    "D": "CCCBCCCB",   # cheap processing recovers useful depth, hypothesis
}


def build_arm(name: str, config: BDHVBSubspaceDecoderConfig, **kwargs) -> MixedDepthModel:
    if name not in ARM_SCHEDULES:
        raise ValueError(f"unknown arm {name!r}; expected one of {sorted(ARM_SCHEDULES)}")
    return MixedDepthModel(config, ARM_SCHEDULES[name], **kwargs)
