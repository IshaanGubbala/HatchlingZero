"""Tier 4 item 22 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md
(section 12, "Architectural Variant Only If Exact Methods Plateau" --
its own precondition is now satisfied: items 14/17/20 all showed exact
sparsity/certification failing to produce usable real-hardware speedup
or a working certificate).

Hierarchical region-gated BDH: a coarse group gate is part of the MODEL
DEFINITION, not a post-hoc predictor of a separate ReLU outcome:

    g = ReLU(x @ C)                      # (B, 1, T, R), R << nh*N -- cheap
    x_sparse[region r] = g[r] * ReLU(x @ E[region r])   # architecturally exact zero when g[r]=0

If g[r]=0, the entire region's x_sparse is exactly zero BY CONSTRUCTION
(not approximated, not predicted) -- eliminating the predictor-error
problem that made every certificate/template approach fail tonight
(items 12-15), because the gate is trained end-to-end as part of the
model rather than fit post-hoc to a frozen model's activation
statistics.

Regions are defined over the FLATTENED (nh, N) neuron space with the
SAME nh-outer/N-inner convention used throughout this session's
diagnostics (matches the decoder's own `(nh*N, D)` layout) -- `R` must
divide `nh*N` evenly, and for a clean per-head split R should also
divide evenly into N-per-head groups (this project's default R=64 at
production shape n_embd=2496/mult=16/n_head=8 gives N=4992,
region_size=624, exactly 8 regions per head).

Two variants, directly informed by tonight's own frozen-forever VB
result (which decisively beat freeze-then-unfreeze warm-start, which
itself beat trainable-from-step-0, at every width and seed tested):
`BDHHierarchicalRegionGated` (C trainable from step 0, the plan's
original "gradual warm-start" framing) and
`BDHHierarchicalRegionGatedFrozen` (C fixed at its random init, never
trained -- the direct, evidence-based analog of the VB lesson applied
to this new gate).
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from reference.hz0h_bdh_torch import BDH, BDHConfig


@dataclasses.dataclass
class BDHHierarchicalRegionGatedConfig(BDHConfig):
    num_regions: int = 64


class BDHHierarchicalRegionGated(BDH):
    def __init__(self, config: BDHHierarchicalRegionGatedConfig):
        super().__init__(config)
        D = config.n_embd
        nh = config.n_head
        N = D * config.mlp_internal_dim_multiplier // nh
        R = config.num_regions
        assert (nh * N) % R == 0, f"num_regions={R} must divide nh*N={nh*N} evenly"
        self.num_regions = R
        self.region_size = (nh * N) // R
        self.gate_proj = nn.Parameter(torch.zeros((D, R)).normal_(std=0.02))

    def _region_gate(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T, D). Returns (B, nh, T, N) -- the per-neuron gate
        value, broadcast from R region gates over region_size neurons
        each, reshaped into the SAME (nh, T, N) layout as x_sparse."""
        C = self.gate_proj
        B, _, T, D = x.shape
        nh = self.config.n_head
        N = D * self.config.mlp_internal_dim_multiplier // nh
        gate_latent = x @ C  # (B, 1, T, R)
        gate = F.relu(gate_latent)
        gate_expanded = gate.repeat_interleave(self.region_size, dim=-1)  # (B, 1, T, nh*N)
        return gate_expanded.squeeze(1).view(B, T, nh, N).permute(0, 2, 1, 3)  # (B, nh, T, N)

    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)

        for level in range(C.n_layer):
            x_latent = x @ self._w(self.encoder)
            raw_sparse = F.relu(x_latent)
            region_gate = self._region_gate(x)
            x_sparse = raw_sparse * region_gate  # (B, nh, T, N) -- architecturally exact zero where gate is zero

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            y_latent = yKV @ self._w(self.encoder_v)
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse

            xy_sparse = self.drop(xy_sparse)

            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self._w(self.decoder)
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class BDHHierarchicalRegionGatedFrozen(BDHHierarchicalRegionGated):
    def __init__(self, config: BDHHierarchicalRegionGatedConfig):
        super().__init__(config)
        self.gate_proj.requires_grad_(False)
