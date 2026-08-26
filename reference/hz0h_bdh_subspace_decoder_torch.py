"""Tier 4 item 23 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md
(section 13, Subspace BDH): the real next step after the required
diagnostic (scripts/hz0h_bdh_subspace_gate_reconstruction_diagnostic.py)
found that a rank-16 approximation of the last-round gate `g_t`
preserves 98.8% of the model's real top-1 predictions, and rank-256
preserves 99.5% -- a genuine positive signal, unlike most of tonight's
other Tier 3/4 findings.

This factors the DECODER matrix specifically -- `nh*N=39936 -> D=2496`,
~99.7M params, roughly a third of the whole 300M-param model -- into a
rank-r bottleneck: `decoder ~= decoder_up @ decoder_down`, shapes
`(nh*N, r)` and `(r, D)`. Unlike the exact-skip kernels that failed
tonight (items 17/20, both real gather/scatter approaches that lost
badly on wall-clock time despite passing their oracle ceilings), this is
just two smaller DENSE matmuls -- no gather, no scatter, no irregular
indexing at all -- so it structurally avoids the specific overhead
pattern that killed those attempts. At r=16 this is a ~147x parameter
reduction for the decoder specifically (39936*16 + 16*2496 ~= 679K vs
99.7M) and a real ~156x FLOP reduction for that matmul (dominated by the
`nh*N*r` term, not the small `r*D` term).
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from reference.hz0h_bdh_torch import BDH, BDHConfig


@dataclasses.dataclass
class BDHSubspaceDecoderConfig(BDHConfig):
    subspace_rank: int = 64


class BDHSubspaceDecoder(BDH):
    def __init__(self, config: BDHSubspaceDecoderConfig):
        super().__init__(config)
        D = config.n_embd
        nh = config.n_head
        N = D * config.mlp_internal_dim_multiplier // nh
        r = config.subspace_rank
        # Replace the dense decoder (nh*N, D) with a real rank-r factorization.
        del self.decoder
        self.decoder_up = nn.Parameter(torch.zeros((nh * N, r)).normal_(std=0.02))
        self.decoder_down = nn.Parameter(torch.zeros((r, D)).normal_(std=0.02))

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
            x_sparse = F.relu(x_latent)

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            y_latent = yKV @ self._w(self.encoder_v)
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            # Real Phase B fix, 2026-08-26: see
            # reference/hz0h_bdh_vb_subspace_decoder_stream_torch.py's matching
            # comment -- avoids a non-contiguous-tensor materialization that
            # dominates cost at batch>1 (verified mathematically identical).
            alpha = torch.matmul(xy_sparse, self._w(self.decoder_up).view(nh, N, -1)).sum(dim=1, keepdim=True)  # (B, 1, T, r) -- the real, learned low-rank bottleneck
            yMLP = alpha @ self._w(self.decoder_down)  # (B, 1, T, D)
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
