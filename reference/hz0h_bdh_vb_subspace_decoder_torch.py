"""Compound architecture: VB frozen-identity value bottleneck (Tier 1
items 4/9/10, `reference/hz0h_bdh_vb_frozen_identity_torch.py` -- P/O
permanently frozen at truncated identity, d_state=624/n_embd=2496 (75%
state compression), val_loss=1.7999/1.8014 across 2 seeds, beats exact
BDH baseline 1.8585) COMBINED with the SVD-warmstarted subspace decoder
(Tier 4 item 23, `reference/hz0h_bdh_subspace_decoder_torch.py` -- decoder
factored to rank 64, SVD-warmstarted from a trained dense checkpoint,
val_loss=1.7972/1.7970 across 2 seeds, also beats baseline). Both pieces
were independently validated wins with DIFFERENT mechanisms (a frozen
structural constraint on the recurrent state vs a warm-started learned
constraint on the decoder) -- section 18 of
plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md's
"Target Architecture" explicitly names combining a compressed recurrent
state with other execution-side wins as the long-term direction. This
is the first real test of whether they compose (additive quality
improvement, roughly the sum of `n/d` compressions/speedups) or interact
(compounding degradation, since both touch different parts of the same
recurrent round and neither was validated in the other's presence).
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig


@dataclasses.dataclass
class BDHVBSubspaceDecoderConfig(BDHVBConfig):
    subspace_rank: int = 64


class BDHVBSubspaceDecoder(BDHVB):
    def __init__(self, config: BDHVBSubspaceDecoderConfig):
        super().__init__(config)
        D = config.n_embd
        nh = config.n_head
        N = D * config.mlp_internal_dim_multiplier // nh
        r = config.subspace_rank
        d_state = config.d_state

        # VB half: P/O frozen at (truncated) identity, never trained --
        # same init as BDHVBFrozenIdentity.
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

        # Subspace half: decoder factored to rank r (random init here --
        # SVD-warmstart, if used, is applied externally after construction,
        # same pattern as scripts/hz0h_bdh_subspace_decoder_warmstart_quality_check.py).
        del self.decoder
        self.decoder_up = nn.Parameter(torch.zeros((nh * N, r)).normal_(std=0.02))
        self.decoder_down = nn.Parameter(torch.zeros((r, D)).normal_(std=0.02))

    def _w(self, name: str) -> torch.Tensor:
        """Effective-weight hook; subclasses may supply training-only adapters."""
        return getattr(self, name)

    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)

        for _level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)

            v_bottleneck = x @ self.P
            yKV_bottleneck = self.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck)
            yKV = yKV_bottleneck @ self.O
            yKV = self.ln(yKV)

            y_latent = yKV @ self._w("encoder_v")
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            # Real Phase B fix, 2026-08-26: see hz0h_bdh_vb_subspace_decoder_stream_torch.py's
            # matching comment -- avoids a non-contiguous-tensor materialization
            # that dominates cost at batch>1 (verified mathematically identical).
            alpha = torch.matmul(xy_sparse, self._w("decoder_up").view(nh, N, -1)).sum(dim=1, keepdim=True)
            yMLP = alpha @ self._w("decoder_down")
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
