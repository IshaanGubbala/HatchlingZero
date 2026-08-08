"""HZ-0H H1: faithful PyTorch port of the official BDH-GPU model.

Ported directly from `github.com/pathwaycom/bdh`'s `bdh.py` (raw source
read directly, not a summarized description -- see
`docs/restart/hz0h_bdh_history_audit.md` and
`docs/restart/hz0h_bdh_component_map.md` for the full sourcing/
verification trail). This is an ISOLATED ORACLE: it does not touch, call,
or depend on any HZ-0A-G mechanism, and nothing in HZ's canonical
backbone depends on this file. Its only job is to be a faithful,
independently-testable reference for comparison, per H1's own mandate.

Two precise, real discrepancies between the paper's prose and the actual
code are preserved here deliberately, not "corrected" toward the paper's
simplified description (see the component map doc for the verification):

1. No softmax, no `K^T @ 1` normalization anywhere in attention -- raw
   `scores @ V`. The paper's abstract states a normalized-average formula;
   the real code has no such division.
2. The causal mask is STRICTLY lower-triangular (`diagonal=-1`) -- a
   position cannot attend to itself, only to strictly earlier positions.
   A standard `diagonal=0` causal mask would silently deviate from the
   official implementation.

Weights are genuinely shared across depth: `encoder`, `encoder_v`,
`decoder`, and the single `ln` module are each ONE set of parameters
reused every iteration of the layer loop, not per-layer instances --
confirmed directly against source, not assumed.
"""
from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def quantize(t: torch.Tensor, q: int = 2) -> torch.Tensor:
    return (t / q).floor() * q


def get_freqs(n: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    return 1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi)


class Attention(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh, D = config.n_head, config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.register_buffer("freqs", get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))

    @staticmethod
    def phases_cos_sin(phases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cos(phases), torch.sin(phases)

    def rope(self, phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = self.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()
        r_phases = (torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh, D = config.n_head, config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.embed = nn.Embedding(config.vocab_size, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)
        self.attn = Attention(config)

        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
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
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)
        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
