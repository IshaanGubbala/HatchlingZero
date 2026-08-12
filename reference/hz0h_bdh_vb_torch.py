"""HZ-BDH-VB: Value-Bottleneck BDH (Phase 2R-B, `plans/HZ Phase 2R State
Redesign Plan.md`).

Real, explicit divergence from upstream BDH-GPU -- NOT presented as a
kernel optimization or a faithful port. Motivated directly by
`docs/restart/hz0h_phase2_streaming_state_size_results.md`'s finding
that the exact BDH state (`reference/hz0h_bdh_torch.py`'s
`bdh_stream_chunk`, shape `(B, n_head, N, D)` per layer) is already
1.95x-3.31x the model's own weight bytes -- too large for a 30%
reduction (the original Phase 3 gate) to matter.

The idea: BDH's per-layer streaming state accumulates
`S_t = S_{t-1} + K_t^T V_t` where `V_t = x_t` (this layer's own D-wide
residual-stream input) -- the state's LAST dimension is D purely
because V is D-wide, not because the state needs D dimensions of real
information. Project V down to a small `d_state` before it enters the
state (`P: D -> d_state`), and back up to D when reading
(`O: d_state -> D`):

    S_t = S_{t-1} + K_t^T P(V_t)          # state shape (B, nh, N, d_state)
    y_t = O(Q_t @ S_{t-1})                # read, then project back to D

`P`/`O` are shared/tied across depth (`nn.Parameter`, same convention as
BDH's own `encoder`/`encoder_v`/`decoder`), not per-layer -- consistent
with BDH's own tied-weight philosophy, though this is an explicit
project choice (2R-C's "grouped depth state" is the separate, not-yet-
built experiment for per-group specialization).

Reuses `reference/hz0h_bdh_torch.py`'s `Attention` class UNCHANGED for
the RoPE/no-softmax-attention math -- `Attention.forward(Q, K, V)`
doesn't assume `V` is D-wide, so passing `V = P(x)` (d_state-wide)
instead of `V = x` (D-wide) works without touching that class at all.
This file is NOT part of the verbatim-upstream oracle
(`reference/hz0h_bdh_torch.py`) and does not modify it.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from reference.hz0h_bdh_torch import Attention, BDHConfig


@dataclasses.dataclass
class BDHVBConfig(BDHConfig):
    d_state: int = 0  # 0 -> defaults to n_embd (no compression, exact-BDH-equivalent width) in __post_init__

    def __post_init__(self) -> None:
        if self.d_state == 0:
            self.d_state = self.n_embd


class BDHVB(nn.Module):
    def __init__(self, config: BDHVBConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        d_state = config.d_state

        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))

        # The value-bottleneck projections -- the only real addition vs
        # reference/hz0h_bdh_torch.py's BDH class.
        self.P = nn.Parameter(torch.zeros((D, d_state)).normal_(std=0.02))
        self.O = nn.Parameter(torch.zeros((d_state, D)).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

            v_bottleneck = x @ self.P  # (B, 1, T, d_state)
            yKV_bottleneck = self.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck)  # (B, nh, T, d_state)
            yKV = yKV_bottleneck @ self.O  # (B, nh, T, D)
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

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def init_bdh_vb_states(model: BDHVB, batch_size: int, device=None, dtype=None) -> list[torch.Tensor]:
    c = model.config
    nh, D, d_state = c.n_head, c.n_embd, c.d_state
    N = D * c.mlp_internal_dim_multiplier // nh
    device = device if device is not None else model.encoder.device
    dtype = dtype if dtype is not None else model.encoder.dtype
    return [torch.zeros(batch_size, nh, N, d_state, device=device, dtype=dtype) for _ in range(c.n_layer)]


def bdh_vb_stream_chunk(
    model: BDHVB, states: list[torch.Tensor], idx_chunk: torch.Tensor, start_position: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Same real streaming derivation as reference/hz0h_bdh_torch.py's
    bdh_stream_chunk (intra-chunk term + cross-chunk state read), with
    V replaced by P(x) (d_state-wide) instead of x (D-wide), and O
    applied to the combined attention output before it re-enters the
    rest of the layer -- see the module docstring for the full
    derivation. State shape (B, nh, N, d_state) per layer, `d_state` <
    `D` being the entire point of this file."""
    c = model.config
    B, L = idx_chunk.shape
    D = c.n_embd
    nh = c.n_head
    N = D * c.mlp_internal_dim_multiplier // nh
    device = idx_chunk.device

    x = model.embed(idx_chunk).unsqueeze(1)
    x = model.ln(x)

    positions = torch.arange(start_position, start_position + L, device=device, dtype=model.attn.freqs.dtype).view(1, 1, L, 1)
    r_phases = positions * model.attn.freqs

    new_states = []
    for level in range(c.n_layer):
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        v_bottleneck = x @ model.P

        intra = (QR @ KR.mT).tril(diagonal=-1) @ v_bottleneck
        prefix_state = states[level]
        cross = QR @ prefix_state
        yKV_bottleneck = intra + cross
        yKV = yKV_bottleneck @ model.O
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, L, N * nh) @ model.decoder
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ v_bottleneck
        new_states.append(prefix_state + chunk_contribution)

    logits = x.view(B, L, D) @ model.lm_head
    return new_states, logits
