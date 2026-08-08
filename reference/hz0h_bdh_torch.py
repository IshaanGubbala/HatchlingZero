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
    ternary: bool = False  # HZ-0H T-lane: see docs/restart/hz0h_ternary_training_design.md


def quantize(t: torch.Tensor, q: int = 2) -> torch.Tensor:
    return (t / q).floor() * q


def _ternary_ste(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """BitNet b1.58-style absmean ternary quantization with a
    straight-through estimator -- same formula as
    `reference/hz0a_torch_model.py`'s `_ste_round_clip`, duplicated (not
    imported) rather than shared, to preserve this file's own "isolated
    oracle" contract (module docstring: "does not touch, call, or depend on
    any HZ-0A-G mechanism"). See
    `docs/restart/hz0h_ternary_training_design.md` for the T0 contract this
    implements. `w` stays full-precision as the actual `nn.Parameter`; only
    the VALUE USED in each forward call is re-quantized to
    `{-1, 0, 1} * gamma`, gradient is identity w.r.t. `w` via STE.
    """
    gamma = w.detach().abs().mean().clamp_min(eps)
    quantized = (w / gamma).round().clamp(-1, 1) * gamma
    return w + (quantized - w).detach()


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

    def _w(self, param: torch.Tensor) -> torch.Tensor:
        """Returns `param` as-is, or its ternary-quantized value (STE) if
        `config.ternary` -- applied only to `encoder`/`encoder_v`/`decoder`
        per the T0 contract (`embed`/`lm_head`/`ln` stay full precision,
        matching this project's existing BitNet convention)."""
        return _ternary_ste(param) if self.config.ternary else param

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)
        for _level in range(C.n_layer):
            x_latent = x @ self._w(self.encoder)
            x_sparse = F.relu(x_latent)
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


# --- H2: streaming/chunked state equivalence ---------------------------------
#
# BDH-GPU's attention has no softmax and a STRICTLY causal mask (see the
# module docstring's discrepancy #1/#2): `scores = (QR @ KR^T).tril(-1);
# out = scores @ V`. Because there's no softmax normalization, this is
# secretly linear attention: for position t, `y_t = sum_{s<t} <QR_t, KR_s> V_s
# = QR_t @ S_t` where `S_t = sum_{s<t} KR_s (x) V_s` is a running
# outer-product state of shape (nh, N, D) per layer. That means BDH-GPU has
# an exact (not approximate) O(1)-state streaming/chunked form, the same
# "chunked linear attention" trick used for reference/hz0a_torch_model.py's
# GDN-2 mixer -- this is what plans/HZ-0H_BDH_Reconciliation_Plan.md's H2
# ("prove full-sequence, one-token, and arbitrary chunked streaming agree")
# asks to prove.
#
# One subtlety: RoPE's phase depends on the ABSOLUTE position of a token in
# the full sequence, not its position within a chunk -- `bdh_stream_chunk`
# takes an explicit `start_position` for this reason, so resuming a stream
# mid-sequence still uses the correct phases.
#
# Multi-layer note: each layer's `V = x` is that layer's OWN input (the
# residual stream value seen at this layer), and every layer collapses back
# to a single (B, 1, T, D) tensor via `decoder` before the next layer runs
# (the `nh` dim never survives past one layer's own attention block) -- so
# per-layer states are independent, each shaped (B, nh, N, D), and a
# strictly-in-order token/chunk traversal (finish all layers for token t
# before starting token t+1) has everything it needs at each layer: prior
# tokens have already been fully processed through every layer, including
# this one, by the time token t is processed.

def init_bdh_states(model: "BDH", batch_size: int, device=None, dtype=None) -> list[torch.Tensor]:
    """Fresh (all-zero) per-layer running states, one per `model.config.n_layer`,
    each shaped (batch_size, n_head, N, n_embd). Represents an empty prefix
    (no tokens streamed yet) -- pass this to start a new stream, or to reset
    one, since it captures no hidden information beyond its shape/dtype."""
    c = model.config
    nh, D = c.n_head, c.n_embd
    N = D * c.mlp_internal_dim_multiplier // nh
    device = device if device is not None else model.encoder.device
    dtype = dtype if dtype is not None else model.encoder.dtype
    return [torch.zeros(batch_size, nh, N, D, device=device, dtype=dtype) for _ in range(c.n_layer)]


def bdh_stream_chunk(
    model: "BDH", states: list[torch.Tensor], idx_chunk: torch.Tensor, start_position: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Process one chunk of tokens (`idx_chunk`: (B, L), L >= 1) through the
    full model, given the running per-layer states from everything streamed
    so far and this chunk's absolute starting position. Returns
    `(new_states, logits)` with `logits` shaped (B, L, vocab_size);
    `new_states` folds in this chunk's own contribution and is ready to pass
    to the next call. L=1 is token-by-token streaming; L=T with a
    freshly-`init_bdh_states` state and `start_position=0` is mathematically
    identical to `BDH.forward`'s parallel computation (same formula, same
    operations -- the intra-chunk term alone covers the whole sequence and
    the cross-chunk term is exactly zero).
    """
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
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        V = x

        intra = (QR @ KR.mT).tril(diagonal=-1) @ V
        prefix_state = states[level]
        cross = QR @ prefix_state
        yKV = intra + cross
        yKV = model.ln(yKV)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, L, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ V
        new_states.append(prefix_state + chunk_contribution)

    logits = x.view(B, L, D) @ model.lm_head
    return new_states, logits


def bdh_stream_sequence(
    model: "BDH", idx: torch.Tensor, chunk_sizes: list[int], states: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Drive `bdh_stream_chunk` over arbitrary chunk boundaries. `chunk_sizes`
    must sum to `idx.shape[1]`; each may be any positive length (1 for pure
    token-by-token, or irregular sizes to test arbitrary chunk boundaries).
    `states` defaults to a fresh stream (`init_bdh_states`) -- pass a
    previously-returned state to resume mid-sequence."""
    B, T = idx.shape
    if sum(chunk_sizes) != T:
        raise ValueError(f"chunk_sizes must sum to sequence length {T}, got {sum(chunk_sizes)}")
    if states is None:
        states = init_bdh_states(model, B, device=idx.device)
    outputs = []
    position = 0
    for size in chunk_sizes:
        chunk = idx[:, position:position + size]
        states, logits = bdh_stream_chunk(model, states, chunk, start_position=position)
        outputs.append(logits)
        position += size
    return states, torch.cat(outputs, dim=1)
