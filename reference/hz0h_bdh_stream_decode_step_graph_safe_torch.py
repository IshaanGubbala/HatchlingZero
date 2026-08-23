"""Single-token BDH streaming decode step, CUDA-graph-safe.

`bdh_stream_chunk` (reference/hz0h_bdh_torch.py) takes `start_position` as
a plain Python int. Under `torch.compile(mode="reduce-overhead")` (which
uses CUDA graphs), a changing Python scalar argument forces dynamo to
record a genuinely new graph for every distinct value it sees -- observed
directly: 51 distinct graph recordings over 64 decode steps, and compiled
throughput 4.5x SLOWER than eager (9.6 vs 42.9 tok/s) because almost every
step pays full re-recording cost instead of a cheap graph replay.

This function is exactly `bdh_stream_chunk` specialized to L=1 (always
true for single-token decode) with `position` accepted as a 0-dim tensor
instead of a Python int, computing `r_phases` directly (`position *
freqs`) instead of going through `torch.arange(start_position,
start_position+L)` -- L=1 means that arange call was always exactly
`[start_position]`, so this changes nothing about the math, only how the
position value flows into the graph (as a tensor value dynamo can treat
as a genuine runtime input, not a specializing constant).

For L=1, `(QR @ KR.mT).tril(diagonal=-1)` is provably always the zero
matrix (a 1x1 matrix has no strictly-below-diagonal entries) -- so
`intra` is omitted entirely rather than computed and discarded; this is
not an approximation, it is what `bdh_stream_chunk`'s own formula already
evaluates to at L=1, made explicit rather than computed and thrown away.

`bdh_stream_decode_step_graph_safe_inplace` goes one step further: BDH's
per-layer state at this model's width (N=4992) is ~1.59GB total across 8
layers -- `bdh_stream_chunk`'s `new_states.append(prefix_state +
chunk_contribution)` allocates a fresh tensor every call, and CUDA-graph
safety then required cloning that entire 1.59GB every decode step before
feeding it back in as the next call's input (confirmed by direct
measurement: this made compiled 1.45x SLOWER than eager, not faster --
the clone's own memory traffic dwarfed the actual per-token compute).
This variant mutates the pre-allocated state buffers in place
(`prefix_state.add_(chunk_contribution)`) and returns the SAME tensor
objects every call, so there is nothing to clone -- the standard,
textbook way stateful CUDA-graph decode loops are built in production
inference engines (persistent buffers mutated in place, not
allocate-and-swap).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH


def bdh_stream_decode_step_graph_safe(
    model: "BDH",
    states: list[torch.Tensor],
    idx_token: torch.Tensor,
    position: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Single-token (L=1) decode step. `idx_token`: (B, 1). `position`:
    0-dim (or (1,)) tensor holding this token's absolute position -- must
    be a tensor, not a Python int, to stay CUDA-graph-safe under compile.
    Returns (new_states, logits) exactly like bdh_stream_chunk with L=1."""
    c = model.config
    B = idx_token.shape[0]
    D = c.n_embd
    nh = c.n_head
    N = D * c.mlp_internal_dim_multiplier // nh

    x = model.embed(idx_token).unsqueeze(1)
    x = model.ln(x)

    r_phases = position.to(model.attn.freqs.dtype).view(1, 1, 1, 1) * model.attn.freqs

    new_states = []
    for level in range(c.n_layer):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        V = x

        prefix_state = states[level]
        cross = QR @ prefix_state
        yKV = model.ln(cross)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, 1, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ V
        new_states.append(prefix_state + chunk_contribution)

    logits = x.view(B, 1, D) @ model.lm_head
    return new_states, logits


def bdh_stream_decode_step_graph_safe_inplace(
    model: "BDH",
    states: list[torch.Tensor],
    idx_token: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    """Same math as bdh_stream_decode_step_graph_safe, but mutates each
    layer's state tensor in place (`.add_()`) instead of returning fresh
    tensors -- see this module's docstring for why. `states` is mutated
    as a side effect (same tensor objects, updated values) and NOT
    returned; only `logits` is returned, since there is nothing new to
    hand back for the state."""
    c = model.config
    B = idx_token.shape[0]
    D = c.n_embd
    nh = c.n_head
    N = D * c.mlp_internal_dim_multiplier // nh

    x = model.embed(idx_token).unsqueeze(1)
    x = model.ln(x)

    r_phases = position.to(model.attn.freqs.dtype).view(1, 1, 1, 1) * model.attn.freqs

    for level in range(c.n_layer):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        V = x

        prefix_state = states[level]
        cross = QR @ prefix_state
        yKV = model.ln(cross)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, 1, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

        chunk_contribution = KR.mT @ V
        prefix_state.add_(chunk_contribution)

    logits = x.view(B, 1, D) @ model.lm_head
    return logits
