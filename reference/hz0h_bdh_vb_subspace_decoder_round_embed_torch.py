"""Round-conditioned recurrence for BDHVBSubspaceDecoder, Phase 2 of
plans/HatchlingZero_Internal_Computation_Phase_2026-08-29.md.

Real, minimal change: z_{r+1} = F(z_r, x, e_r) instead of z_{r+1} =
F(z_r, x) -- a small, learnable per-round embedding `e_r` injected
additively into the residual stream at the START of round r, before
any computation happens. Lets the shared, weight-tied BDH round
function behave differently at different computational depths without
adding separate per-round parameters (the point of the whole
Qwen-integration phase's addressing-resists-compression /
value-tolerates-compression finding still applies here: this injects
into the residual stream BEFORE the addressing computation, i.e. it
conditions what the shared E/E_v/attention SEE, not a routing/gating
change to those weights themselves).

Real, disclosed design choice: `e_r` initialized small (std=0.02,
matching every other new-embedding convention this project uses --
n-gram table, MTP heads), NOT gated with a separate scalar the way
Phase 4/7 were. Round conditioning is not a learned decision the way a
router is (there's no "wrong round" a badly-initialized gate needs
protecting against) -- it's a fixed, deterministic function of which
round the computation is in, so the conservative-gate concern that
motivated Phase 4/7's near-zero init doesn't apply the same way here.
A small init keeps the embedding table's initial influence modest
regardless.

Table size is `n_layer` (the trained depth) -- NOT sized to cover
R-scaling extrapolation (R>n_layer) with dedicated learned rows, since
a depth curriculum that never reaches R>n_layer during training would
leave those extra rows permanently untrained. Real R>n_layer behavior
(if tested) reuses the LAST trained round's embedding row for every
round beyond training depth -- a simple, disclosed heuristic, not a
principled extrapolation scheme.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder


def add_round_embeddings(model: BDHVBSubspaceDecoder, n_rounds: int | None = None) -> None:
    C = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_rounds = n_rounds if n_rounds is not None else C.n_layer
    model.round_embed = nn.Parameter(torch.zeros((n_rounds, C.n_embd), device=device, dtype=dtype).normal_(std=0.02))
    print(f"[round_embed] n_rounds={n_rounds} n_embd={C.n_embd}", flush=True)


def _round_embed_checkpoint_iteration(x: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int, round_idx: int):
    n_rows = model.round_embed.shape[0]
    e_r = model.round_embed[min(round_idx, n_rows - 1)]  # reuse last row beyond trained depth (see module docstring)
    x = model.ln(x + e_r)

    x_latent = x @ model.encoder
    x_sparse = F.relu(x_latent)

    v_bottleneck = x @ model.P
    yKV_bottleneck = model.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck)
    yKV = yKV_bottleneck @ model.O
    yKV = model.ln(yKV)

    y_latent = yKV @ model.encoder_v
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse
    xy_sparse = model.drop(xy_sparse)

    alpha = torch.matmul(xy_sparse, model.decoder_up.view(nh, N, -1)).sum(dim=1, keepdim=True)
    yMLP = alpha @ model.decoder_down
    y = model.ln(yMLP)
    x = model.ln(x + y)
    return x


def bdh_vb_subspace_decoder_forward_round_embed_checkpointed(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
):
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for round_idx in range(n_iterations):
        x = torch.utils.checkpoint.checkpoint(
            _round_embed_checkpoint_iteration, x, model, B, T, D, nh, N, round_idx, use_reentrant=False,
        )

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
