"""Inference-only ablation: how often does BDH's recurrent reasoning
actually need to re-query the persistent state?

Every round of exact BDH does `yKV = attn(Q=x_sparse, K=x_sparse, V=x)` --
this IS the state/context access (attention's streaming-equivalent form
is exactly `q_r @ S_r`, the expensive part). This function keeps every
round's full nonlinear transformation (x_sparse, the multiplicative gate,
the decoder projection, the residual update) exactly as-is, and only
controls how often `yKV` gets recomputed vs reused from the last real
read -- `refresh_every=1` is mathematically identical to vanilla BDH
(fresh read every round, proven bit-exact below); `refresh_every=2` reads
at rounds 0,2,4,6 and reuses the previous read for rounds 1,3,5,7;
`refresh_every=n_iterations` (or higher) reads only once, at round 0.

This is inference-only and deliberately NOT a training change: the goal
is to test on an already-trained checkpoint whether the model's learned
behavior tolerates stale context before deciding whether it's worth
retraining anything. If quality holds at refresh_every=4 (2 real reads
out of 8 rounds), that's real evidence the R-times state-read cost is
mostly redundant, not load-bearing -- the necessary premise before
building any "Context-Latched BDH" architecture.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH


def bdh_context_refresh_forward(
    model: BDH,
    idx: torch.Tensor,
    n_iterations: int,
    refresh_every: int,
    targets: torch.Tensor | None = None,
):
    if refresh_every < 1:
        raise ValueError("refresh_every must be >= 1")
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    cached_yKV = None
    real_reads = 0
    for r in range(n_iterations):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)

        if r % refresh_every == 0 or cached_yKV is None:
            yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = model.ln(yKV)
            cached_yKV = yKV
            real_reads += 1
        else:
            yKV = cached_yKV

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss, real_reads
