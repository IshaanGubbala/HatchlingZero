"""Two-stream gated residual for BDHVBSubspaceDecoder, Phase 4 of
plans/HatchlingZero_Qwen_Integration_Plan_2026-08-26.md#8.

Real, deliberately conservative construction, per spec:

    y = g1 * LN(decoder_down(alpha1)) + g2 * LN(decoder_down2(alpha2))
    x = LN(x + y)

where the FIRST stream (decoder_up/decoder_down, g1) is exactly the
existing compound model's factored decoder, and g1 starts at exactly
1.0 -- not "near" 1.0 -- so at initialization this reproduces the known-
good compound model's behavior up to only the second term. The SECOND
stream (decoder_up2/decoder_down2, a separate small random-init
factored decoder of the same rank) is gated by g2, starting near zero
(0.01, not exact zero -- exact zero would also zero the plastic
stream's own gradient, d(g2*y2)/d(y2's params) = g2). Unlike
Muon/MTP/n-gram (all of which perturbed the model from step 0 and all
three lost their first real test), this starts almost exactly at the
already-validated solution and only has to prove the plastic stream is
worth its keep from there.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder


def add_gated_residual_stream(model: BDHVBSubspaceDecoder, rank: int | None = None, g2_init: float = 0.01,
                               single_stream: bool = False, g1_init: float = 1.0, g1_fixed: bool = False) -> None:
    """single_stream=True builds ONLY g1 (no decoder2/g2 at all) -- the
    isolating ablation for the real 2026-08-28 two-stream result, whose
    g2 ended at 0.0002 (~unchanged from init) while g1 dropped from 1.0
    to 0.583. If that drop alone explains the win, this cheaper
    single-parameter variant should reproduce it without the extra
    decoder_up2/decoder_down2 parameters.

    g1_init/g1_fixed, 2026-08-30: the real adaptive-gate result
    (val_loss=1.4023, gate collapsed to a constant ~0.5508 -- see
    plans/newnewplan.md section B) raised a genuine question this
    project hasn't isolated yet: is a FIXED scalar at ~0.55 already as
    good as the "adaptive" controller, or did the controller's own
    parameterization/training dynamics matter even though its output
    is constant? g1_fixed=True freezes g1 at g1_init for the whole run
    (requires_grad_(False)) -- if a hard-coded g=0.55 reproduces 1.4023,
    the adaptive-gate mechanism is dead weight; if it doesn't, something
    about HOW that value was reached mattered, not just the value."""
    C = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.g1 = nn.Parameter(torch.tensor(g1_init, device=device, dtype=torch.float32))
    if g1_fixed:
        model.g1.requires_grad_(False)
    model._gated_residual_single_stream = single_stream
    if single_stream:
        print(f"[gated_residual] single-stream ablation: g1 only, no decoder2/g2 "
              f"(g1_init={g1_init} g1_fixed={g1_fixed})", flush=True)
        return
    r = rank if rank is not None else C.subspace_rank
    nh = C.n_head
    N = C.n_embd * C.mlp_internal_dim_multiplier // nh
    model.decoder_up2 = nn.Parameter(torch.zeros((nh * N, r), device=device, dtype=dtype).normal_(std=0.02))
    model.decoder_down2 = nn.Parameter(torch.zeros((r, C.n_embd), device=device, dtype=dtype).normal_(std=0.02))
    model.g2 = nn.Parameter(torch.tensor(g2_init, device=device, dtype=torch.float32))
    print(f"[gated_residual] rank={r} g1_init=1.0 g2_init={g2_init}", flush=True)


def _gated_residual_checkpoint_iteration(x: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int) -> torch.Tensor:
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

    alpha1 = torch.matmul(xy_sparse, model.decoder_up.view(nh, N, -1)).sum(dim=1, keepdim=True)
    y1 = model.ln(alpha1 @ model.decoder_down)

    if model._gated_residual_single_stream:
        y = model.g1 * y1
    else:
        alpha2 = torch.matmul(xy_sparse, model.decoder_up2.view(nh, N, -1)).sum(dim=1, keepdim=True)
        y2 = model.ln(alpha2 @ model.decoder_down2)
        y = model.g1 * y1 + model.g2 * y2
    x = model.ln(x + y)
    return x


def bdh_vb_subspace_decoder_forward_gated_residual_checkpointed(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
):
    """model.decoder_up2/decoder_down2/g1/g2 must already be attached (see
    add_gated_residual_stream). This IS a real architectural component
    (like n-gram memory, unlike MTP's train-only aux heads) -- present at
    both train and eval time."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _iteration in range(n_iterations):
        x = torch.utils.checkpoint.checkpoint(
            _gated_residual_checkpoint_iteration, x, model, B, T, D, nh, N, use_reentrant=False,
        )

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
