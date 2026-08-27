"""Multi-token-prediction auxiliary loss for BDHVBSubspaceDecoder, Phase 2
of plans/HatchlingZero_Qwen_Integration_Plan_2026-08-26.md#6.

Real, deliberately SIMPLIFIED MTP, not Qwen3.8-Flash-Next's own 4B
sequential-prediction module (which chains real transformer blocks to
predict t+2 conditioned on a predicted t+1, etc). This is the cheaper
auxiliary-loss variant the integration plan explicitly scoped: reuse
the SAME final round's hidden state `x` for every offset, apply a
separate small linear head per future position (t+2/t+3/t+4), and add
their weighted cross-entropy to the primary t+1 loss. No extra forward
passes through the recurrent rounds -- the added cost is only the extra
head matmuls (D x vocab_size each) and their loss terms.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_subspace_decoder_checkpointed_torch import _vb_subspace_decoder_checkpoint_iteration
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder


def add_mtp_heads(model: BDHVBSubspaceDecoder, offsets: list[int]) -> None:
    """Attaches mtp_head_{k} nn.Parameters (D, vocab_size) for each offset
    k>=2 directly onto the model instance -- same registration mechanism
    P/O already use (plain attribute assignment on an nn.Module), so
    model.parameters() and model.named_parameters() pick them up for free,
    no config/base-class changes needed."""
    C = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for k in offsets:
        if k < 2:
            continue
        name = f"mtp_head_{k}"
        if not hasattr(model, name):
            head = nn.Parameter(torch.zeros(C.n_embd, C.vocab_size, device=device, dtype=dtype).normal_(std=0.02))
            setattr(model, name, head)


def mtp_weight_for(k: int) -> float:
    """Real plan-specified schedule: 1.0, 0.5, 0.25, 0.125 for t+1..t+4."""
    return 1.0 * (0.5 ** (k - 1))


def bdh_vb_subspace_decoder_forward_mtp_checkpointed(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    n_iterations: int,
    target1: torch.Tensor,
    extra_targets: dict[int, torch.Tensor],
):
    """extra_targets[k] must be data[:, k:] (real overlap-truncation, not a
    longer read from the data pipeline -- see plan section 6/16 discussion:
    packed windows are a fixed 256 tokens, so offset-k targets just reuse
    the tail of the SAME window, valid length shrinks by (k-1) tokens,
    truncated from the loss rather than requiring a wider read)."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)
    for _iteration in range(n_iterations):
        x = torch.utils.checkpoint.checkpoint(
            _vb_subspace_decoder_checkpoint_iteration, x, model, B, T, D, nh, N, use_reentrant=False,
        )

    logits = x.view(B, T, D) @ model.lm_head
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target1.view(-1))

    for k, target_k in extra_targets.items():
        Tk = target_k.size(1)
        head = getattr(model, f"mtp_head_{k}")
        logits_k = x[:, :, :Tk, :].reshape(B, Tk, D) @ head
        loss_k = F.cross_entropy(logits_k.reshape(-1, logits_k.size(-1)), target_k.reshape(-1))
        loss = loss + mtp_weight_for(k) * loss_k

    return logits, loss
