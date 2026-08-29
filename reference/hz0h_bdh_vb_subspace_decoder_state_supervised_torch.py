"""State-supervised recurrence for BDHVBSubspaceDecoder, Phase 3 of
plans/HatchlingZero_Internal_Computation_Phase_2026-08-29.md.

Real upgrade of Phase 1's diagnostic-only probe into an actual training
signal: instead of just MEASURING whether z_r decodes the true task
state (a frozen probe, discarded after), attach small per-round linear
heads and train them jointly with the LM loss on real synthetic
examples with known ground truth, hoping this explicitly shapes the
shared recurrence to compute and refine that state across rounds.

L = L_LM + lambda_state * L_state

Real, conservative design, with MTP's real 2026-08-28 failure as the
standing warning that a reasonable-looking auxiliary loss can damage
BDH: lambda swept over {0.01, 0.03, 0.1}, NOT the much larger
coefficients an earlier, unrelated proposal used. L_state is only
computed on batches that carry real state labels (the synthetic
object-location task); ordinary LM batches train on L_LM alone, state
probe heads get zero gradient from those batches. Promotion requires
ordinary LM validation loss to stay intact -- this module does not
itself decide promotion, the training script measures both val_loss
and a real held-out round-state-probe comparison against the Phase 1/2
baselines.

state_probe_heads are conceptually "temporary" (per the plan's own
framing) -- real trainable parameters during training (so they can
provide real gradient), but never used at inference/eval time the way
the model's own lm_head is; evaluation always uses the model's real
`lm_head`/ordinary forward path, matching MTP's own precedent of
aux heads that don't participate in eval.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder


def add_state_probe_heads(model: BDHVBSubspaceDecoder, n_classes: int) -> None:
    C = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.state_probe_heads = nn.Parameter(torch.zeros((C.n_layer, C.n_embd, n_classes), device=device, dtype=dtype).normal_(std=0.02))
    print(f"[state_supervised] n_layer={C.n_layer} n_classes={n_classes}", flush=True)


def _state_supervised_checkpoint_iteration(x: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int):
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


def bdh_vb_subspace_decoder_forward_state_supervised_checkpointed(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
    state_labels: torch.Tensor | None = None,
    lambda_state: float = 0.0,
):
    """state_labels: (B,) long tensor of ground-truth class indices, or
    None for ordinary LM-only batches (no state loss computed, state
    probe heads get zero gradient for that batch -- matches how
    add_domain_banks's hard 0/1 selection makes freezing fall out of
    the forward computation automatically, same mechanism here)."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    state_losses = []
    for level in range(n_iterations):
        x = torch.utils.checkpoint.checkpoint(
            _state_supervised_checkpoint_iteration, x, model, B, T, D, nh, N, use_reentrant=False,
        )
        if state_labels is not None:
            last_token = x[:, :, -1, :].reshape(B, D)
            state_logits = last_token @ model.state_probe_heads[level]
            state_losses.append(F.cross_entropy(state_logits, state_labels))

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    if state_labels is not None and state_losses:
        state_loss = lambda_state * torch.stack(state_losses).mean()
        loss = state_loss if loss is None else loss + state_loss
    return logits, loss
