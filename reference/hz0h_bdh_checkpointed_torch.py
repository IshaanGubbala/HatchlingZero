"""HZ Phase 6 (`plans/HatchlingZero_Reality_Plan.md`): Activation
checkpointing for variable-depth BDH training -- memory-efficient forward
pass that trades compute for GPU memory.

Real motivation: `reference/hz0h_bdh_variable_depth_torch.py` supports
arbitrary iteration counts, enabling curriculum training where depth
ramps 2→4→6→8. However, PyTorch's autograd retains EVERY iteration's
intermediate activations simultaneously (for backward-pass gradient
computation), so peak memory grows linearly with depth iterations.
`docs/restart/hz0h_phase_g_100m_scale_gate_pilot_results.md` documents
the real, measured problem: at 100M parameters, training hit a hard GPU
memory ceiling exactly at the curriculum's depth-2-to-4 transition (peak
memory jumped from 11.05 GiB to 12.14 GiB, crossing the 12 GiB card
ceiling). Activation checkpointing (`torch.utils.checkpoint.checkpoint`,
with modern `use_reentrant=False`) avoids storing intermediate
activations, recomputing them during backward instead. This file
implements this trade-off: SAME MATH as
`bdh_variable_depth_forward` (byte-for-byte identical outputs),
but with memory-efficient gradient computation. The cost is longer
backward pass due to recompute.

Important: this changes HOW gradients are computed (recompute vs store),
not WHAT is computed (same forward math, same backward gradients). Use
only when memory is a bottleneck; profile before and after to confirm
the recompute cost is acceptable.

Deliberately a SEPARATE, opt-in extension file, not a modification of
`reference/hz0h_bdh_torch.py` or `reference/hz0h_bdh_variable_depth_torch.py`
-- same pattern every other BDH extension uses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH


def _bdh_checkpoint_iteration(
    x: torch.Tensor,
    model: BDH,
    C: object,
    B: int,
    T: int,
    D: int,
    nh: int,
    N: int,
) -> torch.Tensor:
    """Single layer iteration for checkpointing. This function is passed
    to torch.utils.checkpoint.checkpoint, which will recompute it during
    backward rather than storing intermediate activations.

    Args:
        x: residual stream, shape (B, 1, T, D)
        model: BDH model instance
        C: model config (passed to avoid capturing in closure)
        B, T, D, nh, N: cached dimension values

    Returns:
        Updated residual stream x, shape (B, 1, T, D)
    """
    x_latent = x @ model._w(model.encoder)
    x_sparse = F.relu(x_latent)

    yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
    yKV = model.ln(yKV)

    y_latent = yKV @ model._w(model.encoder_v)
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse
    xy_sparse = model.drop(xy_sparse)

    yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
    y = model.ln(yMLP)
    x = model.ln(x + y)

    return x


def bdh_variable_depth_forward_checkpointed(
    model: BDH,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
):
    """Same real per-layer computation as `bdh_variable_depth_forward`,
    but each iteration wraps its block computation in
    torch.utils.checkpoint.checkpoint (with use_reentrant=False) to
    avoid storing intermediate activations across the depth loop.

    Memory impact: Reduces peak activation memory (scales linearly with
    n_iterations without checkpointing; nearly constant with it). Exact
    savings depend on model size and sequence length.

    Speed impact: Increases backward time due to recompute (intermediate
    activations are recomputed during backward instead of being stored).
    Speed ratio backward/forward varies but is typically >1x due to this
    recompute. Measure on your target hardware before production use.

    Args:
        model: BDH instance
        idx: input token indices, shape (B, T)
        n_iterations: number of layer-loop iterations (independent of
                      model.config.n_layer)
        targets: optional target tokens for loss computation, shape (B, T)

    Returns:
        (logits, loss): logits shape (B, T, vocab_size), loss scalar or None
    """
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _iteration in range(n_iterations):
        x = torch.utils.checkpoint.checkpoint(
            _bdh_checkpoint_iteration,
            x,
            model,
            C,
            B,
            T,
            D,
            nh,
            N,
            use_reentrant=False,
        )

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return logits, loss
