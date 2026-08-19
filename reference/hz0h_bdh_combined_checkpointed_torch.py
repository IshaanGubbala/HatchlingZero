"""Gradient-checkpointed training forward for the COMBINED recipe
(`reference/hz0h_bdh_combined_best_torch.py`'s `softmax_scaled`
attention), matching the checkpointing this project already has for
plain BDH (`reference/hz0h_bdh_checkpointed_torch.py`'s
`bdh_variable_depth_forward_checkpointed` -- reuse that one directly for
the raw/upstream-attention path; this file exists only because that one
doesn't know about `softmax_scaled` attention).

Real motivation: a 599M-param `mult=32` BDH OOM'd on a 12GB RTX 3060
even at batch_size=4 without checkpointing (Windows dispatch,
`hz0h_matched_param_capstone_request.txt`, 2026-08-19) -- ~9.6GB of
model+gradient+AdamW-optimizer-state alone for a 599M-param model in
fp32, leaving no room for `n_layer` un-checkpointed layers' worth of
wide intermediate tensors (`x_sparse`/`y_sparse`/etc, size `B*T*D*mult`
per tensor, independent of head count). The `combined_best` arm
(`mult=16`) has half that latent blowup but hits the same class of
problem at large enough scale, and needs its own checkpointed path
since its attention differs from upstream's.

Real, disclosed prior finding this reuses rather than re-derives: this
project already measured (`reference/hz0h_bdh_checkpointed_torch.py`'s
`resolve_bdh_activation_policy` docstring) that on CUDA, recompute
checkpointing is a net win eagerly but costs ~27% throughput under
`torch.compile` for only ~20% additional memory savings there. That
finding is about EAGER vs COMPILED CUDA specifically; this file's
checkpointing is only used for training (never for the
`torch.compile`-wrapped throughput measurement in
`scripts/hz0h_bdh_combined_best_comparison.py`, which profiles the
already-trained model's inference path, not this training path).

Never modifies `reference/hz0h_bdh_torch.py`,
`reference/hz0h_bdh_checkpointed_torch.py`, or
`reference/hz0h_bdh_combined_best_torch.py`. Checkpointing must not
change the actual computation -- proven, not asserted, by
`tests/reference/test_hz0h_bdh_combined_checkpointed_torch.py`, which
checks BOTH logits AND gradients match
`reference/hz0h_bdh_combined_best_torch.py`'s uncheckpointed
`combined_bdh_forward` exactly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_combined_best_torch import _softmax_scaled_attention
from reference.hz0h_bdh_torch import BDH


def combined_bdh_forward_training_checkpointed(
    model: BDH,
    idx: torch.Tensor,
    depth: int,
    targets: torch.Tensor | None = None,
):
    """Same computation as
    `combined_bdh_forward_with_trajectory` (`softmax_scaled` attention)
    followed by `combined_bdh_forward`'s logits/loss tail, with each
    recurrent iteration wrapped in `torch.utils.checkpoint.checkpoint`.
    TRAINING only -- no trajectory capture, since that is fundamentally
    incompatible with checkpointing discarding intermediate states by
    design (the jump-operator distillation still needs and uses the
    uncheckpointed `combined_bdh_forward_with_trajectory`)."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd

    freqs = model.attn.freqs
    r_phases = torch.arange(0, T, device=freqs.device, dtype=freqs.dtype).view(1, 1, -1, 1) * freqs

    x = model.ln(model.embed(idx).unsqueeze(1))

    def layer(x_in: torch.Tensor) -> torch.Tensor:
        x_sparse = F.relu(x_in @ model._w(model.encoder))
        yKV = model.ln(_softmax_scaled_attention(model, x_in, x_sparse, r_phases, T))

        y_sparse = F.relu(yKV @ model._w(model.encoder_v))
        xy_sparse = model.drop(x_sparse * y_sparse)

        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        return model.ln(x_in + model.ln(yMLP))

    if not torch.is_grad_enabled():
        for _iteration in range(depth):
            x = layer(x)
    else:
        for _iteration in range(depth):
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
