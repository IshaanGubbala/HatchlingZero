"""Ablations of exact BDH's own INHERITED attention primitives -- the
constants transcribed verbatim from `github.com/pathwaycom/bdh` that this
project faithfully preserved but never actually questioned.

Real motivation: the width sweep
(`scripts/hz0h_bdh_width_flop_frontier_local.py`) found that BDH's
canonical latent width multiplier is roughly 2x larger than it needs to
be -- an inherited hyperparameter that had never been validated at this
project's scale. That is evidence of a general pattern, not a one-off:
faithfulness to upstream produced a verified oracle (correct, and worth
having), but fidelity was never followed by "is this choice optimal?".
Notably, upstream's own `BDHConfig` default is
`mlp_internal_dim_multiplier=128` while this project runs 32 -- so a
large, undocumented deviation was already made long ago. There is no
fidelity purity left to protect here; these are open empirical
questions.

The four inherited attention primitives this file makes testable, each
documented in `docs/restart/hz0h_bdh_component_map.md` as a real,
deliberately-preserved property of the upstream code, and none of them
ever ablated:

1. `theta = 2**16` (65536) in RoPE frequency construction, vs the
   standard RoPE value of 10000 -- 6.5x larger. This sets the positional
   frequency spectrum, which governs long-context behaviour, which is
   BDH's single real inference-side win over the Transformer.
2. `tril(diagonal=-1)`: STRICTLY lower-triangular causal masking, so a
   position cannot attend to itself, only to strictly earlier ones.
   Standard causal attention uses `diagonal=0`.
3. No score scaling: raw `QR @ KR^T` with no `/sqrt(d)`, over an
   expanded latent width (2048 at production shape).
4. No softmax anywhere: raw `scores @ V`.

Never modifies `reference/hz0h_bdh_torch.py` (read-only upstream
oracle). `ablated_bdh_forward` with default arguments is exactly the
oracle's own computation -- proven, not asserted, by
`tests/reference/test_hz0h_bdh_primitive_ablations_torch.py`, which
gates every other claim this file can support.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import Attention, BDH, get_freqs


def build_rope_freqs(model: BDH, theta: float) -> torch.Tensor:
    """Real RoPE frequency buffer at an arbitrary `theta`, built by the
    oracle's OWN `get_freqs` (not a reimplementation) so only the theta
    value differs from upstream."""
    C = model.config
    latent_width = C.mlp_internal_dim_multiplier * C.n_embd // C.n_head
    freqs = get_freqs(latent_width, theta=theta, dtype=torch.float32)
    return freqs.view(1, 1, 1, latent_width).to(model.attn.freqs.device)


def ablated_bdh_forward(
    model: BDH,
    idx: torch.Tensor,
    depth: int,
    targets: torch.Tensor | None = None,
    *,
    freqs: torch.Tensor | None = None,
    mask_diagonal: int = -1,
    scale_scores: bool = False,
    use_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Exact BDH forward at `depth` recurrent iterations, with the four
    inherited attention primitives above made switchable.

    Defaults reproduce the oracle EXACTLY (`freqs=None` uses the model's
    own `theta=2**16` buffer, `mask_diagonal=-1`, no scaling, no
    softmax). Every non-default argument is a real, deliberate deviation
    from upstream.

    Real, disclosed detail for `use_softmax` with `mask_diagonal=-1`:
    position 0 has NO permitted attention targets, so a softmax over an
    all-masked row is genuinely undefined (0/0). Those rows are set to
    exact zero -- the same value the oracle's own unnormalized path
    produces there (an all-zero score row times V is zero), so this is a
    faithful handling of a real edge case rather than a silent NaN.
    """
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    latent_width = D * C.mlp_internal_dim_multiplier // nh

    if freqs is None:
        freqs = model.attn.freqs
    scale = 1.0 / math.sqrt(latent_width) if scale_scores else 1.0

    x = model.ln(model.embed(idx).unsqueeze(1))
    r_phases = torch.arange(0, T, device=freqs.device, dtype=freqs.dtype).view(1, 1, -1, 1) * freqs

    for _level in range(depth):
        x_sparse = F.relu(x @ model._w(model.encoder))
        QR = Attention.rope(r_phases, x_sparse)
        scores = QR @ QR.mT
        if scale_scores:
            scores = scores * scale

        if use_softmax:
            allowed = torch.ones(T, T, dtype=torch.bool, device=scores.device).tril(diagonal=mask_diagonal)
            scores = scores.masked_fill(~allowed, float("-inf"))
            scores = torch.softmax(scores, dim=-1)
            # Rows with no permitted targets softmax to NaN; the oracle's own
            # unnormalized path yields exact zero there, so match that.
            scores = torch.nan_to_num(scores, nan=0.0)
        else:
            scores = scores.tril(diagonal=mask_diagonal)

        yKV = model.ln(scores @ x)
        y_sparse = F.relu(yKV @ model._w(model.encoder_v))
        xy_sparse = model.drop(x_sparse * y_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, latent_width * nh) @ model._w(model.decoder)
        x = model.ln(x + model.ln(yMLP))

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return logits, loss
