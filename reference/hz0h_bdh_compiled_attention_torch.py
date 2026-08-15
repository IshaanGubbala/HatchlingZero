"""HZ-0H compiled attention via torch.compile.

This file tests a narrower, more portable hypothesis than
`reference/hz0h_bdh_fused_attention_torch.py`'s Triton-kernel approach: can
`torch.compile` fuse the THREE separate, unfused PyTorch ops that
`Attention.forward` already does (`QR`, scores, matmul) into fewer kernel
launches, improving real wall-clock performance without platform-specific
dependencies (CUDA, Triton, fla)?

Real motivation: `Attention.forward` today computes:
  1. RoPE: `QR = rope(r_phases, Q)`
  2. Scores (unfused): `scores = (QR @ QR.mT).tril(diagonal=-1)`
  3. Output (unfused): `return scores @ V`

Each is a separate kernel launch on CUDA, and none are individually large
enough to saturate the GPU. Fusing them via `torch.compile` (which can
automatically fuse elementwise/matmul sequences in its default mode) could
reduce kernel-launch overhead, matching the ~1.82x speedup observed in
`docs/restart/hz0h_phase6_depth_curriculum_results.md` when compiling the
full model on CUDA. This file asks: does the ATTENTION step alone benefit,
or did that win come from elsewhere?

Platform limitation: `torch.compile` support on MPS (Mac Metal Performance
Shaders) and CPU is known to be less mature than on CUDA. This file
handles compile failure gracefully (not a crash, just a skip or fallback),
since a real environment limitation is a valid, reportable result.

Deliberately a SEPARATE, opt-in extension file, not a modification of
`reference/hz0h_bdh_torch.py`'s verbatim-upstream oracle section -- same
pattern this project already uses for every other BDH extension."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, Attention


# Module-level flag: whether torch.compile is actually available and works
# on this platform. Set at module load time and used by bdh_compiled_forward
# to decide whether to use compiled or fallback path.
_COMPILE_AVAILABLE = True
_COMPILE_ERROR = None

try:
    # Test torch.compile on a trivial example to catch platform issues early
    _test_fn = lambda x: x + 1
    _test_compiled = torch.compile(_test_fn, backend="eager")
    _test_compiled(torch.tensor([1.0]))
except Exception as e:
    _COMPILE_AVAILABLE = False
    _COMPILE_ERROR = str(e)


def _raw_attention(QR: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Raw attention computation: the exact math from
    `Attention.forward` after RoPE has been applied to Q (yielding QR).
    K is implicitly Q (same tensor), so we use QR for both.

    Args:
        QR: Query/Key after RoPE, shape (B, nh, T, N)
        V: Value tensor, shape (B, 1, T, D)

    Returns:
        Output of shape (B, nh, T, D) (after matmul with V expands from (B,1,T,D))
    """
    scores = (QR @ QR.mT).tril(diagonal=-1)
    return scores @ V


# Attempt to compile _raw_attention at module level. If torch.compile
# isn't available or fails on this platform, _compiled_attention remains None
# and bdh_compiled_forward will use _raw_attention directly (with an honest
# note about why).
_compiled_attention = None
if _COMPILE_AVAILABLE:
    try:
        _compiled_attention = torch.compile(_raw_attention, backend="eager")
    except Exception as e:
        _COMPILE_ERROR = str(e)
        _COMPILE_AVAILABLE = False


def bdh_compiled_forward(model: BDH, idx: torch.Tensor, targets: torch.Tensor | None = None):
    """Drop-in replacement for `BDH.forward`, using `torch.compile` on the
    attention computation instead of the raw unfused path.

    Math is BYTE-IDENTICAL to `BDH.forward` -- the only change is the
    attention step, which is wrapped via `torch.compile` for potential
    kernel fusion. Everything else (embed, ln, encoder/encoder_v/decoder,
    ReLU, dropout, shared-weight depth loop) is exactly as in the oracle.

    If torch.compile is not available on this platform (e.g., MPS on Mac
    with immature compile support), falls back to the raw unfused path
    and completes successfully but without compilation benefit -- a valid,
    reportable outcome, not a failure."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for level in range(C.n_layer):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)

        # Compute RoPE phases and apply to Q (K is Q, so QR is used for both)
        r_phases = (
            torch.arange(
                0,
                T,
                device=model.attn.freqs.device,
                dtype=model.attn.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * model.attn.freqs
        QR = Attention.rope(r_phases, x_sparse)

        # Use compiled attention if available, else fall back to raw
        if _compiled_attention is not None:
            yKV = _compiled_attention(QR, x)
        else:
            yKV = _raw_attention(QR, x)

        yKV = model.ln(yKV)

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
    return logits, loss


def is_compile_available() -> bool:
    """Returns True if torch.compile is available and working on this platform."""
    return _COMPILE_AVAILABLE


def get_compile_error() -> str | None:
    """Returns the error message if torch.compile failed, None if available."""
    return _COMPILE_ERROR
