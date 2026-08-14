"""HZ Next-Phase Plan follow-up: a fused-kernel alternative to
`reference/hz0h_bdh_torch.py`'s `Attention.forward`, motivated by a
real, concrete finding from Phase F -- the matched Transformer trains
~5.3x faster and is ~6.2x more energy-efficient per token than exact
BDH at the same parameter count, and a real, already-observed piece of
evidence points at WHY: at long context, `transformer_prefill` (which
uses PyTorch's fused `scaled_dot_product_attention`, dispatching to a
hand-tuned flash-attention/tensor-core kernel) did NOT OOM where
`bdh_prefill` (raw, unfused `(QR @ KR.mT).tril(diagonal=-1) @ V`) did
(`docs/restart/hz0h_phase_f_same_gpu_comparison_results.md`). BDH's
hand-written attention never gets a fused kernel at all -- this file
tests whether giving it one closes some of that gap.

Real, load-bearing math note, read before trusting anything here: BDH's
own attention is explicitly NOT softmax attention (see
`reference/hz0h_bdh_torch.py`'s own module docstring) -- it's a raw,
UNNORMALIZED weighted sum, `scores @ V` with no softmax anywhere.
`torch.nn.functional.scaled_dot_product_attention` always applies
softmax internally with no way to disable it, so it CANNOT be used
here without changing BDH's actual math. What BDH's raw, undecayed,
strictly-causal weighted sum actually IS, mathematically, is the
zero-decay special case of gated linear attention (GLA) -- exactly the
family `flash_linear_attention`'s `chunk_gla` Triton kernel already
implements, and that this project already uses for a different mixer
(`reference/hz0a_torch_model.py`'s `_gdn2_via_fla_gla`, GDN-2's own
`decay*(1-erase)` gate reduces to GLA's `exp(g)` the same way BDH's
"no gate at all" reduces to `g=0`).

Real, non-obvious wrinkle this file exists to get right: `chunk_gla`'s
own causal convention (inherited from the GDN-2 reference form's
`torch.tril(ones(steps, steps))`, default `diagonal=0`) is
SELF-INCLUSIVE -- position t's output includes its own token's
key/value. BDH's own convention is `diagonal=-1`, SELF-EXCLUSIVE -- a
position cannot attend to itself, only strictly earlier positions (a
real, deliberate BDH design choice, not a bug, per the oracle file's
own docstring). These are NOT the same computation. Fixed here by
shifting K/V forward by one position (position 0 gets an all-zero
key/value) before calling `chunk_gla`: this turns its self-inclusive
sum over `s <= t` of the SHIFTED sequence into exactly BDH's own sum
over `s < t` of the ORIGINAL sequence (`shifted_k[s] = original_k[s-1]`
for s>=1, so `sum_{s<=t} q_t . shifted_k[s] . shifted_v[s] = sum_{s=1}^{t}
q_t . k[s-1] . v[s-1] = sum_{s'=0}^{t-1} q_t . k[s'] . v[s']`, exactly
BDH's own `tril(diagonal=-1)` sum). This is asserted here, not just
argued in prose -- see the real correctness test in
tests/reference/test_hz0h_bdh_fused_attention_torch.py, which requires
real CUDA hardware (chunk_gla is a Triton kernel).

**VERIFIED on real hardware (RTX3060, 2026-08-14)**: all 3 correctness
tests pass, including the exact-oracle-match test (the shift trick's
own claim, the riskiest untested part of the math argument above,
holds under real numerical comparison, not just multiple-choice-shape
coverage) and the forward+gradient-flow test. The forward-pass math is
real and correct. Speed/energy has NOT yet been measured -- do not cite
this as a proven throughput or energy win until that separate
measurement exists; correctness and performance are two different
claims, only the first is confirmed so far.

Deliberately a SEPARATE, opt-in extension file, not a modification of
`reference/hz0h_bdh_torch.py`'s verbatim-upstream oracle section --
same pattern this project already uses for every other BDH extension
(VB, selective writes, block-gating, INT8 state)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, Attention


def bdh_fused_attention(Q: torch.Tensor, V: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Real, exact (not approximate) alternative to `Attention.forward`,
    routed through `flash_linear_attention`'s `chunk_gla` kernel instead
    of a raw unfused matmul. `Q` shape (B, nh, T, N) -- BDH always uses
    K is Q (see `Attention.forward`'s own `assert K is Q`), so only one
    query/key tensor is needed. `V` shape (B, 1, T, D) or (B, nh, T, D)
    -- BDH's own real usage always passes (B, 1, T, D) (broadcast over
    heads via ordinary matmul broadcasting in the raw path); `chunk_gla`
    needs a real per-head V, so a (B, 1, T, D) input is explicitly
    expanded to (B, nh, T, D) here -- a real, disclosed extra memory
    cost this fused path pays that the raw broadcasting path doesn't,
    not free. `freqs` is `Attention.freqs`, the same real RoPE frequency
    buffer BDH already uses (unchanged formula, unchanged buffer)."""
    from fla.ops.gla import chunk_gla

    B, nh, T, N = Q.shape
    r_phases = torch.arange(0, T, device=freqs.device, dtype=freqs.dtype).view(1, 1, -1, 1) * freqs
    QR = Attention.rope(r_phases, Q)  # (B, nh, T, N), identical RoPE math to the raw path

    if V.shape[1] == 1:
        V = V.expand(B, nh, T, V.shape[-1])
    D = V.shape[-1]

    # Shift K (== QR) and V forward by one position along T: position 0
    # becomes an all-zero key/value, position t (t>=1) holds what was
    # originally at position t-1. See module docstring for why this
    # exactly converts chunk_gla's self-inclusive causal sum into BDH's
    # own self-exclusive (diagonal=-1) sum.
    KR_shifted = F.pad(QR, (0, 0, 1, 0))[:, :, :-1, :]
    V_shifted = F.pad(V, (0, 0, 1, 0))[:, :, :-1, :]

    q = QR.transpose(1, 2).contiguous()  # (B, T, nh, N)
    k = KR_shifted.transpose(1, 2).contiguous()  # (B, T, nh, N)
    v = V_shifted.transpose(1, 2).contiguous()  # (B, T, nh, D)
    g = torch.zeros_like(k)  # no decay -- BDH's real recurrence has none, S_t = S_{t-1} + K_t^T V_t

    out, _final_state = chunk_gla(q, k, v, g, scale=1.0, initial_state=None, output_final_state=False)
    return out.transpose(1, 2)  # back to (B, nh, T, D), matching Attention.forward's own output layout


def bdh_fused_forward(model: BDH, idx: torch.Tensor, targets: torch.Tensor | None = None):
    """Real, exact (pending the real-hardware correctness test) drop-in
    replacement for `BDH.forward`'s parallel/dense computation, using
    `bdh_fused_attention` in place of `model.attn(...)`'s raw matmul for
    the one intra-sequence attention step per layer. Everything else
    (embed, ln, encoder/encoder_v/decoder, ReLU sparsity, dropout,
    shared-weight depth loop) is byte-identical to `BDH.forward` --
    only the attention computation itself changes, and only in HOW it's
    computed, not what it computes (pending real verification)."""
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

        yKV = bdh_fused_attention(x_sparse, x, model.attn.freqs)
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
