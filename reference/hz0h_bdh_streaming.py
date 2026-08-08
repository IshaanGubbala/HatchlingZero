"""HZ-0H H2: streaming-state equivalence for BDH-GPU's attention.

H0 found (see docs/restart/hz0h_bdh_component_map.md) that the official
`bdh.py` has no explicit `rho`/state variable -- the paper's state-space
framing (Section 3) is a claimed mathematical equivalence, not a literal
object in the GPU tensor code. This module derives and tests that
equivalence directly rather than assuming it.

Because BDH-GPU's attention has NO softmax and NO normalization
(confirmed in H0/H1 -- raw `scores @ V`, not `softmax(scores) @ V`),
the causal linear attention is EXACTLY decomposable into a running
outer-product state, the same trick linear-attention-as-RNN
reformulations (linear transformers, RWKV, GLA) use:

    out[t] = QR[t] @ S[t],  S[t] = sum_{i<t} KR[i] (x) V[i]   (outer product)
    S[t+1] = S[t] + KR[t] (x) V[t]

This is an EXACT algebraic identity (`sum_n QR[t,n] * (sum_i<t KR[i,n]*V[i,d])
= sum_i<t (sum_n QR[t,n]*KR[i,n]) * V[i,d]`), not an approximation -- `S`
IS a real, concrete candidate for what the paper's `rho` refers to.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0h_bdh_mlx import Attention, BDHConfig


def streaming_attention(attn: Attention, Q: mx.array, V: mx.array) -> tuple[mx.array, mx.array]:
    """Token-by-token streaming form. Q is already the sparse latent
    (x_sparse), matching how BDH.forward calls attn(Q=x_sparse,
    K=x_sparse, V=x) -- K=Q always (H0's confirmed Q=K finding).
    Returns (outputs, final_state) where state has shape (B, nh, N, D)."""
    B, nh, T, N = Q.shape
    D = V.shape[-1]
    r_phases = mx.arange(0, T, dtype=attn.freqs.dtype).reshape(1, 1, -1, 1) * attn.freqs
    QR = attn.rope(r_phases, Q)
    KR = QR

    state = mx.zeros((B, nh, N, D), dtype=Q.dtype)
    outputs = []
    for t in range(T):
        out_t = QR[:, :, t:t + 1, :] @ state
        outputs.append(out_t)
        state = state + mx.swapaxes(KR[:, :, t:t + 1, :], -1, -2) @ V[:, :, t:t + 1, :]
    return mx.concatenate(outputs, axis=2), state


def chunked_streaming_attention(attn: Attention, Q: mx.array, V: mx.array, *, chunk_length: int) -> mx.array:
    """Arbitrary chunk-boundary streaming: processes `chunk_length`-sized
    blocks with the PARALLEL (in-chunk) form, carrying only the running
    outer-product state across chunk boundaries -- proves state handoff
    at arbitrary chunk sizes agrees with both the fully parallel and
    fully token-by-token forms, per H2's own "arbitrary chunked
    streaming" requirement, not just length-1 streaming."""
    B, nh, T, N = Q.shape
    D = V.shape[-1]
    r_phases_full = mx.arange(0, T, dtype=attn.freqs.dtype).reshape(1, 1, -1, 1) * attn.freqs
    QR_full = attn.rope(r_phases_full, Q)
    KR_full = QR_full

    state = mx.zeros((B, nh, N, D), dtype=Q.dtype)
    outputs = []
    start = 0
    while start < T:
        end = min(start + chunk_length, T)
        QR_chunk = QR_full[:, :, start:end, :]
        KR_chunk = KR_full[:, :, start:end, :]
        V_chunk = V[:, :, start:end, :]
        L = end - start

        intra = mx.tril(QR_chunk @ mx.swapaxes(KR_chunk, -1, -2), -1) @ V_chunk
        inter = QR_chunk @ state
        outputs.append(intra + inter)

        state = state + mx.swapaxes(KR_chunk, -1, -2) @ V_chunk
        start = end
    return mx.concatenate(outputs, axis=2)
