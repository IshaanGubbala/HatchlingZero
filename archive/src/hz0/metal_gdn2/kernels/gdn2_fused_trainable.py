"""
Trainable wrapper for the fused-Metal GDN-2 forward kernel.

Uses @mx.custom_function with a user-defined VJP so that:
- Forward: runs the fast fused-Metal single-kernel sequence
- Backward: runs the proven MLX reference step-by-step backward

The VJP replays the forward token-by-token to collect per-token
states, then sweeps backward computing exact gradients for all 7 inputs.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import mlx.core as mx

_LOG = logging.getLogger("hz0.metal_gdn2.fused_trainable")


# ---------------------------------------------------------------------------
# 1.  Define the custom function
# ---------------------------------------------------------------------------

@mx.custom_function
def gdn2_fused_trainable_impl(
    q: mx.array,       # [B, T, H, Dk]
    k: mx.array,       # [B, T, H, Dk]
    v: mx.array,       # [B, T, H, Dv]
    d: mx.array,       # [B, T, H, Dk]  post-sigmoid decay
    e: mx.array,       # [B, T, H, Dk]  post-sigmoid erase
    w: mx.array,       # [B, T, H, Dv]  post-sigmoid write
    state0: mx.array,  # [B, H, Dv, Dk]
) -> Tuple[mx.array, mx.array]:
    """Fused-Metal forward (detached from MLX autograd graph).

    Because this is a @mx.custom_function, MLX will NOT try to differentiate
    through the body — it will call the registered VJP instead.
    """
    from hz0.metal_gdn2.kernels.gdn2_fused_metal import gdn2_fused_forward
    return gdn2_fused_forward(q, k, v, d, e, w, initial_state=state0)


# ---------------------------------------------------------------------------
# 2.  Register the VJP — gradients flow through manual backward sweep
# ---------------------------------------------------------------------------

@gdn2_fused_trainable_impl.vjp
def _gdn2_fused_trainable_vjp(
    primals,
    cotangents,
    outputs,
):
    """
    VJP for the fused-Metal GDN-2 forward.

    Forward at each token t:
        raw1   = s_pre * d_t
        s_clip = clip(raw1)
        erase_val = sum_k (s_clip * e_t * k_t)          # [B,H,Dv]
        raw2   = s_clip - erase_val * k_t + (w_t*v_t)*k_t
        s_new  = clip(raw2)
        out_t  = sum_k (s_new * q_t)

    Backward recomputes each step and propagates gradients through
    both clip operations.
    """
    q, k, v, d, e, w, state0 = primals
    grad_output, grad_final_state = cotangents

    B, T, H, Dk = q.shape
    _, _, _, Dv = v.shape

    # ------------------------------------------------------------------
    # Phase A — replay forward to collect per-token states
    # ------------------------------------------------------------------
    s_pre_list: List[mx.array] = []   # state before each token  (length T)
    s_clip_list: List[mx.array] = []  # state after decay+clip   (length T)
    raw2_list: List[mx.array] = []    # pre-clip after erase+write (length T)
    s_new_list: List[mx.array] = []   # state after erase+write+clip (length T)

    state = state0
    for t in range(T):
        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        d_t = d[:, t]
        e_t = e[:, t]
        w_t = w[:, t]

        s_pre = state
        raw1 = s_pre * d_t[:, :, None, :]
        s_clip = mx.clip(raw1, -100.0, 100.0)

        erase_key = e_t * k_t                               # [B, H, Dk]
        erase_partial = s_clip * erase_key[:, :, None, :]   # [B, H, Dv, Dk]
        erase_val = mx.sum(erase_partial, axis=-1)          # [B, H, Dv]

        raw2 = (
            s_clip
            - erase_val[:, :, :, None] * k_t[:, :, None, :]
            + (w_t * v_t)[:, :, :, None] * k_t[:, :, None, :]
        )
        s_new = mx.clip(raw2, -100.0, 100.0)

        s_pre_list.append(s_pre)
        s_clip_list.append(s_clip)
        raw2_list.append(raw2)
        s_new_list.append(s_new)

        state = s_new

    # ------------------------------------------------------------------
    # Phase B — backward sweep
    # ------------------------------------------------------------------
    gq_list: List[mx.array] = []
    gk_list: List[mx.array] = []
    gv_list: List[mx.array] = []
    gd_list: List[mx.array] = []
    ge_list: List[mx.array] = []
    gw_list: List[mx.array] = []

    # gs propagates backward through the recurrent chain.
    # Initially comes from the final-state cotangent (or zeros).
    gs = grad_final_state if grad_final_state is not None else mx.zeros_like(state0)

    for t in reversed(range(T)):
        s_pre  = s_pre_list[t]
        s_clip = s_clip_list[t]
        raw2   = raw2_list[t]
        s_new  = s_new_list[t]

        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        d_t = d[:, t]
        e_t = e[:, t]
        w_t = w[:, t]

        g_out_t = grad_output[:, t]  # [B, H, Dv]

        # --- clip masks ---
        clip_mask1 = (s_pre * d_t[:, :, None, :] > -100.0) & \
                     (s_pre * d_t[:, :, None, :] < 100.0)
        clip_mask2 = (raw2 > -100.0) & (raw2 < 100.0)

        # --- grad for q_t ---
        # out_t = sum_k(s_new * q_t) => gq_t = sum over Dv of g_out * s_new
        gq_t = mx.sum(g_out_t[:, :, :, None] * s_new, axis=2)  # [B, H, Dk]

        # --- grad for s_new (from output) ---
        g_s_new_out = g_out_t[:, :, :, None] * q_t[:, :, None, :]  # [B, H, Dv, Dk]

        # --- combined grad for raw2 (through clip2) ---
        g_raw2 = (gs + g_s_new_out) * clip_mask2  # [B, H, Dv, Dk]

        # --- erase_val ---
        # erase_val = sum_k(s_clip * e * k) => [B, H, Dv]
        erase_val_t = mx.sum(s_clip * e_t[:, :, None, :] * k_t[:, :, None, :], axis=-1)

        # --- grad for e_t ---
        # d(erase_val)/d(e[b,h,dk]) = s_clip * k at same dk
        # d(-erase_val*k)/d(e) chain: g_raw2 -> erase_val -> e
        g_erase_val = g_raw2 * k_t[:, :, None, :]              # [B, H, Dv, Dk]
        g_erase_val_sum = mx.sum(g_erase_val, axis=-1)         # [B, H, Dv]
        ge_t = mx.sum(
            g_erase_val_sum[:, :, :, None] * s_clip * k_t[:, :, None, :],
            axis=2,
        )  # [B, H, Dk]

        # --- grad for v_t ---
        # raw2 += (w*v) * k => d(raw2)/d(v) = w * k
        gv_t = mx.sum(
            g_raw2 * w_t[:, :, :, None] * k_t[:, :, None, :],
            axis=3,
        )  # [B, H, Dv]

        # --- grad for w_t ---
        # raw2 += (w*v) * k => d(raw2)/d(w) = v * k
        gw_t = mx.sum(
            g_raw2 * v_t[:, :, :, None] * k_t[:, :, None, :],
            axis=3,
        )  # [B, H, Dv]

        # --- grad for k_t ---
        # From raw2: d(-erase_val * k)/d(k) = -erase_val  (per dv)
        #           d((w*v)*k)/d(k) = w*v                 (per dv)
        # From erase_val: d(erase_val)/d(k) = s_clip * e  (per dv,dk)
        g_k_from_raw2 = mx.sum(
            g_raw2 * (-erase_val_t[:, :, :, None] + (w_t * v_t)[:, :, :, None]),
            axis=2,
        )  # [B, H, Dk]
        g_k_from_erase = -mx.sum(
            g_erase_val_sum[:, :, :, None] * s_clip * e_t[:, :, None, :],
            axis=2,
        )  # [B, H, Dk]  (negative: chain through erase_val)
        gk_t = g_k_from_raw2 + g_k_from_erase

        # --- grad for d_t ---
        # raw1 = s_pre * d_t => d(clip(raw1))/d(d_t) = s_pre * mask
        # d_t is [B, H, Dk], state is [B, H, Dv, Dk]
        # Sum over Dv to get [B, H, Dk]
        gd_t = mx.sum(g_raw2 * s_pre * clip_mask1, axis=2)  # [B, H, Dk]

        # --- propagate gs to previous state ---
        # d(clip(raw1))/d(s_pre) = d_t * mask
        gs = g_raw2 * d_t[:, :, None, :] * clip_mask1  # [B, H, Dv, Dk]

        gq_list.append(gq_t)
        gk_list.append(gk_t)
        gv_list.append(gv_t)
        gd_list.append(gd_t)
        ge_list.append(ge_t)
        gw_list.append(gw_t)

    # Reverse lists (they were appended in reverse order) and stack
    gq = mx.stack(list(reversed(gq_list)), axis=1)  # [B, T, H, Dk]
    gk = mx.stack(list(reversed(gk_list)), axis=1)
    gv = mx.stack(list(reversed(gv_list)), axis=1)
    gd = mx.stack(list(reversed(gd_list)), axis=1)
    ge = mx.stack(list(reversed(ge_list)), axis=1)
    gw = mx.stack(list(reversed(gw_list)), axis=1)

    return gq, gk, gv, gd, ge, gw, gs


# ---------------------------------------------------------------------------
# 3.  Public API
# ---------------------------------------------------------------------------

def gdn2_fused_trainable(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    d: mx.array,
    e: mx.array,
    w: mx.array,
    initial_state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Drop-in replacement for gdn2_sequence_ops that uses the fused Metal
    forward and manual backward.

    All gates (d, e, w) must already be sigmoid-activated by the caller.
    """
    B, T, H, Dk = q.shape
    _, _, _, Dv = v.shape

    if initial_state is None:
        initial_state = mx.zeros((B, H, Dv, Dk), dtype=q.dtype)

    return gdn2_fused_trainable_impl(q, k, v, d, e, w, initial_state)


__all__ = ["gdn2_fused_trainable"]
