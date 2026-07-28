"""MLX bridge for the HZ-0A native Metal GDN-2 forward primitive."""

from __future__ import annotations

import mlx.core as mx


_SOURCE = r"""
    uint tid = thread_position_in_grid.x;
    uint key = tid % K;
    uint value = (tid / K) % V;
    uint head = (tid / (K * V)) % H;
    uint batch = tid / (K * V * H);
    if (batch >= B) return;

    uint state_base = ((batch * H + head) * V + value) * K;
    for (uint t = 0; t < S; ++t) {
        uint row = ((batch * S + t) * H + head) * V + value;
        float decay = hz_sigmoid(d[row]);
        float erase = hz_sigmoid(e[row]);
        float write = hz_sigmoid(w[row]);
        float old_state = initial[state_base + key];
        for (uint u = 0; u < t; ++u) {
            uint prior = ((batch * S + u) * H + head) * V + value;
            old_state = hz_sigmoid(d[prior]) * (1.0f - hz_sigmoid(e[prior])) * old_state
                + hz_sigmoid(w[prior]) * v[prior] * k[((batch * S + u) * H + head) * K + key];
        }
        float next = decay * (1.0f - erase) * old_state
            + write * v[row] * k[((batch * S + t) * H + head) * K + key];
        if (key == 0) {
            float output = 0.0f;
            for (uint j = 0; j < K; ++j) {
                float state_j = initial[state_base + j];
                for (uint u = 0; u <= t; ++u) {
                    uint prior = ((batch * S + u) * H + head) * V + value;
                    float prior_decay = hz_sigmoid(d[prior]);
                    float prior_erase = hz_sigmoid(e[prior]);
                    float prior_write = hz_sigmoid(w[prior]);
                    if (u == 0) {
                        state_j = prior_decay * (1.0f - prior_erase) * state_j
                            + prior_write * v[prior] * k[((batch * S + u) * H + head) * K + j];
                    } else {
                        state_j = prior_decay * (1.0f - prior_erase) * state_j
                            + prior_write * v[prior] * k[((batch * S + u) * H + head) * K + j];
                    }
                }
                output += state_j * q[((batch * S + t) * H + head) * K + j];
            }
            y[((batch * S + t) * H + head) * V + value] = output;
        }
        final_state[state_base + key] = next;
    }
"""


def _reference_forward(q, k, v, d, e, w, initial):
    d, e, w = (mx.sigmoid(item) for item in (d, e, w))
    state = initial
    outputs = []
    for t in range(q.shape[1]):
        state = d[:, t, :, :, None] * (1 - e[:, t, :, :, None]) * state
        state = state + w[:, t, :, :, None] * v[:, t, :, :, None] * k[:, t, :, None, :]
        outputs.append(mx.sum(state * q[:, t, :, None, :], axis=-1))
    return mx.stack(outputs, axis=1), state


@mx.custom_function
def native_gdn2_forward_differentiable(q, k, v, d, e, w, initial):
    """Native Metal forward with an MLX recurrence VJP during bring-up."""
    return tuple(native_gdn2_forward(q, k, v, d, e, w, initial))


@native_gdn2_forward_differentiable.vjp
def _native_gdn2_vjp(primals, cotangents, outputs):
    _, gradients = mx.vjp(_reference_forward, list(primals), list(cotangents))
    return gradients


def native_gdn2_forward(q, k, v, d, e, w, initial):
    """Run the recurrence on Metal and return ``(output, final_state)``.

    Inputs use ``[batch, time, heads, channels]`` and state uses
    ``[batch, heads, value_channels, key_channels]``. This wrapper is
    intentionally forward-only until the native cached backward is bridged.
    """
    bsz, steps, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape != (bsz, steps, heads, key_dim):
        raise ValueError("key shape must match q")
    if d.shape != (bsz, steps, heads, value_dim) or e.shape != d.shape or w.shape != d.shape:
        raise ValueError("gate shapes must match value channels")
    if initial.shape != (bsz, heads, value_dim, key_dim):
        raise ValueError("initial state shape mismatch")
    kernel = mx.fast.metal_kernel(
        name="hz0a_gdn2_forward_mlx",
        input_names=["q", "k", "v", "d", "e", "w", "initial"],
        output_names=["y", "final_state"],
        source=_SOURCE,
        header="#include <metal_stdlib>\nusing namespace metal;\nfloat hz_sigmoid(float x) { return 1.0f / (1.0f + metal::exp(-x)); }\n",
    )
    outputs = kernel(
        inputs=[q, k, v, d, e, w, initial],
        template=[("DType", q.dtype), ("B", bsz), ("S", steps), ("H", heads), ("K", key_dim), ("V", value_dim)],
        grid=(bsz * heads * value_dim * key_dim, 1, 1),
        threadgroup=(min(256, bsz * heads * value_dim * key_dim), 1, 1),
        output_shapes=[(bsz, steps, heads, value_dim), initial.shape],
        output_dtypes=[q.dtype, initial.dtype],
    )
    return outputs
