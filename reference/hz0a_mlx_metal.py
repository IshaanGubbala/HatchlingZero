"""MLX bridge for the HZ-0A native Metal GDN-2 forward primitive."""

from __future__ import annotations

import mlx.core as mx


_BACKWARD_BODY = r"""
    uint tid = thread_position_in_grid.x;
    uint value = tid % V;
    uint head = (tid / V) % H;
    uint batch = tid / (V * H);
    if (batch >= B || S > 128 || K > 64) return;
    thread float states[129][64];
    for (uint key = 0; key < K; ++key)
        states[0][key] = initial[((batch * H + head) * V + value) * K + key];
    for (uint t = 0; t < S; ++t) {
        uint input_base = ((batch * S + t) * H + head) * K;
        uint value_base = ((batch * S + t) * H + head) * V;
        for (uint key = 0; key < K; ++key)
            states[t + 1][key] = d[value_base + value] * (1.0f - e[value_base + value]) * states[t][key]
                + w[value_base + value] * v[value_base + value] * k[input_base + key];
    }
    thread float grad_state[64];
    for (uint key = 0; key < K; ++key)
        grad_state[key] = grad_final[((batch * H + head) * V + value) * K + key];
    for (int reverse = int(S) - 1; reverse >= 0; --reverse) {
        uint t = uint(reverse);
        uint input_base = ((batch * S + t) * H + head) * K;
        uint value_base = ((batch * S + t) * H + head) * V;
        uint partial_base = (((batch * S + t) * H + head) * V + value) * K;
        float value_gradient = 0.0f;
        float write_gradient = 0.0f;
        for (uint key = 0; key < K; ++key) {
            float total = grad_state[key] + grad_output[value_base + value] * q[input_base + key];
            grad_q_partial[partial_base + key] = grad_output[value_base + value] * states[t + 1][key];
            grad_k_partial[partial_base + key] = total * w[value_base + value] * v[value_base + value];
            grad_d_partial[partial_base + key] = total * (1.0f - e[value_base + value]) * states[t][key];
            grad_e_partial[partial_base + key] = -total * d[value_base + value] * states[t][key];
            value_gradient += total * w[value_base + value] * k[input_base + key];
            write_gradient += total * v[value_base + value] * k[input_base + key];
            grad_state[key] = total * d[value_base + value] * (1.0f - e[value_base + value]);
        }
        grad_v[value_base + value] = value_gradient;
        grad_w[value_base + value] = write_gradient;
    }
    for (uint key = 0; key < K; ++key)
        grad_initial[((batch * H + head) * V + value) * K + key] = grad_state[key];
"""


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
    q, k, v, d, e, w, initial = primals
    grad_output, grad_final = cotangents
    activated = tuple(mx.sigmoid(item) for item in (d, e, w))
    gradients = native_gdn2_backward(q, k, v, *activated, initial, grad_output, grad_final)
    grad_d, grad_e, grad_w = (gradient * gate * (1 - gate) for gradient, gate in zip(gradients[3:6], activated))
    return (gradients[0], gradients[1], gradients[2], grad_d, grad_e, grad_w, gradients[6])


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


def native_gdn2_backward(q, k, v, d, e, w, initial, grad_output, grad_final):
    bsz, steps, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    kernel = mx.fast.metal_kernel(
        name="hz0a_gdn2_backward_mlx",
        input_names=["q", "k", "v", "d", "e", "w", "initial", "grad_output", "grad_final"],
        output_names=["grad_q_partial", "grad_k_partial", "grad_v", "grad_d_partial", "grad_e_partial", "grad_w", "grad_initial"],
        source=_BACKWARD_BODY,
        header="using namespace metal;",
    )
    partial_shape = (bsz, steps, heads, value_dim, key_dim)
    outputs = kernel(
        inputs=[q, k, v, d, e, w, initial, grad_output, grad_final],
        template=[("B", bsz), ("S", steps), ("H", heads), ("K", key_dim), ("V", value_dim)],
        grid=(bsz * heads * value_dim, 1, 1),
        threadgroup=(min(256, bsz * heads * value_dim), 1, 1),
        output_shapes=[partial_shape, partial_shape, v.shape, partial_shape, partial_shape, w.shape, initial.shape],
        output_dtypes=[q.dtype, k.dtype, v.dtype, d.dtype, e.dtype, w.dtype, initial.dtype],
    )
    return (mx.sum(outputs[0], axis=3), mx.sum(outputs[1], axis=3), outputs[2], mx.sum(outputs[3], axis=4), mx.sum(outputs[4], axis=4), outputs[5], outputs[6])
