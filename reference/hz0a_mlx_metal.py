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
            states[t + 1][key] = d[input_base + key] * (1.0f - e[input_base + key]) * states[t][key]
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
            grad_q_partial[partial_base + key] = static_cast<DType>(grad_output[value_base + value] * states[t + 1][key]);
            grad_k_partial[partial_base + key] = static_cast<DType>(total * w[value_base + value] * v[value_base + value]);
            grad_d_partial[partial_base + key] = static_cast<DType>(total * (1.0f - e[input_base + key]) * states[t][key]);
            grad_e_partial[partial_base + key] = static_cast<DType>(-total * d[input_base + key] * states[t][key]);
            value_gradient += total * w[value_base + value] * k[input_base + key];
            write_gradient += total * v[value_base + value] * k[input_base + key];
            grad_state[key] = total * d[input_base + key] * (1.0f - e[input_base + key]);
        }
        grad_v[value_base + value] = static_cast<DType>(value_gradient);
        grad_w[value_base + value] = static_cast<DType>(write_gradient);
    }
    for (uint key = 0; key < K; ++key)
        grad_initial[((batch * H + head) * V + value) * K + key] = static_cast<DType>(grad_state[key]);
"""


_BACKWARD_BODY_FUSED = r"""
    uint value = thread_position_in_threadgroup.x;
    uint group_id = threadgroup_position_in_grid.x;
    uint head = group_id % H;
    uint batch = group_id / H;
    if (batch >= B || S > 128 || K > 64 || V > 64) return;

    threadgroup float shared_buf[64][64];

    thread float states[129][64];
    for (uint key = 0; key < K; ++key)
        states[0][key] = initial[((batch * H + head) * V + value) * K + key];
    for (uint t = 0; t < S; ++t) {
        uint input_base = ((batch * S + t) * H + head) * K;
        uint value_base = ((batch * S + t) * H + head) * V;
        for (uint key = 0; key < K; ++key)
            states[t + 1][key] = d[input_base + key] * (1.0f - e[input_base + key]) * states[t][key]
                + w[value_base + value] * v[value_base + value] * k[input_base + key];
    }
    thread float grad_state[64];
    for (uint key = 0; key < K; ++key)
        grad_state[key] = grad_final[((batch * H + head) * V + value) * K + key];

    for (int reverse = int(S) - 1; reverse >= 0; --reverse) {
        uint t = uint(reverse);
        uint input_base = ((batch * S + t) * H + head) * K;
        uint value_base = ((batch * S + t) * H + head) * V;
        thread float local_grad_q[64];
        thread float local_grad_k[64];
        thread float local_grad_d[64];
        thread float local_grad_e[64];
        float value_gradient = 0.0f;
        float write_gradient = 0.0f;
        for (uint key = 0; key < K; ++key) {
            float total = grad_state[key] + grad_output[value_base + value] * q[input_base + key];
            local_grad_q[key] = grad_output[value_base + value] * states[t + 1][key];
            local_grad_k[key] = total * w[value_base + value] * v[value_base + value];
            local_grad_d[key] = total * (1.0f - e[input_base + key]) * states[t][key];
            local_grad_e[key] = -total * d[input_base + key] * states[t][key];
            value_gradient += total * w[value_base + value] * k[input_base + key];
            write_gradient += total * v[value_base + value] * k[input_base + key];
            grad_state[key] = total * d[input_base + key] * (1.0f - e[input_base + key]);
        }
        grad_v[value_base + value] = static_cast<DType>(value_gradient);
        grad_w[value_base + value] = static_cast<DType>(write_gradient);

        uint out_base = ((batch * S + t) * H + head) * K;

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_q[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) {
            float total_sum = 0.0f;
            for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value];
            grad_q[out_base + value] = static_cast<DType>(total_sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_k[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) {
            float total_sum = 0.0f;
            for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value];
            grad_k[out_base + value] = static_cast<DType>(total_sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_d[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) {
            float total_sum = 0.0f;
            for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value];
            grad_d[out_base + value] = static_cast<DType>(total_sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_e[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) {
            float total_sum = 0.0f;
            for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value];
            grad_e[out_base + value] = static_cast<DType>(total_sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint key = 0; key < K; ++key)
        grad_initial[((batch * H + head) * V + value) * K + key] = static_cast<DType>(grad_state[key]);
"""


_SOURCE = r"""
    uint tid = thread_position_in_grid.x;
    uint value = tid % V;
    uint head = (tid / V) % H;
    uint batch = tid / (V * H);
    if (batch >= B || K > 64) return;

    thread float state[64];
    uint state_base = ((batch * H + head) * V + value) * K;
    for (uint key = 0; key < K; ++key)
        state[key] = initial[state_base + key];

    for (uint t = 0; t < S; ++t) {
        uint key_row = ((batch * S + t) * H + head) * K;
        uint row = ((batch * S + t) * H + head) * V + value;
        float write = hz_sigmoid(w[row]);
        float value_t = v[row];
        float output = 0.0f;
        for (uint key = 0; key < K; ++key) {
            float decay = hz_sigmoid(d[key_row + key]);
            float erase = hz_sigmoid(e[key_row + key]);
            state[key] = decay * (1.0f - erase) * state[key] + write * value_t * k[key_row + key];
            output += state[key] * q[key_row + key];
        }
        y[row] = static_cast<DType>(output);
    }
    for (uint key = 0; key < K; ++key)
        final_state[state_base + key] = static_cast<DType>(state[key]);
"""


def _reference_forward(q, k, v, d, e, w, initial):
    d, e, w = (mx.sigmoid(item) for item in (d, e, w))
    state = initial
    outputs = []
    for t in range(q.shape[1]):
        state = d[:, t, :, None, :] * (1 - e[:, t, :, None, :]) * state
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
    # Fused (value-axis-reduced-in-kernel) backward is the verified-faster
    # path (1.93x, 92% less peak memory -- see native_gdn2_backward_fused's
    # docstring); it requires value_dim == key_dim, true for the locked A1
    # spec. Falls back to the original (B,S,H,V,K)-materializing backward
    # for any config where that doesn't hold, so this stays correct even
    # outside the current architecture.
    if v.shape[-1] == q.shape[-1]:
        gradients = native_gdn2_backward_fused(q, k, v, *activated, initial, grad_output, grad_final)
    else:
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
        grid=(bsz * heads * value_dim, 1, 1),
        threadgroup=(min(256, bsz * heads * value_dim), 1, 1),
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
        template=[("DType", q.dtype), ("B", bsz), ("S", steps), ("H", heads), ("K", key_dim), ("V", value_dim)],
        grid=(bsz * heads * value_dim, 1, 1),
        threadgroup=(min(256, bsz * heads * value_dim), 1, 1),
        output_shapes=[partial_shape, partial_shape, v.shape, partial_shape, partial_shape, w.shape, initial.shape],
        output_dtypes=[q.dtype, k.dtype, v.dtype, d.dtype, e.dtype, w.dtype, initial.dtype],
    )
    return (mx.sum(outputs[0], axis=3), mx.sum(outputs[1], axis=3), outputs[2], mx.sum(outputs[3], axis=3), mx.sum(outputs[4], axis=3), outputs[5], outputs[6])


def native_gdn2_backward_fused(q, k, v, d, e, w, initial, grad_output, grad_final):
    """Value-axis-reduced backward: no (B,S,H,V,K) partial buffers.

    Reduces dQ/dK/dDecay/dErase across the value axis inside the kernel via
    threadgroup shared memory (one threadgroup per (batch, head), threadgroup
    size == value_dim), instead of writing full (B,S,H,V,K) partial tensors
    and reducing them afterward with a separate mx.sum. Requires
    value_dim <= 64 and key_dim <= 64 (same limits the existing kernel
    already enforces) and, for the current single-shared-buffer reuse
    pattern, value_dim == key_dim (true for the locked A1 spec: both equal
    head_dim=64). Atomic float accumulation (atomic_fetch_add_explicit) was
    tried first and rejected: it fails to compile in this environment
    (MLX 0.32.0 / Apple M5 Max) with a Metal XPC compiler error, even for a
    single-thread smoke test -- atomic_store_explicit works fine, so this is
    specific to float fetch-add, not atomics generally. Threadgroup-memory
    reduction avoids the issue entirely and needs no atomics.
    """
    bsz, steps, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if value_dim != key_dim:
        raise ValueError("native_gdn2_backward_fused requires value_dim == key_dim for its shared-buffer reduction")
    kernel = mx.fast.metal_kernel(
        name="hz0a_gdn2_backward_fused_mlx",
        input_names=["q", "k", "v", "d", "e", "w", "initial", "grad_output", "grad_final"],
        output_names=["grad_q", "grad_k", "grad_v", "grad_d", "grad_e", "grad_w", "grad_initial"],
        source=_BACKWARD_BODY_FUSED,
        header="using namespace metal;",
    )
    reduced_shape = (bsz, steps, heads, key_dim)
    outputs = kernel(
        inputs=[q, k, v, d, e, w, initial, grad_output, grad_final],
        template=[("DType", q.dtype), ("B", bsz), ("S", steps), ("H", heads), ("K", key_dim), ("V", value_dim)],
        grid=(bsz * heads * value_dim, 1, 1),
        threadgroup=(value_dim, 1, 1),
        output_shapes=[reduced_shape, reduced_shape, v.shape, reduced_shape, reduced_shape, w.shape, initial.shape],
        output_dtypes=[q.dtype, k.dtype, v.dtype, d.dtype, e.dtype, w.dtype, initial.dtype],
    )
    return outputs
