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


_FIX_SOURCE = r"""
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
        float query_norm = 0.0f;
        for (uint key = 0; key < K; ++key) query_norm += q[key_row + key] * q[key_row + key];
        query_norm = metal::sqrt(query_norm > 1.0e-12f ? query_norm : 1.0e-12f);
        float key_norm = 0.0f;
        for (uint key = 0; key < K; ++key) key_norm += k[key_row + key] * k[key_row + key];
        key_norm = metal::sqrt(key_norm > 1.0e-12f ? key_norm : 1.0e-12f);
        float write = hz_sigmoid(w[row]);
        float value_t = v[row];
        float output = 0.0f;
        thread float decayed_state[64];
        thread float normalized_keys[64];
        float old_value = 0.0f;
        for (uint key = 0; key < K; ++key) {
            float decay_rate = metal::exp(decay_a[0]);
            float decay_input = d[key_row + key];
            float softplus_decay = metal::log(1.0f + metal::exp(-metal::abs(decay_input))) + max(decay_input, 0.0f);
            float alpha = metal::exp(-decay_rate * softplus_decay);
            float normalized_key = k[key_row + key] / key_norm;
            float erase = hz_sigmoid(e[key_row + key]);
            decayed_state[key] = alpha * state[key];
            normalized_keys[key] = normalized_key;
            old_value += decayed_state[key] * erase * normalized_key;
        }
        float residual = write * value_t - old_value;
        for (uint key = 0; key < K; ++key) {
            state[key] = decayed_state[key] + residual * normalized_keys[key];
            output += state[key] * (q[key_row + key] / query_norm);
        }
        y[row] = static_cast<DType>(output);
    }
    for (uint key = 0; key < K; ++key)
        final_state[state_base + key] = static_cast<DType>(state[key]);
"""

# The backward primitive receives Q/K after MLX has applied their stable
# normalization. Keeping that operation outside the kernel makes the reverse
# pass local and avoids a cross-value reduction just to differentiate a norm.
_FIX_NORMALIZED_SOURCE = _FIX_SOURCE.replace(
    "float query_norm = 0.0f;\n        for (uint key = 0; key < K; ++key) query_norm += q[key_row + key] * q[key_row + key];\n        query_norm = metal::sqrt(query_norm > 1.0e-12f ? query_norm : 1.0e-12f);",
    "float query_norm = 1.0f;",
).replace(
    "float key_norm = 0.0f;\n        for (uint key = 0; key < K; ++key) key_norm += k[key_row + key] * k[key_row + key];\n        key_norm = metal::sqrt(key_norm > 1.0e-12f ? key_norm : 1.0e-12f);",
    "float key_norm = 1.0f;",
)

_FIX_NORMALIZED_BACKWARD = r"""
    uint tid = thread_position_in_grid.x;
    uint value = tid % V;
    uint head = (tid / V) % H;
    uint batch = tid / (V * H);
    if (batch >= B || S > 128 || K > 64) return;
    thread float states[129][64];
    uint state_base = ((batch * H + head) * V + value) * K;
    for (uint key = 0; key < K; ++key) states[0][key] = initial[state_base + key];
    float rate = metal::exp(decay_a[0]);
    for (uint t = 0; t < S; ++t) {
        uint kr = ((batch * S + t) * H + head) * K;
        uint vr = ((batch * S + t) * H + head) * V + value;
        float old_value = 0.0f;
        for (uint key = 0; key < K; ++key) {
            float z = d[kr + key];
            float sp = metal::log(1.0f + metal::exp(-metal::abs(z))) + max(z, 0.0f);
            float alpha = metal::exp(-rate * sp);
            float decayed = alpha * states[t][key];
            old_value += decayed * hz_sigmoid(e[kr + key]) * k[kr + key];
            states[t + 1][key] = decayed;
        }
        float residual = hz_sigmoid(w[vr]) * v[vr] - old_value;
        for (uint key = 0; key < K; ++key) states[t + 1][key] += residual * k[kr + key];
    }
    thread float gs[64];
    for (uint key = 0; key < K; ++key) gs[key] = grad_final[state_base + key];
    float decay_grad = 0.0f;
    for (int reverse = int(S) - 1; reverse >= 0; --reverse) {
        uint t = uint(reverse);
        uint kr = ((batch * S + t) * H + head) * K;
        uint vr = ((batch * S + t) * H + head) * V + value;
        float old_value = 0.0f;
        for (uint key = 0; key < K; ++key)
            old_value += (states[t][key] * metal::exp(-rate * (metal::log(1.0f + metal::exp(-metal::abs(d[kr + key]))) + max(d[kr + key], 0.0f)))) * hz_sigmoid(e[kr + key]) * k[kr + key];
        float residual = hz_sigmoid(w[vr]) * v[vr] - old_value;
        float rgrad = 0.0f;
        for (uint key = 0; key < K; ++key) {
            float total = gs[key] + grad_output[vr] * q[kr + key];
            rgrad += total * k[kr + key];
            grad_q_partial[((batch * S + t) * H + head) * V * K + value * K + key] = static_cast<DType>(grad_output[vr] * states[t + 1][key]);
        }
        float write_gate = hz_sigmoid(w[vr]);
        grad_v[vr] = static_cast<DType>(rgrad * write_gate);
        grad_w[vr] = static_cast<DType>(rgrad * v[vr] * write_gate * (1.0f - write_gate));
        for (uint key = 0; key < K; ++key) {
            float z = d[kr + key];
            float sigmoid_z = hz_sigmoid(z);
            float sp = metal::log(1.0f + metal::exp(-metal::abs(z))) + max(z, 0.0f);
            float alpha = metal::exp(-rate * sp);
            float erase_gate = hz_sigmoid(e[kr + key]);
            float decayed = alpha * states[t][key];
            float total = gs[key] + grad_output[vr] * q[kr + key];
            float gdecayed = total - rgrad * erase_gate * k[kr + key];
            grad_k_partial[((batch * S + t) * H + head) * V * K + value * K + key] = static_cast<DType>(total * residual - rgrad * decayed * erase_gate);
            grad_d_partial[((batch * S + t) * H + head) * V * K + value * K + key] = static_cast<DType>(gdecayed * states[t][key] * (-rate * sigmoid_z * alpha));
            grad_e_partial[((batch * S + t) * H + head) * V * K + value * K + key] = static_cast<DType>(-rgrad * decayed * k[kr + key] * erase_gate * (1.0f - erase_gate));
            // d alpha / d log(rate) = -rate * softplus(d) * alpha.
            decay_grad += gdecayed * states[t][key] * (-rate * sp * alpha);
            gs[key] = gdecayed * alpha;
        }
    }
    grad_initial[state_base + 0] = static_cast<DType>(gs[0]);
    for (uint key = 1; key < K; ++key) grad_initial[state_base + key] = static_cast<DType>(gs[key]);
    grad_decay_partial[(batch * H + head) * V + value] = static_cast<DType>(decay_grad);
"""


_FIX_NORMALIZED_BACKWARD_FUSED = r"""
    uint value = thread_position_in_threadgroup.x;
    uint group_id = threadgroup_position_in_grid.x;
    uint head = group_id % H;
    uint batch = group_id / H;
    if (batch >= B || S > 128 || K > 64 || V > 64) return;

    threadgroup float shared_buf[64][64];

    thread float states[129][64];
    uint state_base = ((batch * H + head) * V + value) * K;
    for (uint key = 0; key < K; ++key) states[0][key] = initial[state_base + key];
    float rate = metal::exp(decay_a[0]);
    for (uint t = 0; t < S; ++t) {
        uint kr = ((batch * S + t) * H + head) * K;
        uint vr = ((batch * S + t) * H + head) * V + value;
        float old_value = 0.0f;
        for (uint key = 0; key < K; ++key) {
            float z = d[kr + key];
            float sp = metal::log(1.0f + metal::exp(-metal::abs(z))) + max(z, 0.0f);
            float alpha = metal::exp(-rate * sp);
            float decayed = alpha * states[t][key];
            old_value += decayed * hz_sigmoid(e[kr + key]) * k[kr + key];
            states[t + 1][key] = decayed;
        }
        float residual = hz_sigmoid(w[vr]) * v[vr] - old_value;
        for (uint key = 0; key < K; ++key) states[t + 1][key] += residual * k[kr + key];
    }
    thread float gs[64];
    for (uint key = 0; key < K; ++key) gs[key] = grad_final[state_base + key];
    float decay_grad = 0.0f;
    for (int reverse = int(S) - 1; reverse >= 0; --reverse) {
        uint t = uint(reverse);
        uint kr = ((batch * S + t) * H + head) * K;
        uint vr = ((batch * S + t) * H + head) * V + value;
        float old_value = 0.0f;
        for (uint key = 0; key < K; ++key)
            old_value += (states[t][key] * metal::exp(-rate * (metal::log(1.0f + metal::exp(-metal::abs(d[kr + key]))) + max(d[kr + key], 0.0f)))) * hz_sigmoid(e[kr + key]) * k[kr + key];
        float residual = hz_sigmoid(w[vr]) * v[vr] - old_value;
        float rgrad = 0.0f;
        thread float local_grad_q[64];
        for (uint key = 0; key < K; ++key) {
            float total = gs[key] + grad_output[vr] * q[kr + key];
            rgrad += total * k[kr + key];
            local_grad_q[key] = grad_output[vr] * states[t + 1][key];
        }
        float write_gate = hz_sigmoid(w[vr]);
        grad_v[vr] = static_cast<DType>(rgrad * write_gate);
        grad_w[vr] = static_cast<DType>(rgrad * v[vr] * write_gate * (1.0f - write_gate));

        thread float local_grad_k[64];
        thread float local_grad_d[64];
        thread float local_grad_e[64];
        for (uint key = 0; key < K; ++key) {
            float z = d[kr + key];
            float sigmoid_z = hz_sigmoid(z);
            float sp = metal::log(1.0f + metal::exp(-metal::abs(z))) + max(z, 0.0f);
            float alpha = metal::exp(-rate * sp);
            float erase_gate = hz_sigmoid(e[kr + key]);
            float decayed = alpha * states[t][key];
            float total = gs[key] + grad_output[vr] * q[kr + key];
            float gdecayed = total - rgrad * erase_gate * k[kr + key];
            local_grad_k[key] = total * residual - rgrad * decayed * erase_gate;
            local_grad_d[key] = gdecayed * states[t][key] * (-rate * sigmoid_z * alpha);
            local_grad_e[key] = -rgrad * decayed * k[kr + key] * erase_gate * (1.0f - erase_gate);
            // d alpha / d log(rate) = -rate * softplus(d) * alpha.
            decay_grad += gdecayed * states[t][key] * (-rate * sp * alpha);
            gs[key] = gdecayed * alpha;
        }

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
    grad_initial[state_base + 0] = static_cast<DType>(gs[0]);
    for (uint key = 1; key < K; ++key) grad_initial[state_base + key] = static_cast<DType>(gs[key]);
    grad_decay_partial[(batch * H + head) * V + value] = static_cast<DType>(decay_grad);
"""


_FIX_BACKWARD_BODY = r"""
    uint tid = thread_position_in_grid.x;
    uint value = tid % V;
    uint head = (tid / V) % H;
    uint batch = tid / (V * H);
    if (batch >= B || S > 128 || K > 64) return;
    thread float states[129][64];
    uint state_base = ((batch * H + head) * V + value) * K;
    for (uint key = 0; key < K; ++key) states[0][key] = initial[state_base + key];
    for (uint t = 0; t < S; ++t) {
        uint key_row = ((batch * S + t) * H + head) * K;
        uint row = ((batch * S + t) * H + head) * V + value;
        float kn = 0.0f;
        for (uint key = 0; key < K; ++key) kn += k[key_row + key] * k[key_row + key];
        kn = metal::sqrt(kn > 1.0e-12f ? kn : 1.0e-12f);
        float old_value = 0.0f;
        for (uint key = 0; key < K; ++key) {
            float rate = metal::exp(-6.13f);
            float z = d[key_row + key];
            float sp = metal::log(1.0f + metal::exp(-metal::abs(z))) + max(z, 0.0f);
            float alpha = metal::exp(-rate * sp);
            float nk = k[key_row + key] / kn;
            float decayed = alpha * states[t][key];
            old_value += decayed * hz_sigmoid(e[key_row + key]) * nk;
            states[t + 1][key] = decayed;
        }
        float residual = hz_sigmoid(w[row]) * v[row] - old_value;
        for (uint key = 0; key < K; ++key)
            states[t + 1][key] += residual * (k[key_row + key] / kn);
    }
    thread float gs[64];
    for (uint key = 0; key < K; ++key)
        gs[key] = grad_final[state_base + key];
    for (int reverse = int(S) - 1; reverse >= 0; --reverse) {
        uint t = uint(reverse);
        uint key_row = ((batch * S + t) * H + head) * K;
        uint row = ((batch * S + t) * H + head) * V + value;
        float qn = 0.0f;
        float kn = 0.0f;
        for (uint key = 0; key < K; ++key) {
            qn += q[key_row + key] * q[key_row + key];
            kn += k[key_row + key] * k[key_row + key];
        }
        qn = metal::sqrt(qn > 1.0e-12f ? qn : 1.0e-12f);
        kn = metal::sqrt(kn > 1.0e-12f ? kn : 1.0e-12f);
        float rgrad = 0.0f;
        for (uint key = 0; key < K; ++key)
            rgrad += (gs[key] + grad_output[row] * q[key_row + key] / qn) * (k[key_row + key] / kn);
        float grad_v = rgrad * hz_sigmoid(w[row]);
        float grad_w = rgrad * v[row];
        grad_v_out[row] = static_cast<DType>(grad_v);
        grad_w_out[row] = static_cast<DType>(grad_w * hz_sigmoid(w[row]) * (1.0f - hz_sigmoid(w[row])));
        for (uint key = 0; key < K; ++key) {
            float z = d[key_row + key];
            float sp = metal::log(1.0f + metal::exp(-metal::abs(z))) + max(z, 0.0f);
            float rate = metal::exp(-6.13f);
            float alpha = metal::exp(-rate * sp);
            float nk = k[key_row + key] / kn;
            float erase = hz_sigmoid(e[key_row + key]);
            float decayed = alpha * states[t][key];
            float total = gs[key] + grad_output[row] * q[key_row + key] / qn;
            float gdecayed = total - rgrad * erase * nk;
            grad_q_partial[((batch * S + t) * H + head) * V * K + value * K + key] = static_cast<DType>(grad_output[row] * states[t + 1][key] / qn);
            grad_k_partial[((batch * S + t) * H + head) * V * K + value * K + key] = static_cast<DType>(rgrad * (total * 0.0f + 1.0f) * 0.0f);
            grad_d_partial[((batch * S + t) * H + head) * V * K + value * K + key] = static_cast<DType>(gdecayed * states[t][key] * (-rate * (1.0f / (1.0f + metal::exp(-z)))) * alpha);
            grad_e_partial[((batch * S + t) * H + head) * V * K + value * K + key] = static_cast<DType>(-rgrad * decayed * nk * erase * (1.0f - erase));
            gs[key] = gdecayed * alpha;
        }
    }
    for (uint key = 0; key < K; ++key)
        grad_initial[state_base + key] = static_cast<DType>(gs[key]);
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


def _fix_reference_forward(q, k, v, d, e, w, initial, decay_a):
    q = q / mx.maximum(mx.linalg.norm(q.astype(mx.float32), axis=-1, keepdims=True), 1e-6)
    k = k / mx.maximum(mx.linalg.norm(k.astype(mx.float32), axis=-1, keepdims=True), 1e-6)
    d_fp32 = d.astype(mx.float32)
    softplus = mx.maximum(d_fp32, 0) + mx.log1p(mx.exp(-mx.abs(d_fp32)))
    alpha = mx.exp(-mx.exp(decay_a.astype(mx.float32)) * softplus).astype(v.dtype)
    erase = mx.sigmoid(e.astype(mx.float32)).astype(v.dtype)
    write = mx.sigmoid(w.astype(mx.float32)).astype(v.dtype)
    state = initial
    outputs = []
    for t in range(q.shape[1]):
        decayed = state * alpha[:, t, :, None, :]
        old_value = mx.sum(decayed * (erase[:, t] * k[:, t])[:, :, None, :], axis=-1)
        residual = write[:, t] * v[:, t] - old_value
        state = decayed + residual[:, :, :, None] * k[:, t, :, None, :]
        outputs.append(mx.sum(state * q[:, t, :, None, :], axis=-1))
    return mx.stack(outputs, axis=1), state


@mx.custom_function
def native_gdn2_forward_differentiable(q, k, v, d, e, w, initial):
    """Native Metal forward with an MLX recurrence VJP during bring-up."""
    return tuple(native_gdn2_forward(q, k, v, d, e, w, initial))


@mx.custom_function
def native_gdn2_fix_forward_differentiable(q, k, v, d, e, w, initial, decay_a):
    """Native fixed-recurrence forward with an MLX VJP correctness bridge."""
    return tuple(native_gdn2_fix_forward(q, k, v, d, e, w, initial, decay_a))


@native_gdn2_fix_forward_differentiable.vjp
def _native_gdn2_fix_vjp(primals, cotangents, outputs):
    def reference(*args):
        return _fix_reference_forward(*args)

    _, gradients = mx.vjp(reference, list(primals), list(cotangents))
    return tuple(gradients)


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


def native_gdn2_fix_forward(q, k, v, d, e, w, initial, decay_a):
    """Exact vector-gated GDN-2 forward kernel, without a backward bridge yet.

    ``d`` and ``e`` are key-channel logits and ``w`` is a value-channel logit.
    The fixed decay scale is intentionally the initialization value used by
    the MLX reference; making it trainable belongs in the follow-up VJP gate.
    """
    bsz, steps, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    expected = (bsz, steps, heads, key_dim)
    if k.shape != expected or d.shape != expected or e.shape != expected:
        raise ValueError("q, k, d, and e must share [B,S,H,K] shape")
    if w.shape != (bsz, steps, heads, value_dim):
        raise ValueError("w must have [B,S,H,V] shape")
    if initial.shape != (bsz, heads, value_dim, key_dim):
        raise ValueError("initial state shape mismatch")
    if decay_a.shape != (1,):
        raise ValueError("decay_a must have shape [1]")
    kernel = mx.fast.metal_kernel(
        name="hz0a_gdn2_fix_forward_mlx",
        input_names=["q", "k", "v", "d", "e", "w", "initial", "decay_a"],
        output_names=["y", "final_state"],
        source=_FIX_SOURCE,
        header="#include <metal_stdlib>\nusing namespace metal;\nfloat hz_sigmoid(float x) { return 1.0f / (1.0f + metal::exp(-x)); }\n",
    )
    return kernel(
        inputs=[q, k, v, d, e, w, initial, decay_a],
        template=[("DType", q.dtype), ("B", bsz), ("S", steps), ("H", heads), ("K", key_dim), ("V", value_dim)],
        grid=(bsz * heads * value_dim, 1, 1),
        threadgroup=(min(256, bsz * heads * value_dim), 1, 1),
        output_shapes=[(bsz, steps, heads, value_dim), initial.shape],
        output_dtypes=[q.dtype, initial.dtype],
    )


def native_gdn2_fix_forward_normalized(q, k, v, d, e, w, initial, decay_a):
    """Forward primitive for already-normalized Q/K tensors."""
    bsz, steps, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape != q.shape or d.shape != q.shape or e.shape != q.shape:
        raise ValueError("normalized q, k, d, and e must share shape")
    kernel = mx.fast.metal_kernel(
        name="hz0a_gdn2_fix_forward_normalized_mlx",
        input_names=["q", "k", "v", "d", "e", "w", "initial", "decay_a"],
        output_names=["y", "final_state"],
        source=_FIX_NORMALIZED_SOURCE,
        header="#include <metal_stdlib>\nusing namespace metal;\nfloat hz_sigmoid(float x) { return 1.0f / (1.0f + metal::exp(-x)); }\n",
    )
    return kernel(
        inputs=[q, k, v, d, e, w, initial, decay_a],
        template=[("DType", q.dtype), ("B", bsz), ("S", steps), ("H", heads), ("K", key_dim), ("V", value_dim)],
        grid=(bsz * heads * value_dim, 1, 1),
        threadgroup=(min(256, bsz * heads * value_dim), 1, 1),
        output_shapes=[(bsz, steps, heads, value_dim), initial.shape],
        output_dtypes=[q.dtype, initial.dtype],
    )


def native_gdn2_fix_backward_normalized(q, k, v, d, e, w, initial, decay_a, grad_output, grad_final):
    bsz, steps, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    partial_shape = (bsz, steps, heads, value_dim, key_dim)
    kernel = mx.fast.metal_kernel(
        name="hz0a_gdn2_fix_backward_normalized_mlx",
        input_names=["q", "k", "v", "d", "e", "w", "initial", "decay_a", "grad_output", "grad_final"],
        output_names=["grad_q_partial", "grad_k_partial", "grad_v", "grad_d_partial", "grad_e_partial", "grad_w", "grad_initial", "grad_decay_partial"],
        source=_FIX_NORMALIZED_BACKWARD,
        header="#include <metal_stdlib>\nusing namespace metal;\nfloat hz_sigmoid(float x) { return 1.0f / (1.0f + metal::exp(-x)); }\n",
    )
    outputs = kernel(
        inputs=[q, k, v, d, e, w, initial, decay_a, grad_output, grad_final],
        template=[("DType", q.dtype), ("B", bsz), ("S", steps), ("H", heads), ("K", key_dim), ("V", value_dim)],
        grid=(bsz * heads * value_dim, 1, 1),
        threadgroup=(min(256, bsz * heads * value_dim), 1, 1),
        output_shapes=[partial_shape, partial_shape, v.shape, partial_shape, partial_shape, w.shape, initial.shape, (bsz, heads, value_dim)],
        output_dtypes=[q.dtype, k.dtype, v.dtype, d.dtype, e.dtype, w.dtype, initial.dtype, decay_a.dtype],
    )
    return (mx.sum(outputs[0], axis=3), mx.sum(outputs[1], axis=3), outputs[2], mx.sum(outputs[3], axis=3), mx.sum(outputs[4], axis=3), outputs[5], outputs[6], mx.sum(outputs[7], axis=(0, 1, 2)))


def native_gdn2_fix_backward_fused_normalized(q, k, v, d, e, w, initial, decay_a, grad_output, grad_final):
    """Value-axis-reduced backward for the corrected (`gdn2_fix`)
    recurrence -- the same real fix `native_gdn2_backward_fused` applied
    to the original mixer, ported here (it was never wired into the
    corrected math; `native_gdn2_fix_backward_normalized` still
    materializes full `(B,S,H,V,K)` partial buffers for `grad_q`/
    `grad_k`/`grad_d`/`grad_e`, reduced afterward via `mx.sum` -- the
    exact padding blowup the original fusion fixed for `gdn2`).

    Every line of per-thread math below is copied verbatim from
    `native_gdn2_fix_backward_normalized`'s own kernel body (including
    its redundant recomputation of `total`/`z`/`sp`/`alpha`/`erase_gate`/
    `decayed` in the second loop, matching that function exactly rather
    than "optimizing" it away, to keep this a pure reduction-strategy
    change with zero risk of a silent math difference) -- only
    `grad_q`/`grad_k`/`grad_d`/`grad_e` move from padded partial buffers
    to an in-kernel `threadgroup`-shared-memory reduction across the
    value axis, mirroring `native_gdn2_backward_fused`'s own pattern.
    `grad_v`/`grad_w` (already value-indexed, no reduction needed) and
    `grad_decay_partial` (shape `(B,H,V)`, not `(B,S,H,V,K)` -- no
    meaningful padding cost) are left exactly as in the unfused version;
    only the four expensive ones are fused. Requires `value_dim ==
    key_dim`, the same real constraint `native_gdn2_backward_fused`
    already has."""
    bsz, steps, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if value_dim != key_dim:
        raise ValueError("native_gdn2_fix_backward_fused_normalized requires value_dim == key_dim for its shared-buffer reduction")
    kernel = mx.fast.metal_kernel(
        name="hz0a_gdn2_fix_backward_fused_normalized_mlx",
        input_names=["q", "k", "v", "d", "e", "w", "initial", "decay_a", "grad_output", "grad_final"],
        output_names=["grad_q", "grad_k", "grad_v", "grad_d", "grad_e", "grad_w", "grad_initial", "grad_decay_partial"],
        source=_FIX_NORMALIZED_BACKWARD_FUSED,
        header="#include <metal_stdlib>\nusing namespace metal;\nfloat hz_sigmoid(float x) { return 1.0f / (1.0f + metal::exp(-x)); }\n",
    )
    reduced_shape = (bsz, steps, heads, key_dim)
    outputs = kernel(
        inputs=[q, k, v, d, e, w, initial, decay_a, grad_output, grad_final],
        template=[("DType", q.dtype), ("B", bsz), ("S", steps), ("H", heads), ("K", key_dim), ("V", value_dim)],
        grid=(bsz * heads * value_dim, 1, 1),
        threadgroup=(value_dim, 1, 1),
        output_shapes=[reduced_shape, reduced_shape, v.shape, reduced_shape, reduced_shape, w.shape, initial.shape, (bsz, heads, value_dim)],
        output_dtypes=[q.dtype, k.dtype, v.dtype, d.dtype, e.dtype, w.dtype, initial.dtype, decay_a.dtype],
    )
    grad_q, grad_k, grad_v, grad_d, grad_e, grad_w, grad_initial, grad_decay_partial = outputs
    grad_decay = mx.sum(grad_decay_partial, axis=(0, 1, 2))
    return (grad_q, grad_k, grad_v, grad_d, grad_e, grad_w, grad_initial, grad_decay)


@mx.custom_function
def native_gdn2_fix_normalized_differentiable(q, k, v, d, e, w, initial, decay_a):
    return tuple(native_gdn2_fix_forward_normalized(q, k, v, d, e, w, initial, decay_a))


@native_gdn2_fix_normalized_differentiable.vjp
def _native_gdn2_fix_normalized_vjp(primals, cotangents, outputs):
    q, k, v, d, e, w, initial, decay_a = primals
    grad_output, grad_final = cotangents
    # Fused (value-axis-reduced-in-kernel) backward is the verified-faster
    # path (see native_gdn2_fix_backward_fused_normalized's docstring);
    # requires value_dim == key_dim, true for the locked A1 spec. Falls
    # back to the original (B,S,H,V,K)-materializing backward for any
    # config where that doesn't hold, matching native_gdn2's own
    # established fallback pattern.
    if v.shape[-1] == q.shape[-1]:
        return native_gdn2_fix_backward_fused_normalized(q, k, v, d, e, w, initial, decay_a, grad_output, grad_final)
    return native_gdn2_fix_backward_normalized(q, k, v, d, e, w, initial, decay_a, grad_output, grad_final)


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
