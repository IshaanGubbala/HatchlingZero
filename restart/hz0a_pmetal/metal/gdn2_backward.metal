#include <metal_stdlib>
using namespace metal;

// Cached reverse scan for smoke/parity workloads. The caller must keep
// sequence_length <= 128 and key_dim <= 64; production chunking remains the
// mechanism for longer sequences.
kernel void hz0a_gdn2_backward(
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
    device const float *decay [[buffer(3)]], device const float *erase [[buffer(4)]], device const float *write [[buffer(5)]],
    device const float *initial_state [[buffer(6)]], device const float *grad_output [[buffer(7)]],
    device const float *grad_final [[buffer(8)]], device atomic_float *grad_q [[buffer(9)]], device atomic_float *grad_k [[buffer(10)]],
    device float *grad_v [[buffer(11)]], device atomic_float *grad_decay [[buffer(12)]], device atomic_float *grad_erase [[buffer(13)]],
    device float *grad_write [[buffer(14)]], device atomic_float *grad_initial [[buffer(15)]], constant uint &batch_size [[buffer(16)]],
    constant uint &sequence_length [[buffer(17)]], constant uint &heads [[buffer(18)]], constant uint &value_dim [[buffer(19)]],
    constant uint &key_dim [[buffer(20)]], uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.x, h = gid.y, value = gid.z;
    if (b >= batch_size || h >= heads || value >= value_dim || sequence_length > 128 || key_dim > 64) return;
    thread float states[129][64];
    for (uint key = 0; key < key_dim; ++key) {
        uint initial_index = ((b * heads + h) * value_dim + value) * key_dim + key;
        states[0][key] = initial_state[initial_index];
    }
    for (uint t = 0; t < sequence_length; ++t) {
        uint input_base = ((b * sequence_length + t) * heads + h) * key_dim;
        uint value_base = ((b * sequence_length + t) * heads + h) * value_dim;
        for (uint key = 0; key < key_dim; ++key)
            states[t + 1][key] = decay[input_base + key] * (1.0f - erase[input_base + key]) * states[t][key] + write[value_base + value] * v[value_base + value] * k[input_base + key];
    }
    thread float grad_state[64];
    for (uint key = 0; key < key_dim; ++key) {
        uint state_index = ((b * heads + h) * value_dim + value) * key_dim + key;
        grad_state[key] = grad_final[state_index];
    }
    for (int reverse = int(sequence_length) - 1; reverse >= 0; --reverse) {
        uint t = uint(reverse);
        uint input_base = ((b * sequence_length + t) * heads + h) * key_dim;
        uint value_base = ((b * sequence_length + t) * heads + h) * value_dim;
        float value_gradient = 0.0f;
        float write_gradient = 0.0f;
        for (uint key = 0; key < key_dim; ++key) {
            float total = grad_state[key] + grad_output[((b * sequence_length + t) * heads + h) * value_dim + value] * q[input_base + key];
            atomic_fetch_add_explicit(&grad_q[input_base + key], grad_output[((b * sequence_length + t) * heads + h) * value_dim + value] * states[t + 1][key], memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_k[input_base + key], total * write[value_base + value] * v[value_base + value], memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_decay[input_base + key], total * (1.0f - erase[input_base + key]) * states[t][key], memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_erase[input_base + key], -total * decay[input_base + key] * states[t][key], memory_order_relaxed);
            value_gradient += total * write[value_base + value] * k[input_base + key];
            write_gradient += total * v[value_base + value] * k[input_base + key];
            grad_state[key] = total * decay[input_base + key] * (1.0f - erase[input_base + key]);
        }
        grad_v[value_base + value] = value_gradient;
        grad_write[value_base + value] = write_gradient;
    }
    for (uint key = 0; key < key_dim; ++key) {
        uint state_index = ((b * heads + h) * value_dim + value) * key_dim + key;
        atomic_fetch_add_explicit(&grad_initial[state_index], grad_state[key], memory_order_relaxed);
    }
}
