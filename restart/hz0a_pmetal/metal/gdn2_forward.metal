#include <metal_stdlib>
using namespace metal;

// One thread owns one (batch, head, value) row and walks time sequentially.
kernel void hz0a_gdn2_forward(
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device const float *decay [[buffer(3)]],
    device const float *erase [[buffer(4)]], device const float *write [[buffer(5)]],
    device const float *initial_state [[buffer(6)]], device float *output [[buffer(7)]],
    device float *final_state [[buffer(8)]], constant uint &batch_size [[buffer(9)]],
    constant uint &sequence_length [[buffer(10)]], constant uint &heads [[buffer(11)]],
    constant uint &value_dim [[buffer(12)]], constant uint &key_dim [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.x, h = gid.y, value = gid.z;
    if (b >= batch_size || h >= heads || value >= value_dim || key_dim > 64) return;
    thread float state[64];
    for (uint key = 0; key < key_dim; ++key)
        state[key] = initial_state[((b * heads + h) * value_dim + value) * key_dim + key];
    for (uint t = 0; t < sequence_length; ++t) {
        uint input_base = ((b * sequence_length + t) * heads + h) * key_dim;
        uint value_base = ((b * sequence_length + t) * heads + h) * value_dim;
        float readout = 0.0f;
        for (uint key = 0; key < key_dim; ++key) {
            state[key] = decay[input_base + key] * (1.0f - erase[input_base + key]) * state[key]
                + write[value_base + value] * v[value_base + value] * k[input_base + key];
            readout += state[key] * q[input_base + key];
        }
        output[((b * sequence_length + t) * heads + h) * value_dim + value] = readout;
    }
    for (uint key = 0; key < key_dim; ++key)
        final_state[((b * heads + h) * value_dim + value) * key_dim + key] = state[key];
}
