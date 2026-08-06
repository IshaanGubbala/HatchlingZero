#include <metal_stdlib>
using namespace metal;

inline float hz0e_silu(float x) {
    return x / (1.0f + exp(-x));
}

// Reference expert stage: one thread computes one output feature. The
// flattened dispatch slot selects expert = slot / capacity; negative slots
// use the shared fallback. This is intentionally unfused and correctness-first.
//
// `fallback_d_ff` is a REAL, separate hidden width from `expert_d_ff` --
// not assumed equal to `dim`. A prior version of this kernel hardcoded the
// fallback's hidden width to `dim`, which does not match
// `reference/hz0e_moe_contract.py`'s own real contract (the shared
// fallback is FULL dense-FFN width, `dense_d_ff`, e.g. 2304 at the real
// HZ-0A/E model scale -- NOT `dim`, e.g. 768). Using the old, hardcoded
// version against the real model's real fallback shape would have
// silently computed the WRONG overflow-token output, not just run slower
// or faster -- caught before any end-to-end benchmark was built on top of
// this kernel, not after.
kernel void hz0e_moe_swiglu(
    device const float *input [[buffer(0)]],
    device const int *dispatch_slot [[buffer(1)]],
    device const float *expert_weights [[buffer(2)]],
    device const float *expert_biases [[buffer(3)]],
    device const float *fallback_weights [[buffer(4)]],
    device const float *fallback_biases [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant uint &capacity [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &expert_d_ff [[buffer(9)]],
    constant uint &tokens [[buffer(10)]],
    constant uint &fallback_d_ff [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint token = gid.x;
    uint out = gid.y;
    if (token >= tokens || out >= dim) return;
    int slot = dispatch_slot[token];
    bool fallback = slot < 0;
    uint dff = fallback ? fallback_d_ff : expert_d_ff;
    uint expert = fallback ? 0 : uint(slot) / capacity;
    uint input_base = token * dim;
    float value = fallback
        ? fallback_biases[2 * fallback_d_ff + out]
        : expert_biases[expert * (2 * expert_d_ff + dim) + 2 * expert_d_ff + out];
    for (uint j = 0; j < dff; ++j) {
        float gate = fallback ? fallback_biases[j] : expert_biases[expert * (2 * expert_d_ff + dim) + j];
        float up = fallback ? fallback_biases[fallback_d_ff + j] : expert_biases[expert * (2 * expert_d_ff + dim) + expert_d_ff + j];
        for (uint i = 0; i < dim; ++i) {
            if (fallback) {
                gate += fallback_weights[j * dim + i] * input[input_base + i];
                up += fallback_weights[fallback_d_ff * dim + j * dim + i] * input[input_base + i];
            } else {
                uint base = expert * (3 * expert_d_ff * dim);
                gate += expert_weights[base + j * dim + i] * input[input_base + i];
                up += expert_weights[base + expert_d_ff * dim + j * dim + i] * input[input_base + i];
            }
        }
        float hidden = hz0e_silu(gate) * up;
        if (fallback) value += fallback_weights[2 * fallback_d_ff * dim + out * fallback_d_ff + j] * hidden;
        else value += expert_weights[expert * (3 * expert_d_ff * dim) + 2 * expert_d_ff * dim + out * expert_d_ff + j] * hidden;
    }
    output[token * dim + out] = value;
}
