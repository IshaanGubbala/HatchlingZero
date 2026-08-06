#include <metal_stdlib>
using namespace metal;

inline float hz0e_silu(float x) {
    return x / (1.0f + exp(-x));
}

// Two-stage expert SwiGLU. A prior single-stage version used one thread
// per (token, out) output scalar, with each thread independently
// recomputing the ENTIRE per-token gate/up hidden activation (an
// O(expert_d_ff/fallback_d_ff * dim) reduction) from scratch -- since
// there are `dim` such threads per token, that meant the SAME hidden
// activation was redundantly recomputed `dim` times per token, an O(dim)
// algorithmic blowup (measured: ~290-440ms/call at the real model's
// tokens=128/dim=768/expert_d_ff=576/fallback_d_ff=2304 scale, for what
// should be well under 10ms of real work). Splitting into two dispatches
// -- stage 1 computes each token's hidden activation ONCE per (token,
// dff-index), stage 2 reduces it down to (token, out) -- removes the
// redundant recompute entirely: total work drops from
// O(tokens * dim * dff * dim) to O(tokens * dim * dff), matching the
// intended budget. A device-side buffer re-upload/residency fix
// (`upload_weights`/`forward_cached`) was tried FIRST and directly
// measured to NOT fix this (isolated forward_cached with fully
// pre-uploaded weights was still ~290ms, identical to the uncached
// path) -- that ruled out buffer upload as the bottleneck and pointed
// here instead. Both stages preserve the exact same accumulation order
// as the original single-stage kernel, so output values are unchanged
// (not just "close"), which is why the same exact-value tests still
// pass unmodified.
//
// `fallback_d_ff` is a REAL, separate hidden width from `expert_d_ff` --
// not assumed equal to `dim`. See git history for the earlier fix that
// made this explicit (a prior version hardcoded the fallback's hidden
// width to `dim`, which does not match `reference/hz0e_moe_contract.py`'s
// real contract).

kernel void hz0e_moe_swiglu_hidden(
    device const float *input [[buffer(0)]],
    device const int *dispatch_slot [[buffer(1)]],
    device const float *expert_weights [[buffer(2)]],
    device const float *expert_biases [[buffer(3)]],
    device const float *fallback_weights [[buffer(4)]],
    device const float *fallback_biases [[buffer(5)]],
    device float *hidden [[buffer(6)]],
    constant uint &capacity [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &expert_d_ff [[buffer(9)]],
    constant uint &tokens [[buffer(10)]],
    constant uint &fallback_d_ff [[buffer(11)]],
    constant uint &max_d_ff [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint token = gid.x;
    uint j = gid.y;
    if (token >= tokens) return;
    int slot = dispatch_slot[token];
    bool fallback = slot < 0;
    uint dff = fallback ? fallback_d_ff : expert_d_ff;
    if (j >= dff) return;
    uint expert = fallback ? 0 : uint(slot) / capacity;
    uint input_base = token * dim;
    float gate = fallback ? fallback_biases[j] : expert_biases[expert * (2 * expert_d_ff + dim) + j];
    float up = fallback ? fallback_biases[fallback_d_ff + j] : expert_biases[expert * (2 * expert_d_ff + dim) + expert_d_ff + j];
    for (uint i = 0; i < dim; ++i) {
        float xv = input[input_base + i];
        if (fallback) {
            gate += fallback_weights[j * dim + i] * xv;
            up += fallback_weights[fallback_d_ff * dim + j * dim + i] * xv;
        } else {
            uint base = expert * (3 * expert_d_ff * dim);
            gate += expert_weights[base + j * dim + i] * xv;
            up += expert_weights[base + expert_d_ff * dim + j * dim + i] * xv;
        }
    }
    hidden[token * max_d_ff + j] = hz0e_silu(gate) * up;
}

kernel void hz0e_moe_swiglu_down(
    device const int *dispatch_slot [[buffer(0)]],
    device const float *expert_weights [[buffer(1)]],
    device const float *expert_biases [[buffer(2)]],
    device const float *fallback_weights [[buffer(3)]],
    device const float *fallback_biases [[buffer(4)]],
    device const float *hidden [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant uint &capacity [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &expert_d_ff [[buffer(9)]],
    constant uint &tokens [[buffer(10)]],
    constant uint &fallback_d_ff [[buffer(11)]],
    constant uint &max_d_ff [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint token = gid.x;
    uint out = gid.y;
    if (token >= tokens || out >= dim) return;
    int slot = dispatch_slot[token];
    bool fallback = slot < 0;
    uint dff = fallback ? fallback_d_ff : expert_d_ff;
    uint expert = fallback ? 0 : uint(slot) / capacity;
    float value = fallback
        ? fallback_biases[2 * fallback_d_ff + out]
        : expert_biases[expert * (2 * expert_d_ff + dim) + 2 * expert_d_ff + out];
    uint hidden_base = token * max_d_ff;
    for (uint j = 0; j < dff; ++j) {
        float h = hidden[hidden_base + j];
        if (fallback) value += fallback_weights[2 * fallback_d_ff * dim + out * fallback_d_ff + j] * h;
        else value += expert_weights[expert * (3 * expert_d_ff * dim) + 2 * expert_d_ff * dim + out * expert_d_ff + j] * h;
    }
    output[token * dim + out] = value;
}
