#include <metal_stdlib>
using namespace metal;

// Fixed-shape top-1 scatter. `grouped_tokens` uses UINT_MAX for padding;
// overflow tokens are intentionally left on the fallback path.
kernel void hz0e_moe_scatter(
    device const int *dispatch_slot [[buffer(0)]],
    device const float *expert_outputs [[buffer(1)]],
    device const float *fallback_outputs [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint token = gid.x;
    uint feature = gid.y;
    if (token >= tokens || feature >= width) return;

    float value = fallback_outputs[token * width + feature];
    int slot = dispatch_slot[token];
    if (slot >= 0) value = expert_outputs[(uint(slot) * width) + feature];
    output[token * width + feature] = value;
}
