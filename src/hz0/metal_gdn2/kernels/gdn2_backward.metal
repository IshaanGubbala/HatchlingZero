// GDN-2 Backward Pass Kernel
// Metal Shading Language implementation for GPU-accelerated gradient computation

#include <metal_stdlib>
using namespace metal;

// MARK: - Kernel Data Structures

struct GDN2BackwardParams {
    uint batch_size;
    uint num_heads;
    uint dv;  // value dimension
    uint dk;  // key dimension
};

// MARK: - Decay Backward Kernel
// state' = state * decay[:, :, None, :]
// d_state_old = d_state_new * decay
// d_decay = sum_dv(d_state_new * state)

kernel void gdn2_decay_backward(
    device float* d_state_new [[ buffer(0) ]],  // [B, H, Dv, Dk]
    device float* state [[ buffer(1) ]],         // [B, H, Dv, Dk]
    device float* decay [[ buffer(2) ]],         // [B, H, Dk]
    device float* d_state_out [[ buffer(3) ]],   // [B, H, Dv, Dk]
    device float* d_decay_out [[ buffer(4) ]],   // [B, H, Dk]
    constant GDN2BackwardParams& params [[ buffer(5) ]],
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 tid [[ thread_position_in_threadgroup ]],
    uint3 threads_per_threadgroup [[ threads_per_threadgroup ]]
) {
    // Grid layout: (B*H, 1, 1) with Dv threads per threadgroup
    // Process one (b, h) pair per threadgroup

    uint b_h_idx = gid.x;
    uint b = b_h_idx / params.num_heads;
    uint h = b_h_idx % params.num_heads;
    uint dv_idx = tid.x;

    if (b >= params.batch_size || h >= params.num_heads || dv_idx >= params.dv) {
        return;
    }

    // Base offset: b * H * Dv * Dk + h * Dv * Dk + dv * Dk
    uint base_offset = (b * params.num_heads + h) * params.dv * params.dk;

    // d_state = d_state_new * decay
    for (uint dk_idx = 0; dk_idx < params.dk; dk_idx++) {
        uint flat_idx = base_offset + dv_idx * params.dk + dk_idx;
        uint decay_idx = (b * params.num_heads + h) * params.dk + dk_idx;

        float d_sn = d_state_new[flat_idx];
        float dec = decay[decay_idx];

        d_state_out[flat_idx] = d_sn * dec;

        // Accumulate d_decay (sum over Dv)
        // Note: Use atomic operations for thread-safe accumulation
        uint d_decay_idx = (b * params.num_heads + h) * params.dk + dk_idx;
        float contribution = d_sn * state[flat_idx];

        // TODO: Replace with proper atomic float add when available
        d_decay_out[d_decay_idx] += contribution;
    }
}

// MARK: - Query Backward Kernel
// output = sum_k(state * query)
// d_query = sum_dv(d_output * state)

kernel void gdn2_query_backward(
    device float* d_output [[ buffer(0) ]],      // [B, H, Dv]
    device float* state [[ buffer(1) ]],         // [B, H, Dv, Dk]
    device float* d_query_out [[ buffer(2) ]],   // [B, H, Dk]
    constant GDN2BackwardParams& params [[ buffer(3) ]],
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 tid [[ thread_position_in_threadgroup ]],
    uint3 threads_per_threadgroup [[ threads_per_threadgroup ]]
) {
    // Grid: (B*H, 1, 1) with Dk threads
    // Each thread computes d_query for one Dk element

    uint b_h_idx = gid.x;
    uint b = b_h_idx / params.num_heads;
    uint h = b_h_idx % params.num_heads;
    uint dk_idx = tid.x;

    if (b >= params.batch_size || h >= params.num_heads || dk_idx >= params.dk) {
        return;
    }

    // d_query[:, :, k] = sum_dv(d_output[:, :, v] * state[:, :, v, k])
    uint base_state = (b * params.num_heads + h) * params.dv * params.dk;
    uint d_output_base = (b * params.num_heads + h) * params.dv;

    float d_q = 0.0f;

    for (uint dv_idx = 0; dv_idx < params.dv; dv_idx++) {
        uint state_idx = base_state + dv_idx * params.dk + dk_idx;
        uint d_out_idx = d_output_base + dv_idx;

        d_q += d_output[d_out_idx] * state[state_idx];
    }

    // Write result
    uint d_query_idx = (b * params.num_heads + h) * params.dk + dk_idx;
    d_query_out[d_query_idx] = d_q;
}

// MARK: - Full Backward Pipeline
// Compose all stages in sequence

kernel void gdn2_full_backward(
    device float* d_output [[ buffer(0) ]],      // Loss gradient
    device float* state_in [[ buffer(1) ]],      // Input state
    device float* query [[ buffer(2) ]],
    device float* key [[ buffer(3) ]],
    device float* value [[ buffer(4) ]],
    device float* decay [[ buffer(5) ]],
    device float* erase [[ buffer(6) ]],
    device float* write [[ buffer(7) ]],
    // Outputs
    device float* d_state [[ buffer(8) ]],
    device float* d_query_out [[ buffer(9) ]],
    device float* d_key_out [[ buffer(10) ]],
    device float* d_value_out [[ buffer(11) ]],
    device float* d_decay_out [[ buffer(12) ]],
    device float* d_erase_out [[ buffer(13) ]],
    device float* d_write_out [[ buffer(14) ]],
    constant GDN2BackwardParams& params [[ buffer(15) ]],
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 tid [[ thread_position_in_threadgroup ]],
    uint3 threads_per_threadgroup [[ threads_per_threadgroup ]]
) {
    // Placeholder: calls decay and query backward
    // Full implementation will compose all four stages

    // Stage 1: Query backward
    // (Reuse gdn2_query_backward logic)

    // Stage 2: Decay backward
    // (Reuse gdn2_decay_backward logic)

    // Stages 3-4: Erase and update backward
    // (To be implemented)
}

// MARK: - Testing Kernel

kernel void gdn2_backward_test(
    device float* output [[ buffer(0) ]],
    constant GDN2BackwardParams& params [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    // Simple test kernel to verify compilation
    if (gid < 100) {
        output[gid] = float(gid) * 0.5f;
    }
}
