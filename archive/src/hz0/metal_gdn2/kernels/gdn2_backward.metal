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

// MARK: - Erase Backward Kernel
// erase_value = sum_k(state * erase * key)
// d_state += d_erase_value * (erase * key)
// d_erase += sum_dv(d_erase_value * state * key)
// d_key += sum_dv(d_erase_value * state * erase)

kernel void gdn2_erase_backward(
    device float* d_erase_value [[ buffer(0) ]],  // [B, H, Dv]
    device float* state [[ buffer(1) ]],          // [B, H, Dv, Dk]
    device float* erase [[ buffer(2) ]],          // [B, H, Dk]
    device float* key [[ buffer(3) ]],            // [B, H, Dk]
    device float* d_state_out [[ buffer(4) ]],    // [B, H, Dv, Dk]
    device float* d_erase_out [[ buffer(5) ]],    // [B, H, Dk]
    device float* d_key_out [[ buffer(6) ]],      // [B, H, Dk]
    constant GDN2BackwardParams& params [[ buffer(7) ]],
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 tid [[ thread_position_in_threadgroup ]],
    uint3 threads_per_threadgroup [[ threads_per_threadgroup ]]
) {
    uint b_h_idx = gid.x;
    uint b = b_h_idx / params.num_heads;
    uint h = b_h_idx % params.num_heads;
    uint dv_dk_idx = tid.x;

    if (b >= params.batch_size || h >= params.num_heads || dv_dk_idx >= params.dv * params.dk) {
        return;
    }

    uint dv_idx = dv_dk_idx / params.dk;
    uint dk_idx = dv_dk_idx % params.dk;

    uint base_offset = (b * params.num_heads + h) * params.dv * params.dk;
    uint flat_idx = base_offset + dv_idx * params.dk + dk_idx;
    uint erase_idx = (b * params.num_heads + h) * params.dk + dk_idx;
    uint d_erase_val_idx = (b * params.num_heads + h) * params.dv + dv_idx;

    float d_ev = d_erase_value[d_erase_val_idx];
    float er = erase[erase_idx];
    float k = key[erase_idx];
    float s = state[flat_idx];

    // d_state += d_erase_value * erase * key
    d_state_out[flat_idx] += d_ev * er * k;

    // d_erase += d_erase_value * state * key
    d_erase_out[erase_idx] += d_ev * s * k;

    // d_key += d_erase_value * state * erase
    d_key_out[erase_idx] += d_ev * s * er;
}

// MARK: - Update Backward Kernel
// state = state - erase_update + write_update
// erase_update = erase_value * key
// write_update = (write * value) * key

kernel void gdn2_update_backward(
    device float* d_state_after [[ buffer(0) ]],  // [B, H, Dv, Dk]
    device float* erase_value [[ buffer(1) ]],    // [B, H, Dv]
    device float* write [[ buffer(2) ]],          // [B, H, Dv]
    device float* value [[ buffer(3) ]],          // [B, H, Dv]
    device float* key [[ buffer(4) ]],            // [B, H, Dk]
    device float* d_erase_value_out [[ buffer(5) ]],  // [B, H, Dv]
    device float* d_write_out [[ buffer(6) ]],    // [B, H, Dv]
    device float* d_value_out [[ buffer(7) ]],    // [B, H, Dv]
    device float* d_key_out [[ buffer(8) ]],      // [B, H, Dk]
    constant GDN2BackwardParams& params [[ buffer(9) ]],
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 tid [[ thread_position_in_threadgroup ]],
    uint3 threads_per_threadgroup [[ threads_per_threadgroup ]]
) {
    uint b_h_idx = gid.x;
    uint b = b_h_idx / params.num_heads;
    uint h = b_h_idx % params.num_heads;
    uint dv_dk_idx = tid.x;

    if (b >= params.batch_size || h >= params.num_heads || dv_dk_idx >= params.dv * params.dk) {
        return;
    }

    uint dv_idx = dv_dk_idx / params.dk;
    uint dk_idx = dv_dk_idx % params.dk;

    uint base_offset = (b * params.num_heads + h) * params.dv * params.dk;
    uint flat_idx = base_offset + dv_idx * params.dk + dk_idx;
    uint key_idx = (b * params.num_heads + h) * params.dk + dk_idx;
    uint ev_idx = (b * params.num_heads + h) * params.dv + dv_idx;

    float d_sa = d_state_after[flat_idx];
    float ev = erase_value[ev_idx];
    float w = write[ev_idx];
    float v = value[ev_idx];
    float k = key[key_idx];

    // Erase contribution: state -= erase_value * key
    // d_erase_value -= sum_k(d_state_after * key)
    d_erase_value_out[ev_idx] -= d_sa * k;

    // d_key -= (erase_value + write * value) * d_state_after
    d_key_out[key_idx] -= d_sa * (ev + w * v);

    // Write contribution: state += (write * value) * key
    // d_write += sum_k(d_state_after * value * key)
    d_write_out[ev_idx] += d_sa * v * k;

    // d_value += sum_k(d_state_after * write * key)
    d_value_out[ev_idx] += d_sa * w * k;
}

// MARK: - Full Backward Pipeline
// Compose all stages in sequence

kernel void gdn2_full_backward(
    device float* d_output [[ buffer(0) ]],      // Loss gradient [B, H, Dv]
    device float* state_in [[ buffer(1) ]],      // Input state [B, H, Dv, Dk]
    device float* query [[ buffer(2) ]],         // [B, H, Dk]
    device float* key [[ buffer(3) ]],           // [B, H, Dk]
    device float* value [[ buffer(4) ]],         // [B, H, Dv]
    device float* decay [[ buffer(5) ]],         // [B, H, Dk]
    device float* erase [[ buffer(6) ]],         // [B, H, Dk]
    device float* write [[ buffer(7) ]],         // [B, H, Dv]
    // Outputs
    device float* d_state_out [[ buffer(8) ]],
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
    // Full pipeline composition:
    // 1. Query backward (compute d_query from d_output)
    // 2. Decay backward (propagate gradients through decay)
    // 3. Erase backward (compute d_erase_value, d_erase, d_key)
    // 4. Update backward (compute d_write, d_value, accumulate d_key)

    // Note: In production, this would be split into multiple passes
    // or fused into a single coherent kernel for efficiency.
    // For now, this skeleton establishes the algorithm structure.

    uint b_h_idx = gid.x;
    if (b_h_idx >= params.batch_size * params.num_heads) {
        return;
    }

    // Placeholder: each stage would execute here
    // Query backward: d_query = sum_dv(d_output * state)
    // Then propagate through remaining stages
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
