// Metal kernel for streaming GDN-2 single-token forward pass
// Phase 15-1: Hardware-accelerated streaming step

#include <metal_stdlib>
using namespace metal;

// Sigmoid activation (used in gates)
inline float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

/// Streaming GDN-2 step: Single token, accumulated state
///
/// Kernel computes:
/// 1. Decay: state *= decay
/// 2. Erase: state *= (1 - erase)  [key-selective]
/// 3. Write: state += token * write [value update]
/// 4. Query: output = state · query [dot product]
/// 5. Clip: state = clip(state, -100, 100)
///
/// Args:
///   query:   [B, D_k] query projection
///   key:     [B, D_k] key projection
///   value:   [B, D_v] value (token embedding)
///   state:   [B, D_v, D_k] accumulated state
///   decay_logit:  scalar decay gate (pre-sigmoid)
///   erase_logit:  scalar erase gate (pre-sigmoid)
///   write_logit:  scalar write gate (pre-sigmoid)
///   output:  [B, D_v] output buffer (write)
///   state_new: [B, D_v, D_k] updated state (write)
///   gridsize: number of threads
///
kernel void gdn2_step_forward(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device const float *state [[buffer(3)]],
    device const float *decay_logit [[buffer(4)]],
    device const float *erase_logit [[buffer(5)]],
    device const float *write_logit [[buffer(6)]],
    device float *output [[buffer(7)]],
    device float *state_new [[buffer(8)]],
    constant uint &B [[buffer(9)]],        // batch size
    constant uint &D_v [[buffer(10)]],     // value dimension
    constant uint &D_k [[buffer(11)]],     // key dimension
    uint gid [[thread_position_in_grid]]
) {
    // Thread processes one element in output [B, D_v]
    // gid = b * D_v + v_idx

    if (gid >= B * D_v) return;

    uint b = gid / D_v;
    uint v_idx = gid % D_v;

    // Load gates (same for entire batch)
    float decay = sigmoid(decay_logit[0]);      // ∈ (0,1)
    float erase = sigmoid(erase_logit[0]);      // ∈ (0,1)
    float write = sigmoid(write_logit[0]);      // ∈ (0,1)

    // Accumulate output via dot product (state · query)
    float output_val = 0.0;

    // Iterate over key dimension: output[b,v] = sum_k(state[b,v,k] * query[b,k])
    for (uint k_idx = 0; k_idx < D_k; ++k_idx) {
        // State position: [b * D_v * D_k + v_idx * D_k + k_idx]
        uint state_idx = b * D_v * D_k + v_idx * D_k + k_idx;
        uint query_idx = b * D_k + k_idx;

        // Load state and apply decay
        float s_decayed = state[state_idx] * decay;

        // Apply erase gate (key-selective)
        float s_erased = s_decayed * (1.0 - erase);

        // Query contribution
        float q = query[query_idx];
        output_val += s_erased * q;

        // Update state with write gate
        float v = value[b * D_v + v_idx];
        float s_new = s_erased + write * v;

        // Clip to prevent unbounded growth
        s_new = clamp(s_new, -100.0f, 100.0f);

        // Write back to state_new
        state_new[state_idx] = s_new;
    }

    // Write output
    output[gid] = output_val;
}

/// Backward kernel: Gradient computation via VJP
///
/// Computes gradients w.r.t. all inputs given grad_output
/// This is complex - full VJP requires chain rule through all gates
///
kernel void gdn2_step_backward(
    device const float *grad_output [[buffer(0)]],      // [B, D_v] gradient
    device const float *grad_state_in [[buffer(1)]],    // [B, D_v, D_k] gradient
    device const float *query [[buffer(2)]],
    device const float *key [[buffer(3)]],
    device const float *value [[buffer(4)]],
    device const float *state [[buffer(5)]],
    device const float *decay_logit [[buffer(6)]],
    device const float *erase_logit [[buffer(7)]],
    device const float *write_logit [[buffer(8)]],
    device float *grad_query [[buffer(9)]],             // [B, D_k] output
    device float *grad_key [[buffer(10)]],              // [B, D_k] output
    device float *grad_value [[buffer(11)]],            // [B, D_v] output
    device float *grad_state [[buffer(12)]],            // [B, D_v, D_k] output
    device float *grad_decay [[buffer(13)]],            // scalar output
    device float *grad_erase [[buffer(14)]],            // scalar output
    device float *grad_write [[buffer(15)]],            // scalar output
    constant uint &B [[buffer(16)]],
    constant uint &D_v [[buffer(17)]],
    constant uint &D_k [[buffer(18)]],
    uint gid [[thread_position_in_grid]]
) {
    // VJP (Vector-Jacobian Product) computation
    // For each output, compute gradient w.r.t. all inputs

    // This is a placeholder - full VJP is complex
    // Would need to:
    // 1. Compute gradient flow through dot product
    // 2. Compute gradient through gates (sigmoid derivatives)
    // 3. Compute gradient through state accumulation
    // 4. Accumulate gradients for scalar parameters

    if (gid >= B * D_v) return;

    uint b = gid / D_v;
    uint v_idx = gid % D_v;

    // Load gate derivatives: d(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
    float decay = sigmoid(decay_logit[0]);
    float decay_grad = decay * (1.0 - decay);
    float erase = sigmoid(erase_logit[0]);
    float erase_grad = erase * (1.0 - erase);
    float write = sigmoid(write_logit[0]);
    float write_grad = write * (1.0 - write);

    // Gradient accumulation through output dot product
    float grad_out = grad_output[gid];

    for (uint k_idx = 0; k_idx < D_k; ++k_idx) {
        uint state_idx = b * D_v * D_k + v_idx * D_k + k_idx;
        uint query_idx = b * D_k + k_idx;

        float q = query[query_idx];
        float v = value[b * D_v + v_idx];
        float s = state[state_idx];

        // Gradient w.r.t. query: grad_query[b,k] += state_erased[b,v,k] * grad_out[b,v]
        // (simplified - would need state after erase)
        atomic_fetch_add_explicit(&grad_query[query_idx], grad_out * (s * decay), memory_order_relaxed);

        // Gradient w.r.t. state (chain through all operations)
        // grad_state[b,v,k] = grad_out * query[b,k] + grad_state_in[b,v,k]
        grad_state[state_idx] = grad_out * q + grad_state_in[state_idx];
    }

    // Note: Full backward requires atomic operations for scalar gradients
    // and careful chain-rule through all gates. This is simplified.
}
