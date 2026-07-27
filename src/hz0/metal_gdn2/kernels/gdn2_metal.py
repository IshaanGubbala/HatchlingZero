"""
Native Metal kernel for GDN-2.

Adapts MLX-LM's existing Metal GatedDeltaNet structure.
Grid layout:
  - One grid plane = one (batch, head)
  - One grid row = one value channel
  - 32 SIMD lanes over key dimension
"""

import mlx.core as mx
from typing import Tuple, Optional


def gdn2_metal_forward(
    queries: mx.array,    # [B, T, H, Dk]
    keys: mx.array,       # [B, T, H, Dk]
    values: mx.array,     # [B, T, H, Dv]
    decays: mx.array,     # [B, T, H, Dk]
    erases: mx.array,     # [B, T, H, Dk]
    writes: mx.array,     # [B, T, H, Dv]
    initial_state: Optional[mx.array] = None,  # [B, H, Dv, Dk]
) -> Tuple[mx.array, mx.array]:
    """
    GDN-2 forward via Metal.

    For now: direct MLX implementation.
    Production version: compile to mx.fast.metal_kernel with:
      - Fused decay/erase/write
      - SIMD reduction over Dk
      - State residency in registers
    """
    B, T, H, Dk = queries.shape
    _, _, Dv = values.shape[-3:]

    if initial_state is None:
        initial_state = mx.zeros((B, H, Dv, Dk), dtype=queries.dtype)

    state = initial_state
    outputs = []

    for t in range(T):
        # Extract token t
        q_t = queries[:, t]      # [B, H, Dk]
        k_t = keys[:, t]         # [B, H, Dk]
        v_t = values[:, t]       # [B, H, Dv]
        d_t = decays[:, t]       # [B, H, Dk]
        e_t = erases[:, t]       # [B, H, Dk]
        w_t = writes[:, t]       # [B, H, Dv]

        # Decay
        state = state * d_t[:, :, None, :]  # [B, H, Dv, Dk]

        # Erase
        erase_value = mx.sum(state * e_t[:, :, None, :] * k_t[:, :, None, :], axis=-1)  # [B, H, Dv]

        # Write
        state = state - erase_value[:, :, :, None] * k_t[:, :, None, :]
        state = state + (w_t * v_t)[:, :, :, None] * k_t[:, :, None, :]

        # Query
        output = mx.sum(state * q_t[:, :, None, :], axis=-1)  # [B, H, Dv]
        outputs.append(output)

    outputs = mx.stack(outputs, axis=1)  # [B, T, H, Dv]
    return outputs, state


def gdn2_metal_streaming(
    query: mx.array,     # [B, H, Dk]
    key: mx.array,       # [B, H, Dk]
    value: mx.array,     # [B, H, Dv]
    decay: mx.array,     # [B, H, Dk]
    erase: mx.array,     # [B, H, Dk]
    write: mx.array,     # [B, H, Dv]
    state: mx.array,     # [B, H, Dv, Dk]
) -> Tuple[mx.array, mx.array]:
    """Single-token Metal forward."""
    # Decay
    state = state * decay[:, :, None, :]

    # Erase
    erase_value = mx.sum(state * erase[:, :, None, :] * key[:, :, None, :], axis=-1)

    # Write
    state = state - erase_value[:, :, :, None] * key[:, :, None, :]
    state = state + (write * value)[:, :, :, None] * key[:, :, None, :]

    # Query
    output = mx.sum(state * query[:, :, None, :], axis=-1)

    return output, state


class GDN2MetalKernel:
    """
    Wrapper for Metal-optimized GDN-2.

    Currently uses MLX ops. Production version:
    - mx.fast.metal_kernel for fused operations
    - Register state residency
    - SIMD cooperative reductions
    """

    @staticmethod
    def forward(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        decays: mx.array,
        erases: mx.array,
        writes: mx.array,
        initial_state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass via Metal."""
        return gdn2_metal_forward(
            queries, keys, values, decays, erases, writes, initial_state
        )

    @staticmethod
    def streaming(
        query: mx.array,
        key: mx.array,
        value: mx.array,
        decay: mx.array,
        erase: mx.array,
        write: mx.array,
        state: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Single-token forward via Metal."""
        return gdn2_metal_streaming(query, key, value, decay, erase, write, state)


# Metal shader pseudocode (for reference, not executable here)
"""
kernel gdn2_forward_kernel(
    uint3 grid_id [[threadgroup_position_in_grid]],
    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 grid_size [[threads_per_threadgroup]]
) {
    // Grid: [B*H, Dv, 1]
    // Each lane processes one (batch, head, value_channel) → cooperative SIMD over Dk

    uint bh_idx = grid_id.x;
    uint v_idx = grid_id.y;
    uint batch = bh_idx / H;
    uint head = bh_idx % H;

    uint simd_lane = local_id.x;  // 0..31 (32-wide SIMD)
    uint steps_per_lane = (Dk + 31) / 32;

    // Load state[batch, head, v_idx, :] into registers
    float state[steps_per_lane];
    for (uint k = 0; k < steps_per_lane; k++) {
        uint k_idx = simd_lane + k * 32;
        if (k_idx < Dk) {
            state[k] = state_global[batch, head, v_idx, k_idx];
        }
    }

    for (uint t = 0; t < T; t++) {
        // Decay
        for (uint k = 0; k < steps_per_lane; k++) {
            uint k_idx = simd_lane + k * 32;
            if (k_idx < Dk) {
                state[k] *= decay[batch, t, head, k_idx];
            }
        }

        // Erase: SIMD reduction
        float erase_partial = 0.0;
        for (uint k = 0; k < steps_per_lane; k++) {
            uint k_idx = simd_lane + k * 32;
            if (k_idx < Dk) {
                erase_partial += state[k] *
                                 erase[batch, t, head, k_idx] *
                                 key[batch, t, head, k_idx];
            }
        }
        float erase_value = simd_sum(erase_partial);

        // Write
        for (uint k = 0; k < steps_per_lane; k++) {
            uint k_idx = simd_lane + k * 32;
            if (k_idx < Dk) {
                state[k] -= erase_value * key[batch, t, head, k_idx];
                state[k] += write[batch, t, head, v_idx] *
                           value[batch, t, head, v_idx] *
                           key[batch, t, head, k_idx];
            }
        }

        // Query: SIMD reduction
        float output_partial = 0.0;
        for (uint k = 0; k < steps_per_lane; k++) {
            uint k_idx = simd_lane + k * 32;
            if (k_idx < Dk) {
                output_partial += state[k] * query[batch, t, head, k_idx];
            }
        }
        float output = simd_sum(output_partial);
        output_global[batch, t, head, v_idx] = output;
    }

    // Write back final state
    for (uint k = 0; k < steps_per_lane; k++) {
        uint k_idx = simd_lane + k * 32;
        if (k_idx < Dk) {
            state_global[batch, head, v_idx, k_idx] = state[k];
        }
    }
}
"""
