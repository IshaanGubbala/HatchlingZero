"""
Pure MLX reference implementation of GDN-2 recurrent operations.
State layout: [B, H, Dv, Dk] where rows are value channels, columns are key dimension.
Each output value can be assigned to a Metal grid row with SIMD reduction over Dk.
"""

import mlx.core as mx
from typing import Tuple


def gdn2_step_decay(
    state: mx.array,  # [B, H, Dv, Dk]
    decay: mx.array,  # [B, H, Dk]
) -> mx.array:
    """Apply channel-wise decay to state. State shape unchanged."""
    # Broadcast decay [B, H, Dk] -> [B, H, 1, Dk] then multiply
    return state * decay[:, :, None, :]


def gdn2_step_erase(
    state: mx.array,  # [B, H, Dv, Dk]
    erase: mx.array,  # [B, H, Dk]
    key: mx.array,   # [B, H, Dk]
) -> Tuple[mx.array, mx.array]:
    """
    Selective erase reduction. Returns updated state and erase value.
    erase_read = sum over Dk of: state * (erase * key)
    """
    # Broadcast: erase [B, H, Dk] and key [B, H, Dk]
    # erase_read per value channel = sum_k (state[i, k] * erase[k] * key[k])
    erase_key_product = erase * key  # [B, H, Dk]

    # state [B, H, Dv, Dk] * erase_key_product [B, H, 1, Dk] -> [B, H, Dv, Dk]
    erase_partial = state * erase_key_product[:, :, None, :]

    # Sum over key dimension -> [B, H, Dv]
    erase_value = mx.sum(erase_partial, axis=-1)

    return erase_value


def gdn2_step_update(
    state: mx.array,     # [B, H, Dv, Dk]
    erase_value: mx.array,  # [B, H, Dv]
    key: mx.array,       # [B, H, Dk]
    write: mx.array,     # [B, H, Dv]
    value: mx.array,     # [B, H, Dv]
) -> mx.array:
    """
    Erase and write update to state.
    state -= erase_value * key
    state += write * value * key
    """
    # erase_value [B, H, Dv] -> [B, H, Dv, 1] for broadcasting
    erase_update = erase_value[:, :, :, None] * key[:, :, None, :]  # [B, H, Dv, Dk]

    # write [B, H, Dv], value [B, H, Dv], key [B, H, Dk]
    # write * value -> [B, H, Dv]
    # (write * value) [B, H, Dv, 1] * key [B, H, 1, Dk] -> [B, H, Dv, Dk]
    write_value_product = write * value  # [B, H, Dv]
    write_update = write_value_product[:, :, :, None] * key[:, :, None, :]  # [B, H, Dv, Dk]

    return state - erase_update + write_update


def gdn2_step_query(
    state: mx.array,  # [B, H, Dv, Dk]
    query: mx.array,  # [B, H, Dk]
) -> mx.array:
    """
    Query output. Sum over key dimension.
    output = sum_k (state * query)
    """
    # state [B, H, Dv, Dk] * query [B, H, 1, Dk] -> [B, H, Dv, Dk]
    output_partial = state * query[:, :, None, :]

    # Sum over key dimension -> [B, H, Dv]
    return mx.sum(output_partial, axis=-1)


def gdn2_step(
    state: mx.array,   # [B, H, Dv, Dk]
    query: mx.array,   # [B, H, Dk]
    key: mx.array,     # [B, H, Dk]
    value: mx.array,   # [B, H, Dv]
    decay: mx.array,   # [B, H, Dk]
    erase: mx.array,   # [B, H, Dk]
    write: mx.array,   # [B, H, Dv]
) -> Tuple[mx.array, mx.array]:
    """
    Single GDN-2 recurrent step. Returns new state and output.

    Includes stabilization: clip state after each update to prevent overflow.
    """
    # Decay
    state = gdn2_step_decay(state, decay)

    # Erase
    erase_value = gdn2_step_erase(state, erase, key)

    # Write
    state = gdn2_step_update(state, erase_value, key, write, value)

    # Stabilize: Clip state values to prevent unbounded growth
    # This is safe because GDN-2 state represents content addressable memory
    # Clipping preserves the relative signal structure
    state = mx.clip(state, -100.0, 100.0)

    # Query
    output = gdn2_step_query(state, query)

    return state, output


def gdn2_sequence_ops(
    queries: mx.array,  # [B, T, H, Dk]
    keys: mx.array,     # [B, T, H, Dk]
    values: mx.array,   # [B, T, H, Dv]
    decays: mx.array,   # [B, T, H, Dk]
    erases: mx.array,   # [B, T, H, Dk]
    writes: mx.array,   # [B, T, H, Dv]
    initial_state: mx.array = None,  # [B, H, Dv, Dk]
) -> Tuple[mx.array, mx.array]:
    """
    Full sequence processing. Returns outputs and final state.
    Pure loop—used only for correctness validation, not training.
    """
    B, T, H, Dk = queries.shape
    _, _, Dv = values.shape[-3:]

    if initial_state is None:
        initial_state = mx.zeros((B, H, Dv, Dk), dtype=queries.dtype)

    state = initial_state
    outputs = []

    for t in range(T):
        q_t = queries[:, t]      # [B, H, Dk]
        k_t = keys[:, t]         # [B, H, Dk]
        v_t = values[:, t]       # [B, H, Dv]
        d_t = decays[:, t]       # [B, H, Dk]
        e_t = erases[:, t]       # [B, H, Dk]
        w_t = writes[:, t]       # [B, H, Dv]

        state, output = gdn2_step(state, q_t, k_t, v_t, d_t, e_t, w_t)
        outputs.append(output)

    outputs = mx.stack(outputs, axis=1)  # [B, T, H, Dv]
    return outputs, state


def gdn2_streaming_ops(
    query: mx.array,   # [B, H, Dk]
    key: mx.array,     # [B, H, Dk]
    value: mx.array,   # [B, H, Dv]
    decay: mx.array,   # [B, H, Dk]
    erase: mx.array,   # [B, H, Dk]
    write: mx.array,   # [B, H, Dv]
    state: mx.array,   # [B, H, Dv, Dk]
) -> Tuple[mx.array, mx.array]:
    """
    Single-token streaming step. Stateful interface.
    Returns output and updated state.
    """
    new_state, output = gdn2_step(state, query, key, value, decay, erase, write)
    return output, new_state
