"""
NumPy FP64 reference implementation for GDN-2.
Used for finite-difference gradient validation.
"""

import numpy as np
from typing import Tuple


def gdn2_step_numpy(
    state: np.ndarray,   # [B, H, Dv, Dk]
    query: np.ndarray,   # [B, H, Dk]
    key: np.ndarray,     # [B, H, Dk]
    value: np.ndarray,   # [B, H, Dv]
    decay: np.ndarray,   # [B, H, Dk]
    erase: np.ndarray,   # [B, H, Dk]
    write: np.ndarray,   # [B, H, Dv]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single GDN-2 step in NumPy FP64.
    Returns new state and output.
    """
    # Decay
    state = state * decay[:, :, None, :]

    # Erase: sum_k (state * erase * key)
    erase_value = np.sum(state * erase[:, :, None, :] * key[:, :, None, :], axis=-1)

    # Write: state -= erase_value * key; state += write * value * key
    state = state - erase_value[:, :, :, None] * key[:, :, None, :]
    state = state + (write * value)[:, :, :, None] * key[:, :, None, :]

    # Query: sum_k (state * query)
    output = np.sum(state * query[:, :, None, :], axis=-1)

    return state, output


def gdn2_sequence_numpy(
    queries: np.ndarray,  # [B, T, H, Dk]
    keys: np.ndarray,     # [B, T, H, Dk]
    values: np.ndarray,   # [B, T, H, Dv]
    decays: np.ndarray,   # [B, T, H, Dk]
    erases: np.ndarray,   # [B, T, H, Dk]
    writes: np.ndarray,   # [B, T, H, Dv]
    initial_state: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full sequence in NumPy FP64."""
    B, T, H, Dk = queries.shape
    _, _, Dv = values.shape[-3:]

    if initial_state is None:
        initial_state = np.zeros((B, H, Dv, Dk), dtype=np.float64)

    state = initial_state.astype(np.float64)
    outputs = []

    for t in range(T):
        q_t = queries[:, t].astype(np.float64)
        k_t = keys[:, t].astype(np.float64)
        v_t = values[:, t].astype(np.float64)
        d_t = decays[:, t].astype(np.float64)
        e_t = erases[:, t].astype(np.float64)
        w_t = writes[:, t].astype(np.float64)

        state, output = gdn2_step_numpy(state, q_t, k_t, v_t, d_t, e_t, w_t)
        outputs.append(output)

    outputs = np.stack(outputs, axis=1)  # [B, T, H, Dv]
    return outputs, state
