"""
GDN-2 NumPy reference implementation (FP64).

Pure NumPy for numerical validation and gradient checking.
State layout: [B, H, Dv, Dk] where rows=value channels, columns=key dimension.

Reference for:
- Forward equivalence tests
- Gradient checking (finite differences)
- Special-case recovery (KDA, original GDN)
"""

import numpy as np
from typing import Tuple, Optional


def gdn2_step_decay(
    state: np.ndarray,  # [B, H, Dv, Dk]
    decay: np.ndarray,  # [B, H, Dk]
) -> np.ndarray:
    """Apply channel-wise decay: state *= decay."""
    return state * decay[:, :, None, :]


def gdn2_step_erase(
    state: np.ndarray,      # [B, H, Dv, Dk]
    erase: np.ndarray,      # [B, H, Dk]
    key: np.ndarray,        # [B, H, Dk]
) -> np.ndarray:
    """Selective erase: erase_read = sum_k(state * erase * key)."""
    erase_key = erase * key  # [B, H, Dk]
    erase_partial = state * erase_key[:, :, None, :]  # [B, H, Dv, Dk]
    return np.sum(erase_partial, axis=-1)  # [B, H, Dv]


def gdn2_step_update(
    state: np.ndarray,          # [B, H, Dv, Dk]
    erase_value: np.ndarray,    # [B, H, Dv]
    key: np.ndarray,            # [B, H, Dk]
    write: np.ndarray,          # [B, H, Dv]
    value: np.ndarray,          # [B, H, Dv]
) -> np.ndarray:
    """Write update: state -= erase_value*key, state += write*value*key."""
    erase_update = erase_value[:, :, :, None] * key[:, :, None, :]  # [B, H, Dv, Dk]
    write_product = write * value  # [B, H, Dv]
    write_update = write_product[:, :, :, None] * key[:, :, None, :]  # [B, H, Dv, Dk]
    return state - erase_update + write_update


def gdn2_step(
    state: np.ndarray,   # [B, H, Dv, Dk]
    query: np.ndarray,   # [B, H, Dk]
    key: np.ndarray,     # [B, H, Dk]
    value: np.ndarray,   # [B, H, Dv]
    decay: np.ndarray,   # [B, H, Dk]
    erase: np.ndarray,   # [B, H, Dk]
    write: np.ndarray,   # [B, H, Dv]
) -> Tuple[np.ndarray, np.ndarray]:
    """Single GDN-2 step: decay → erase → write → query."""
    state = gdn2_step_decay(state, decay)
    erase_value = gdn2_step_erase(state, erase, key)
    state = gdn2_step_update(state, erase_value, key, write, value)

    # Stability: clip state (same as MLX version)
    state = np.clip(state, -100.0, 100.0)

    # Query: output = sum_k(state * query)
    output_partial = state * query[:, :, None, :]  # [B, H, Dv, Dk]
    output = np.sum(output_partial, axis=-1)  # [B, H, Dv]

    return state, output


def gdn2_sequence(
    queries: np.ndarray,     # [B, T, H, Dk]
    keys: np.ndarray,        # [B, T, H, Dk]
    values: np.ndarray,      # [B, T, H, Dv]
    decays: np.ndarray,      # [B, T, H, Dk]
    erases: np.ndarray,      # [B, T, H, Dk]
    writes: np.ndarray,      # [B, T, H, Dv]
    initial_state: Optional[np.ndarray] = None,  # [B, H, Dv, Dk]
) -> Tuple[np.ndarray, np.ndarray]:
    """Process full sequence (NumPy reference, FP64)."""
    B, T, H, Dk = queries.shape
    _, _, Dv = values.shape[-3:]

    if initial_state is None:
        initial_state = np.zeros((B, H, Dv, Dk), dtype=np.float64)

    state = initial_state.copy()
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

    outputs = np.stack(outputs, axis=1)  # [B, T, H, Dv]
    return outputs, state


def gdn2_streaming(
    query: np.ndarray,   # [B, H, Dk]
    key: np.ndarray,     # [B, H, Dk]
    value: np.ndarray,   # [B, H, Dv]
    decay: np.ndarray,   # [B, H, Dk]
    erase: np.ndarray,   # [B, H, Dk]
    write: np.ndarray,   # [B, H, Dv]
    state: np.ndarray,   # [B, H, Dv, Dk]
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-token streaming step."""
    return gdn2_step(state, query, key, value, decay, erase, write)


if __name__ == "__main__":
    print("GDN-2 NumPy reference. Use for gradient checking and validation.")
