"""Exact vector-gated GDN-2 reference in HZ-0A's [Dv, Dk] state layout."""

from __future__ import annotations

import numpy as np


def normalize_keys(keys: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norms = np.sqrt(np.sum(np.square(keys), axis=-1, keepdims=True))
    return keys / np.maximum(norms, eps)


def gdn2_fix_step(state, query, key, value, alpha, erase, write, *, normalize_key: bool = True):
    """Run one step with state [B,H,Dv,Dk] and channels split by role."""
    if normalize_key:
        key = normalize_keys(key)
    decayed = state * alpha[:, :, None, :]
    erase_key = erase * key
    old_value = np.sum(decayed * erase_key[:, :, None, :], axis=-1)
    residual_value = write * value - old_value
    next_state = decayed + residual_value[:, :, :, None] * key[:, :, None, :]
    output = np.sum(next_state * query[:, :, None, :], axis=-1)
    return output, next_state


def gdn2_fix_scan(query, key, value, alpha, erase, write, initial_state=None, *, normalize_key: bool = True):
    batch, steps, heads, d_k = query.shape
    d_v = value.shape[-1]
    state = (np.zeros((batch, heads, d_v, d_k), dtype=np.result_type(value, np.float32))
             if initial_state is None else np.array(initial_state, copy=True))
    outputs = []
    for t in range(steps):
        output, state = gdn2_fix_step(
            state, query[:, t], key[:, t], value[:, t], alpha[:, t], erase[:, t], write[:, t],
            normalize_key=normalize_key,
        )
        outputs.append(output)
    return np.stack(outputs, axis=1), state


def gdn2_fix_chunk_scan(query, key, value, alpha, erase, write, chunk_size, initial_state=None, *, normalize_key: bool = True):
    state = initial_state
    outputs = []
    for start in range(0, query.shape[1], chunk_size):
        end = min(start + chunk_size, query.shape[1])
        output, state = gdn2_fix_scan(
            query[:, start:end], key[:, start:end], value[:, start:end],
            alpha[:, start:end], erase[:, start:end], write[:, start:end], state,
            normalize_key=normalize_key,
        )
        outputs.append(output)
    return np.concatenate(outputs, axis=1), state
