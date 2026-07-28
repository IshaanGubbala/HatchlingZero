from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class GDN2ReferenceState:
    state: np.ndarray


def gdn2_numpy_step(
    state: np.ndarray,
    decay_logits: np.ndarray,
    erase_logits: np.ndarray,
    write_logits: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    """Small auditable reference recurrence with separated decay/erase/write gates.

    This is a Mac-friendly mathematical reference for HZ experimentation rather
    than a claim of kernel parity with the full upstream NVIDIA implementation.
    """
    decay = _sigmoid_np(decay_logits)
    erase = _sigmoid_np(erase_logits)
    write = _sigmoid_np(write_logits)
    retained = decay * (1.0 - erase) * state
    injected = write * np.tanh(candidate)
    return retained + injected


def gdn2_numpy_sequence(
    decay_logits: np.ndarray,
    erase_logits: np.ndarray,
    write_logits: np.ndarray,
    candidate: np.ndarray,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if decay_logits.shape != erase_logits.shape or decay_logits.shape != write_logits.shape:
        raise ValueError("Gate logits must have the same shape.")
    if decay_logits.shape != candidate.shape:
        raise ValueError("Candidate tensor must match gate tensor shape.")
    if decay_logits.ndim != 3:
        raise ValueError("Expected [batch, seq, dim] tensors.")

    batch, seq, dim = decay_logits.shape
    if initial_state is None:
        state = np.zeros((batch, dim), dtype=decay_logits.dtype)
    else:
        state = np.array(initial_state, copy=True)
    outputs = np.zeros_like(candidate)
    for t in range(seq):
        state = gdn2_numpy_step(
            state=state,
            decay_logits=decay_logits[:, t],
            erase_logits=erase_logits[:, t],
            write_logits=write_logits[:, t],
            candidate=candidate[:, t],
        )
        outputs[:, t] = state
    return outputs, state


def gdn2_numpy_stream(
    decay_logits: np.ndarray,
    erase_logits: np.ndarray,
    write_logits: np.ndarray,
    candidate: np.ndarray,
    chunk_size: int,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    outputs = []
    state = initial_state
    seq = decay_logits.shape[1]
    for start in range(0, seq, chunk_size):
        end = min(start + chunk_size, seq)
        chunk_out, state = gdn2_numpy_sequence(
            decay_logits[:, start:end],
            erase_logits[:, start:end],
            write_logits[:, start:end],
            candidate[:, start:end],
            initial_state=state,
        )
        outputs.append(chunk_out)
    return np.concatenate(outputs, axis=1), state


def gdn2_torch_reference(
    decay_logits: torch.Tensor,
    erase_logits: torch.Tensor,
    write_logits: torch.Tensor,
    candidate: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch wrapper around the NumPy reference for deterministic comparisons."""
    device = candidate.device
    dtype = candidate.dtype
    arrays = [
        decay_logits.detach().cpu().numpy(),
        erase_logits.detach().cpu().numpy(),
        write_logits.detach().cpu().numpy(),
        candidate.detach().cpu().numpy(),
    ]
    state_array = None if initial_state is None else initial_state.detach().cpu().numpy()
    outputs, final_state = gdn2_numpy_sequence(*arrays, initial_state=state_array)
    return (
        torch.from_numpy(outputs).to(device=device, dtype=dtype),
        torch.from_numpy(final_state).to(device=device, dtype=dtype),
    )
