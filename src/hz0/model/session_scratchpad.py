from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ScratchpadLogEntry:
    read_weights: torch.Tensor
    write_weights: torch.Tensor
    state_norm: torch.Tensor


class SessionScratchpad:
    """Bounded session-local memory for the HZ-0B track.

    The scratchpad is intentionally small and resettable. It is not wired into
    the language model forward path yet; this module provides the isolated
    read/write mechanics and logging surface needed before that integration.
    """

    def __init__(self, num_slots: int, dim: int, momentum: float = 0.9) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        self.num_slots = num_slots
        self.dim = dim
        self.momentum = momentum

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_slots, self.dim, device=device, dtype=dtype)

    def read(self, query: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 2:
            raise ValueError("query must be [batch, dim].")
        scores = torch.matmul(state, query.unsqueeze(-1)).squeeze(-1) / max(self.dim, 1) ** 0.5
        weights = torch.softmax(scores, dim=-1)
        readout = torch.sum(state * weights.unsqueeze(-1), dim=1)
        return readout, weights

    def write(self, key: torch.Tensor, value: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if key.ndim != 2 or value.ndim != 2:
            raise ValueError("key and value must be [batch, dim].")
        scores = torch.matmul(state, key.unsqueeze(-1)).squeeze(-1) / max(self.dim, 1) ** 0.5
        weights = torch.softmax(scores, dim=-1)
        update = weights.unsqueeze(-1) * torch.tanh(value).unsqueeze(1)
        next_state = self.momentum * state + (1.0 - self.momentum) * update
        next_state = torch.clamp(next_state, min=-1.0, max=1.0)
        return next_state, weights

    def step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        state: torch.Tensor,
        *,
        log: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, ScratchpadLogEntry | None]:
        readout, read_weights = self.read(query, state)
        next_state, write_weights = self.write(key, value, state)
        if not log:
            return readout, next_state, None
        entry = ScratchpadLogEntry(
            read_weights=read_weights,
            write_weights=write_weights,
            state_norm=next_state.norm(dim=-1).mean(dim=-1),
        )
        return readout, next_state, entry
