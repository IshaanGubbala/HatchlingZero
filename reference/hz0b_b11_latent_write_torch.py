"""Torch port of `reference/hz0b_b8_latent_write.py` (the real,
unsupervised write+read memory controller), for the synthetic-backbone
B11 comparison run on CUDA. `write_gate`, `key`, `value` are all learned
functions of hidden state, no `should_write` label anywhere -- same
mechanism as the MLX version, ported term-for-term against
`reference/hz0b_memory_simulator_torch.py` (itself parity-tested against
the MLX memory reference, see
`tests/reference/test_hz0b_memory_simulator_torch_parity.py`)."""
from __future__ import annotations

import torch
import torch.nn as nn

from reference.hz0b_memory_simulator_torch import MemoryState, read as memory_read, reset, write as memory_write


class LatentWriteController(nn.Module):
    def __init__(self, d_model: int, key_dim: int, value_dim: int, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        scale = (2.0 / d_model) ** 0.5

        def linear(in_dim, out_dim):
            layer = nn.Linear(in_dim, out_dim)
            with torch.no_grad():
                layer.weight.normal_(std=scale, generator=generator)
                layer.bias.zero_()
            return layer

        self.query_proj = linear(d_model, key_dim)
        self.gate_proj = linear(d_model, d_model)
        self.value_to_hidden = linear(value_dim, d_model)
        self.write_gate_proj = linear(d_model, 1)
        self.key_proj = linear(d_model, key_dim)
        self.value_proj = linear(d_model, value_dim)
        self.occupancy_gate_w = nn.Parameter(torch.zeros(1))
        self.key_dim = key_dim
        self.value_dim = value_dim

    def gated_read(self, hidden_state: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        query = self.query_proj(hidden_state)
        readout, _ = memory_read(memory_state, query, hard=False, confidence_weighted=True)
        gate = torch.sigmoid(self.gate_proj(hidden_state))
        return hidden_state + gate * self.value_to_hidden(readout)

    def forward(self, hidden: torch.Tensor, *, num_slots: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden: [batch, seq, d_model] -> (output [batch, seq, d_model], write_gates [batch, seq])."""
        batch, seq, d_model = hidden.shape
        memory_state = reset(batch, num_slots, self.key_dim, self.value_dim, device=hidden.device)
        outputs, gates = [], []
        for t in range(seq):
            h_t = hidden[:, t, :]
            output = self.gated_read(h_t, memory_state)
            max_confidence = memory_state.confidence.max(dim=-1).values
            write_logit = self.write_gate_proj(h_t).squeeze(-1) + max_confidence * self.occupancy_gate_w[0]
            write_gate = torch.sigmoid(write_logit)
            key = self.key_proj(h_t)
            value = self.value_proj(h_t)
            candidate_state, _, _ = memory_write(memory_state, key, value, write_gate, step=t)
            memory_state = _blend_state_by_row(memory_state, candidate_state, write_gate)
            outputs.append(output)
            gates.append(write_gate)
        return torch.stack(outputs, dim=1), torch.stack(gates, dim=1)


def _blend_state_by_row(old: MemoryState, new: MemoryState, mask_1d: torch.Tensor) -> MemoryState:
    """mask_1d: [batch] in (0,1) -- continuous blend between old and new
    state per batch row, matching `hz0b_write_integration._blend_state_by_row`
    (MLX) exactly, just per-tensor here instead of via `dataclasses.replace`
    over an arbitrary field list."""
    from dataclasses import replace

    m = mask_1d.view(-1, 1, 1)
    m2 = mask_1d.view(-1, 1)
    return replace(
        old,
        keys=old.keys * (1 - m) + new.keys * m,
        values=old.values * (1 - m) + new.values * m,
        confidence=old.confidence * (1 - m2) + new.confidence * m2,
        age=torch.where(mask_1d.view(-1, 1) > 0.5, new.age, old.age),
        protection=old.protection * (1 - m2) + new.protection * m2,
        write_count=torch.where(mask_1d.view(-1, 1) > 0.5, new.write_count, old.write_count),
        last_write_step=torch.where(mask_1d.view(-1, 1) > 0.5, new.last_write_step, old.last_write_step),
        write_source=torch.where(mask_1d.view(-1, 1) > 0.5, new.write_source, old.write_source),
    )


def param_count(d_model: int, key_dim: int, value_dim: int) -> int:
    return (
        d_model * key_dim + key_dim  # query_proj
        + d_model * d_model + d_model  # gate_proj
        + value_dim * d_model + d_model  # value_to_hidden
        + d_model * 1 + 1  # write_gate_proj
        + d_model * key_dim + key_dim  # key_proj
        + d_model * value_dim + value_dim  # value_proj
        + 1  # occupancy_gate_w
    )
