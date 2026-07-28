"""Bounded session-local memory for the HZ-0B track.

HZ-0B is described in the development plan as a "low-rank, bounded synaptic
memory with explicit reset and persistence rules". This module is the v1
implementation that satisfies those three contract points explicitly:

* **Bounded**: write values pass through ``tanh`` and the state is hard-clamped
  to ``[-1, 1]`` on every step. The state shape is ``[batch, num_slots, dim]``
  with ``num_slots`` small (the HZ-0B config uses 8).
* **Explicit reset rule**: ``reset(...)`` returns ``torch.zeros(...)`` and is
  called at the start of every forward pass by ``HybridLM._apply_scratchpad``.
  Cross-session state cannot leak.
* **Explicit persistence rule**: writes are routed through *learned slot
  addresses* (``self.slot_addresses``) using straight-through hard routing.
  Unselected slots pass through unchanged, so distractor (filler) tokens
  cannot disturb a binding whose slot they do not route to, even after tens
  of filler steps.

This iteration of the module adds **per-token hard-route diagnostics**
(see ``hz0b-mem-fix-plan-2026-07-26.md`` Phase 3): the straight-through hard
routing index (``argmax(slot_addresses @ signal)``) is now surfaced on every
``ScratchpadLogEntry`` as ``read_hard_idx`` and ``write_hard_idx`` (``int64``
shape ``[batch]``). Probe runs then aggregate ``route_match_rate``,
``slot_occupancy``, ``slot_collision_rate``, ``soft_routing_entropy_mean``,
and ``dead_slot_fraction`` directly from the logs. This separates three
previously-conflated failure modes: routing failed vs storage failed vs
readout injection failed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ScratchpadLogEntry:
    read_weights: torch.Tensor
    write_weights: torch.Tensor
    state_norm: torch.Tensor
    read_hard_idx: torch.Tensor
    write_hard_idx: torch.Tensor


class SessionScratchpad(nn.Module):
    """Slot-addressed fast-weight memory bank.

    The key ideas:

    * Each slot has a learned ``slot_addresses`` vector in the projected
      ``key``/``query`` space. ``nn.init.orthogonal_`` maximises the angular
      separation between slots at initialisation so hard routing gives
      diverse slot usage from day one (this is the cure for "dead-slot
      collapse" that otherwise shows up under straight-through argmax).
    * Hard routing (``argmax``) is used at both write and read time with the
      straight-through estimator
      ``one_hot(argmax) + softmax - softmax.detach()`` so the forward pass is
      a true one-hot dispatch while gradients still flow through the soft
      version. This lets the model learn slot addresses that align with its
      own projected key/query distributions.
    * Writes are slot-local: at the routed slot the new ``tanh(value)``
      replaces the old content; at every other slot the state passes through
      unchanged. Combined with orthogonal slot addresses, distractor tokens
      (the filler spans between a (key, value) binding and its query) almost
      never pick the binding's slot, so the binding survives the full
      filler span.
    * Per-token ``read_hard_idx`` and ``write_hard_idx`` (``int64`` shape
      ``[batch]``) are surfaced on every ``ScratchpadLogEntry`` so probe
      runs can compute ``route_match_rate`` and slot-occupancy diagnostics
      independent of LM-loss measurements.
    """

    def __init__(self, num_slots: int, dim: int, momentum: float = 0.0) -> None:
        super().__init__()
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1].")
        self.num_slots = num_slots
        self.dim = dim
        self.momentum = momentum
        # `momentum` is now an *intra-slot persistence* knob: when a write
        # addresses the same slot twice, the new value is blended with the
        # previous content by ``(1 - momentum)``. ``momentum == 0`` keeps the
        # original "replace" semantics at the routed slot; ``momentum == 1``
        # freezes the slot on the first write. The default is 0 (replace)
        # because that is the simplest fast-weight behaviour: each new
        # binding wins at its slot, older contents at other slots are
        # untouched.
        self.slot_addresses = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.orthogonal_(self.slot_addresses)

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_slots, self.dim, device=device, dtype=dtype)

    def _route(
        self,
        signal: torch.Tensor,
        *,
        oracle_slot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hard-route ``signal`` to the slot whose address matches best.

        When ``oracle_slot`` is provided (int64 ``[batch]``), bypass the
        learned ``slot_addresses`` projection entirely: the hard routing
        index is the oracle slot directly. This is the Phase-4 oracle
        diagnostic from ``docs/hz0b/mem-fix-plan-2026-07-26.md``: the
        probe CLI computes ``slot = token_id % num_slots`` from the raw
        prompt and threads it through every position so writes/reads
        land on the same slot deterministically, isolating routing from
        storage and readout.

        Returns:
            ste: ``[batch, num_slots]`` one-hot routing mask with STE for gradients.
            hard_idx: ``[batch]`` ``int64`` argmax index. Used for diagnostics
                (route_match_rate, slot_occupancy, dead_slot_fraction).
            soft: ``[batch, num_slots]`` the softmax distribution (for
                entropy diagnostics). Under oracle mode this is uniform
                ``1 / num_slots`` so the entropy diagnostic reflects the
                constant oracle signal rather than a learned distribution.
        """
        if signal.ndim != 2:
            raise ValueError("signal must be [batch, dim].")
        if oracle_slot is not None:
            hard_idx = oracle_slot.to(torch.long).contiguous()
            if hard_idx.shape != (signal.shape[0],):
                raise ValueError(
                    f"oracle_slot shape {hard_idx.shape} != batch size {signal.shape[0]}"
                )
            if int(hard_idx.min()) < 0 or int(hard_idx.max()) >= self.num_slots:
                raise ValueError(
                    f"oracle_slot values out of [0, {self.num_slots}) range: "
                    f"min={int(hard_idx.min())} max={int(hard_idx.max())}"
                )
            hard = F.one_hot(hard_idx, num_classes=self.num_slots).to(signal.dtype)
            uniform = torch.full(
                (signal.shape[0], self.num_slots),
                1.0 / max(self.num_slots, 1),
                dtype=signal.dtype,
                device=signal.device,
            )
            soft = uniform
            ste = hard + soft - soft.detach()
            return ste, hard_idx, soft
        scores = torch.matmul(
            self.slot_addresses.unsqueeze(0),
            signal.unsqueeze(-1),
        ).squeeze(-1) / max(self.dim, 1) ** 0.5
        hard_idx = scores.argmax(dim=-1)
        hard = F.one_hot(hard_idx, num_classes=self.num_slots).to(scores.dtype)
        soft = scores.softmax(dim=-1)
        ste = hard + soft - soft.detach()
        # Hard routing index (separate from the STE mask) so the probe
        # runner can aggregate slot identity across the sequence.
        return ste, hard_idx, soft

    def read(
        self,
        query: torch.Tensor,
        state: torch.Tensor,
        *,
        oracle_slot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.ndim != 2:
            raise ValueError("query must be [batch, dim].")
        ste, hard_idx, soft = self._route(query, oracle_slot=oracle_slot)
        readout = torch.sum(state * ste.unsqueeze(-1), dim=1)
        return readout, hard_idx, soft

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        state: torch.Tensor,
        *,
        oracle_slot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if key.ndim != 2 or value.ndim != 2:
            raise ValueError("key and value must be [batch, dim].")
        ste, hard_idx, soft = self._route(key, oracle_slot=oracle_slot)
        new_value = torch.tanh(value)
        # Selected slot: blend old and new by ``momentum``.
        #   momentum == 0.0 -> "next = new_value" (replace)
        #   momentum == 1.0 -> "next = state"     (freeze)
        selected_blend = (
            state * self.momentum + (1.0 - self.momentum) * new_value.unsqueeze(1)
        )
        # Unselected slots pass through unchanged: `state * (1 - ste) + selected_blend * ste`
        #                                   = `state + ste * (selected_blend - state)`
        merged = state * (1.0 - ste.unsqueeze(-1)) + ste.unsqueeze(-1) * selected_blend
        next_state = torch.clamp(merged, min=-1.0, max=1.0)
        return next_state, hard_idx, soft

    def step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        state: torch.Tensor,
        *,
        log: bool = False,
        oracle_slot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ScratchpadLogEntry | None]:
        readout, read_hard_idx, read_soft = self.read(query, state, oracle_slot=oracle_slot)
        next_state, write_hard_idx, write_soft = self.write(key, value, state, oracle_slot=oracle_slot)
        if not log:
            return readout, next_state, None
        entry = ScratchpadLogEntry(
            read_weights=read_soft,
            write_weights=write_soft,
            state_norm=next_state.norm(dim=-1).mean(dim=-1),
            read_hard_idx=read_hard_idx,
            write_hard_idx=write_hard_idx,
        )
        return readout, next_state, entry
