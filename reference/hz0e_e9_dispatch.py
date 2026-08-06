"""Deterministic grouped-token dispatch for the HZ-0E MoE contract.

This is the backend-neutral planning layer. It deliberately does not execute
expert weights; Metal and MLX implementations can consume the same plan.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DispatchPlan:
    capacity: int
    expert_index: np.ndarray
    rank: np.ndarray
    accepted: np.ndarray
    overflow: np.ndarray
    dispatch_slot: np.ndarray
    grouped_tokens: np.ndarray
    grouped_ranks: np.ndarray

    @property
    def token_count(self) -> int:
        return int(self.expert_index.size)


def build_dispatch_plan(expert_index: np.ndarray, num_experts: int,
                        capacity_factor: float) -> DispatchPlan:
    """Build a stable, bounded top-1 dispatch plan.

    ``grouped_tokens[e, r]`` is the original token index for expert ``e`` and
    queue position ``r``. ``-1`` is padding, so the plan is directly suitable
    for fixed-shape device buffers. Overflow tokens are never placed in an
    expert queue and must use the shared fallback path.
    """
    indices = np.asarray(expert_index, dtype=np.int64).reshape(-1)
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if capacity_factor <= 0 or not np.isfinite(capacity_factor):
        raise ValueError("capacity_factor must be finite and positive")
    if np.any(indices < 0) or np.any(indices >= num_experts):
        raise ValueError("expert_index contains an out-of-range expert")

    n = indices.size
    capacity = max(1, int(np.ceil(capacity_factor * n / num_experts)))
    rank = np.empty(n, dtype=np.int64)
    seen = np.zeros(num_experts, dtype=np.int64)
    grouped = np.full((num_experts, capacity), -1, dtype=np.int64)
    grouped_ranks = np.full((num_experts, capacity), -1, dtype=np.int64)
    for token, expert in enumerate(indices):
        position = seen[expert]
        rank[token] = position
        seen[expert] += 1
        if position < capacity:
            grouped[expert, position] = token
            grouped_ranks[expert, position] = position

    accepted = rank < capacity
    dispatch_slot = np.where(accepted, indices * capacity + rank, -1).astype(np.int64)
    return DispatchPlan(
        capacity=capacity,
        expert_index=indices.copy(),
        rank=rank,
        accepted=accepted,
        overflow=~accepted,
        dispatch_slot=dispatch_slot,
        grouped_tokens=grouped,
        grouped_ranks=grouped_ranks,
    )


def scatter_expert_outputs(plan: DispatchPlan, expert_outputs: np.ndarray,
                           fallback_outputs: np.ndarray) -> np.ndarray:
    """Scatter grouped expert outputs back to token order, with fallback."""
    expert_outputs = np.asarray(expert_outputs)
    fallback_outputs = np.asarray(fallback_outputs)
    if expert_outputs.shape[:2] != plan.grouped_tokens.shape:
        raise ValueError("expert_outputs must be [experts, capacity, ...]")
    if fallback_outputs.shape[0] != plan.token_count:
        raise ValueError("fallback_outputs must have one row per token")
    result = fallback_outputs.copy()
    for expert, queue in enumerate(plan.grouped_tokens):
        for slot, token in enumerate(queue):
            if token >= 0:
                result[token] = expert_outputs[expert, slot]
    return result
