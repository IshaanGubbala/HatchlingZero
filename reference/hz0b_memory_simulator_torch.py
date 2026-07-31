"""Torch port of `reference/hz0b_memory_simulator.py` (B2's pure memory
simulator), for B11 experiments that need to run on CUDA -- the real
frozen HZ-0A checkpoint is MLX/Metal-only and does not transfer to the
Windows/CUDA machine (`docs/rtx3060_windows_setup.md`), so any B11 work
dispatched there needs its own torch-native memory implementation, not a
stub.

Matches the MLX reference exactly, including the 2026-07-30 fixes
(`MATCH_THRESHOLD=0.999`, `confidence_weighted` default `True` on read) --
same constants, same equations, same shapes `[batch, num_slots, dim]`.
Not re-derived independently; ported term-for-term, the same discipline
`hz0b-pmetal-memory`'s Rust port used against this same Python reference.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

SOURCE_SUPERVISED = 0
SOURCE_LATENT = 1
PROTECTION_BLOCK_THRESHOLD = 0.5
MATCH_THRESHOLD = 0.999


@dataclass(frozen=True)
class MemoryState:
    keys: torch.Tensor
    values: torch.Tensor
    confidence: torch.Tensor
    age: torch.Tensor
    protection: torch.Tensor
    write_count: torch.Tensor
    last_write_step: torch.Tensor
    write_source: torch.Tensor


def reset(batch_size: int, num_slots: int, key_dim: int, value_dim: int, *, device=None) -> MemoryState:
    return MemoryState(
        keys=torch.zeros(batch_size, num_slots, key_dim, device=device),
        values=torch.zeros(batch_size, num_slots, value_dim, device=device),
        confidence=torch.zeros(batch_size, num_slots, device=device),
        age=torch.zeros(batch_size, num_slots, dtype=torch.int32, device=device),
        protection=torch.zeros(batch_size, num_slots, device=device),
        write_count=torch.zeros(batch_size, num_slots, dtype=torch.int32, device=device),
        last_write_step=torch.zeros(batch_size, num_slots, dtype=torch.int32, device=device),
        write_source=torch.zeros(batch_size, num_slots, dtype=torch.int32, device=device),
    )


def _cosine_similarity(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    query_norm = query / torch.sqrt((query * query).sum(dim=-1, keepdim=True) + 1e-8)
    keys_norm = keys / torch.sqrt((keys * keys).sum(dim=-1, keepdim=True) + 1e-8)
    return (keys_norm * query_norm.unsqueeze(1)).sum(dim=-1)


def read(state: MemoryState, query: torch.Tensor, *, slot_idx: torch.Tensor | None = None, hard: bool = False, confidence_weighted: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_slots, _ = state.keys.shape
    if slot_idx is not None:
        weights = torch.zeros(batch, num_slots, device=state.keys.device)
        weights.scatter_(1, slot_idx.long().unsqueeze(1), 1.0)
    else:
        scores = _cosine_similarity(query, state.keys)
        if confidence_weighted:
            scores = scores + torch.log(state.confidence + 1e-6)
        if hard:
            hard_idx = torch.argmax(scores, dim=-1)
            weights = torch.zeros(batch, num_slots, device=state.keys.device)
            weights.scatter_(1, hard_idx.unsqueeze(1), 1.0)
        else:
            weights = torch.softmax(scores, dim=-1)
    readout = (state.values * weights.unsqueeze(-1)).sum(dim=1)
    return readout, weights


def _choose_write_slot(state: MemoryState, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_slots, _ = state.keys.shape
    similarity = _cosine_similarity(key, state.keys)
    is_match = similarity > MATCH_THRESHOLD
    is_empty = state.confidence < 1e-6
    is_protected = state.protection >= PROTECTION_BLOCK_THRESHOLD

    has_match = is_match.any(dim=-1)
    masked_similarity = torch.where(is_match, similarity, torch.full_like(similarity, -2.0))
    matched_slot = torch.argmax(masked_similarity, dim=-1)
    match_is_protected = torch.gather(is_protected, 1, matched_slot.unsqueeze(1)).squeeze(1)

    eviction_score = state.confidence * (1.0 - state.protection) - state.age.float() * 1e-6
    eviction_score = torch.where(is_protected, torch.full_like(eviction_score, float("inf")), eviction_score)
    empty_score = torch.where(is_empty & ~is_protected, torch.full_like(eviction_score, -1.0), torch.full_like(eviction_score, float("inf")))
    eviction_slot = torch.argmin(torch.minimum(empty_score, eviction_score), dim=-1)
    all_protected = is_protected.all(dim=-1)

    slot_idx = torch.where(has_match, matched_slot, eviction_slot)
    forced_reject = torch.where(has_match, match_is_protected, all_protected)
    return slot_idx, forced_reject


def write(state: MemoryState, key: torch.Tensor, value: torch.Tensor, strength: torch.Tensor, *, step: int, source: int = SOURCE_LATENT, slot_idx: torch.Tensor | None = None) -> tuple[MemoryState, torch.Tensor, torch.Tensor]:
    batch, num_slots, _ = state.keys.shape
    if slot_idx is None:
        slot_idx, rejected = _choose_write_slot(state, key)
    else:
        target_protection = torch.gather(state.protection, 1, slot_idx.long().unsqueeze(1)).squeeze(1)
        rejected = target_protection >= PROTECTION_BLOCK_THRESHOLD

    one_hot = torch.zeros(batch, num_slots, device=state.keys.device)
    one_hot.scatter_(1, slot_idx.long().unsqueeze(1), 1.0)
    write_mask = one_hot * (1.0 - rejected.float()).unsqueeze(1)

    new_keys = state.keys * (1 - write_mask.unsqueeze(-1)) + write_mask.unsqueeze(-1) * key.unsqueeze(1)
    new_values = state.values * (1 - write_mask.unsqueeze(-1)) + write_mask.unsqueeze(-1) * value.unsqueeze(1)
    new_confidence = state.confidence * (1 - write_mask) + write_mask * strength.unsqueeze(1)
    new_age = torch.where(write_mask > 0, torch.zeros_like(state.age), state.age)
    new_write_count = state.write_count + write_mask.int()
    new_last_write_step = torch.where(write_mask > 0, torch.full_like(state.last_write_step, step), state.last_write_step)
    new_write_source = torch.where(write_mask > 0, torch.full_like(state.write_source, source), state.write_source)

    new_state = replace(state, keys=new_keys, values=new_values, confidence=new_confidence, age=new_age, write_count=new_write_count, last_write_step=new_last_write_step, write_source=new_write_source)
    return new_state, slot_idx, rejected


def forget_or_decay(state: MemoryState, *, decay_rate: float = 0.9) -> MemoryState:
    effective_decay = decay_rate * (1.0 - state.protection) + state.protection
    new_confidence = state.confidence * effective_decay
    new_age = state.age + 1
    return replace(state, confidence=new_confidence, age=new_age)


def protect(state: MemoryState, slot_idx: torch.Tensor, strength: torch.Tensor) -> MemoryState:
    batch, num_slots, _ = state.keys.shape
    mask = torch.zeros(batch, num_slots, device=state.keys.device)
    mask.scatter_(1, slot_idx.long().unsqueeze(1), 1.0)
    new_protection = state.protection * (1 - mask) + mask * strength.unsqueeze(1)
    return replace(state, protection=new_protection)
