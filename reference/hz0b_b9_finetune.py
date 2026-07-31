"""HZ-0B Phase B9 ("partial and full fine-tuning"): utilities for
unfreezing specific pieces of the real, frozen HZ-0A checkpoint and
fine-tuning them jointly with the B7 write controller.

Per the plan's own staging: "1. unfreeze only memory-adjacent projections
2. unfreeze selected upper HZ-0A layers 3. consider full fine-tuning only
if necessary." Stage 1 here treats "memory-adjacent" as the LAST HZ-0A
block -- the one immediately upstream of where B6/B7's memory injection
point lives (`reference/hz0b_b6_hz0a_integration.py`'s `frozen_hidden_states`
stops right before `final_norm`, i.e. right after the last block).

MLX has no torch-style `requires_grad` flag -- "frozen" vs "trainable" is
just "not included" vs "included" in whatever pytree `mx.value_and_grad`
differentiates. These helpers extract a block's own parameters into a
flat dict (so it can be combined with the B7 controller's own dict and
trained jointly by the same `mx.value_and_grad` call), and write updated
values back into the live model object before each forward pass -- the
same `.update()` pattern used throughout this project's checkpoint
loading, not a new mechanism.
"""
from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def block_params_dict(model, block_index: int, *, prefix: str = "block") -> dict:
    """Flattens `model.blocks[block_index]`'s own parameters into a flat
    dict keyed `"{prefix}.{dotted.param.name}"`, disjoint from the B7
    controller's own dict keys (which never start with `prefix`) so the
    two can be merged into one combined trainable-params dict safely."""
    flat = tree_flatten(model.blocks[block_index].parameters())
    return {f"{prefix}.{key}": value for key, value in flat}


def apply_block_params(model, block_index: int, combined_dict: dict, *, prefix: str = "block") -> None:
    """Writes the `prefix.*`-keyed entries of `combined_dict` back into
    `model.blocks[block_index]` in place, mutating the live model object
    -- so a subsequent `model(...)` call inside the same loss function
    sees the updated block. Does not touch the saved checkpoint file on
    disk; only this in-memory model instance."""
    block_items = [(key[len(prefix) + 1:], value) for key, value in combined_dict.items() if key.startswith(f"{prefix}.")]
    model.blocks[block_index].update(tree_unflatten(block_items))


def block_param_count(model, block_index: int) -> int:
    return sum(value.size for _, value in tree_flatten(model.blocks[block_index].parameters()))


def multi_block_params_dict(model, block_indices: list[int]) -> dict:
    """B9 Stage 2 ("unfreeze selected upper HZ-0A layers", plural):
    same idea as `block_params_dict`, generalized to several blocks at
    once, each under its own `block{index}.` prefix so they stay
    disjoint from each other and from the controller's own keys."""
    combined = {}
    for index in block_indices:
        combined.update(block_params_dict(model, index, prefix=f"block{index}"))
    return combined


def apply_multi_block_params(model, block_indices: list[int], combined_dict: dict) -> None:
    for index in block_indices:
        apply_block_params(model, index, combined_dict, prefix=f"block{index}")


def multi_block_param_count(model, block_indices: list[int]) -> int:
    return sum(block_param_count(model, index) for index in block_indices)
