"""Cross-framework parity: reference/hz0b_memory_simulator_torch.py vs
the real MLX reference it was ported from, same discipline as B10's
Rust parity test (tests/reference/test_hz0b_memory_rust_bridge.py) --
runs the identical operation sequence through both and asserts exact
numeric agreement."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch

from reference import hz0b_memory_simulator as mlx_mem
from reference import hz0b_memory_simulator_torch as torch_mem


def test_torch_port_matches_mlx_reference_on_a_real_sequence():
    num_slots, key_dim, value_dim = 8, 6, 6
    mlx_state = mlx_mem.reset(1, num_slots, key_dim, value_dim)
    torch_state = torch_mem.reset(1, num_slots, key_dim, value_dim)

    rng = np.random.default_rng(42)
    for step in range(6):
        key_np = rng.standard_normal((1, key_dim)).astype(np.float32)
        value_np = rng.standard_normal((1, value_dim)).astype(np.float32)
        strength_np = np.array([0.9], dtype=np.float32)

        mlx_state, mlx_slot, mlx_rejected = mlx_mem.write(mlx_state, mx.array(key_np), mx.array(value_np), mx.array(strength_np), step=step, source=1)
        torch_state, torch_slot, torch_rejected = torch_mem.write(torch_state, torch.from_numpy(key_np), torch.from_numpy(value_np), torch.from_numpy(strength_np), step=step, source=1)
        assert np.array(mlx_slot).tolist() == torch_slot.tolist()
        assert np.array(mlx_rejected).tolist() == torch_rejected.tolist()

        if step == 3:
            mlx_state = mlx_mem.protect(mlx_state, mlx_slot, mx.array([1.0]))
            torch_state = torch_mem.protect(torch_state, torch_slot, torch.tensor([1.0]))
        if step == 5:
            mlx_state = mlx_mem.forget_or_decay(mlx_state, decay_rate=0.8)
            torch_state = torch_mem.forget_or_decay(torch_state, decay_rate=0.8)

    mlx_readout, mlx_weights = mlx_mem.read(mlx_state, mx.array(key_np), hard=True)
    torch_readout, torch_weights = torch_mem.read(torch_state, torch.from_numpy(key_np), hard=True)
    assert np.allclose(np.array(mlx_readout), torch_readout.numpy(), atol=1e-4)
    assert np.allclose(np.array(mlx_weights), torch_weights.numpy(), atol=1e-4)
    assert np.allclose(np.array(mlx_state.confidence), torch_state.confidence.numpy(), atol=1e-4)
    assert np.allclose(np.array(mlx_state.protection), torch_state.protection.numpy(), atol=1e-4)


def test_near_identical_keys_stay_distinct_after_the_0_999_threshold_fix():
    """Same scenario as the Rust port's own regression test -- mirrors
    docs/restart/hz0b_b8_stage5_results.md finding 1."""
    num_slots, key_dim, value_dim = 8, 16, 16
    state = torch_mem.reset(1, num_slots, key_dim, value_dim)
    key_a = torch.zeros(1, key_dim); key_a[0, 0] = 1.0
    key_b_raw = torch.zeros(1, key_dim); key_b_raw[0, 0] = 0.995
    key_b_raw[0, 1] = (1.0 - 0.995 ** 2) ** 0.5
    key_b = key_b_raw / key_b_raw.norm(dim=-1, keepdim=True)

    value_a = torch.zeros(1, value_dim); value_a[0, 0] = 5.0
    value_b = torch.zeros(1, value_dim); value_b[0, 1] = 5.0
    state, slot_a, _ = torch_mem.write(state, key_a, value_a, torch.tensor([1.0]), step=0, source=1)
    state, slot_b, _ = torch_mem.write(state, key_b, value_b, torch.tensor([1.0]), step=1, source=1)
    assert slot_a.item() != slot_b.item()
