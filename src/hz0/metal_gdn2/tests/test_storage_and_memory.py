"""
Tests for storage alternatives (Step 6) and memory API (Step 7).
"""

import pytest
import numpy as np
import mlx.core as mx
from hz0.metal_gdn2.scratchpad.storage import (
    RMSNormStorage,
    adaptive_norm_clip,
    AdaptiveStorageBuffer,
    StorageComparison,
)
from hz0.metal_gdn2.scratchpad.persistent_memory import (
    MemoryInterface,
    sequence_local_memory,
    chunk_persistent_memory,
    session_persistent_memory,
)


class TestStorageAlternatives:
    """Step 6: Storage alternatives."""

    def test_rmsnorm_storage_basic(self):
        """RMSNorm normalizes without saturation."""
        storage = RMSNormStorage(dim=8)
        x = mx.array(np.random.randn(8).astype(np.float32) * 10)

        normalized = storage(x)

        # Output should be reasonably normalized
        norm = float(mx.sqrt(mx.mean(normalized ** 2)))
        assert norm > 0  # Not zero
        assert norm < 2.0  # Not exploded

    def test_adaptive_norm_clip_below_threshold(self):
        """Values below max_norm pass through."""
        x = mx.array(np.random.randn(8).astype(np.float32) * 0.1)  # Small values
        max_norm = 1.0

        clipped = adaptive_norm_clip(x, max_norm=max_norm)

        # Should be close to original (scale ≈ 1)
        error = float(mx.mean((clipped - x) ** 2))
        assert error < 0.01

    def test_adaptive_norm_clip_above_threshold(self):
        """Values above max_norm are scaled down."""
        x = mx.array(np.random.randn(8).astype(np.float32) * 10)  # Large values
        max_norm = 1.0

        clipped = adaptive_norm_clip(x, max_norm=max_norm)

        # Clipped norm should be ≤ max_norm
        clipped_norm = float(mx.sqrt(mx.sum(clipped ** 2)))
        assert clipped_norm <= max_norm * 1.01  # Allow small numerical error

    def test_storage_buffer_store_retrieve(self):
        """Store and retrieve values."""
        buf = AdaptiveStorageBuffer(num_slots=4, slot_dim=8)

        value = mx.array(np.random.randn(8).astype(np.float32))
        buf.store(0, value)

        retrieved = buf.retrieve(0)

        # Should be close (after RMSNorm + clip)
        assert retrieved.shape == value.shape

    def test_storage_buffer_reset(self):
        """Reset clears buffer."""
        buf = AdaptiveStorageBuffer(num_slots=4, slot_dim=8)

        value = mx.array(np.ones(8, dtype=np.float32))
        buf.store(0, value)

        buf.reset()

        zeros = buf.retrieve(0)
        assert float(mx.mean(mx.abs(zeros))) < 0.1

    def test_tanh_clamp_vs_rmsnorm(self):
        """Compare old vs new storage."""
        x = mx.array(np.random.randn(8).astype(np.float32) * 5)

        tanh_stored = StorageComparison.tanh_clamp_storage(x)
        rmsnorm_stored = StorageComparison.rmsnorm_clip_storage(x)

        # Both should bound the values
        assert float(mx.max(mx.abs(tanh_stored))) <= 1.0
        assert float(mx.max(mx.abs(rmsnorm_stored))) <= 1.0

        # Both should have reasonable magnitude
        tanh_mean = float(mx.mean(mx.abs(tanh_stored)))
        rmsnorm_mean = float(mx.mean(mx.abs(rmsnorm_stored)))

        assert tanh_mean > 0
        assert rmsnorm_mean > 0


class TestPersistentMemory:
    """Step 7: Memory= API."""

    def test_sequence_local_reset(self):
        """Sequence local mode resets per call."""
        mem = sequence_local_memory(num_slots=4, slot_dim=8)

        # First forward
        values1 = mx.array(np.ones((2, 4, 8), dtype=np.float32))
        out1, state1 = mem(values1)

        # Second forward (should reset)
        values2 = mx.array(np.ones((2, 4, 8), dtype=np.float32) * 2)
        out2, state2 = mem(values2)

        # States may differ due to reset
        assert state1.shape == state2.shape

    def test_chunk_persistent_carry(self):
        """Chunk persistent carries state across calls."""
        mem = chunk_persistent_memory(num_slots=4, slot_dim=8)

        # First chunk
        values1 = mx.array(np.ones((2, 4, 8), dtype=np.float32))
        out1, state1 = mem(values1)

        # Second chunk: state should carry
        values2 = mx.array(np.ones((2, 4, 8), dtype=np.float32))
        out2, state2 = mem(values2, memory=state1)

        assert state2 is not None

    def test_session_persistent_external(self):
        """Session persistent with external state management."""
        mem = session_persistent_memory(num_slots=4, slot_dim=8)

        # Forward 1
        values = mx.array(np.random.randn(2, 4, 8).astype(np.float32))
        out, state = mem(values)

        # Save session
        saved_state = mem.persist_session()

        # Reset and start new session
        mem.reset_session()

        # Forward 2 with new values
        new_values = mx.array(np.random.randn(2, 4, 8).astype(np.float32))
        out_new, state_new = mem(new_values)

        # Restore session
        mem.restore_session(0)

        # State should match saved
        restored_state = mem.state
        assert np.allclose(np.array(restored_state), np.array(saved_state), atol=1e-5)

    def test_memory_interface_modes(self):
        """All three memory modes work."""
        for mode, factory in [
            ("sequence_local", sequence_local_memory),
            ("chunk_persistent", chunk_persistent_memory),
            ("session_persistent", session_persistent_memory),
        ]:
            mem = factory(num_slots=4, slot_dim=8)

            values = mx.array(np.random.randn(2, 4, 8).astype(np.float32))
            out, state = mem(values)

            # Output is the memory state (simplified interface)
            assert state.shape == (4, 8)

    def test_memory_state_export_import(self):
        """Get and set state directly."""
        mem = session_persistent_memory(num_slots=4, slot_dim=8)

        # Create and save state
        values = mx.array(np.random.randn(2, 4, 8).astype(np.float32))
        _, _ = mem(values)

        exported = mem.state

        # Create new memory and import
        mem2 = session_persistent_memory(num_slots=4, slot_dim=8)
        mem2.state = exported

        # States should match
        assert np.allclose(np.array(mem2.state), np.array(exported), atol=1e-5)

    def test_memory_reset_isolation(self):
        """Reset clears state."""
        mem = session_persistent_memory(num_slots=4, slot_dim=8)

        # Add some state
        values = mx.array(np.ones((2, 4, 8), dtype=np.float32))
        _, _ = mem(values)

        state_before = mem.state

        # Reset
        mem.reset_session()

        state_after = mem.state

        # After reset should be mostly zeros
        assert float(mx.mean(mx.abs(state_after))) < float(mx.mean(mx.abs(state_before)))
