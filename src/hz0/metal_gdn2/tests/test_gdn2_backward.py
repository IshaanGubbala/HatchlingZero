"""
Tests for GDN-2 backward pass and custom VJP.
"""

import pytest
import numpy as np
import mlx.core as mx
from src.hz0.metal_gdn2.kernels.gdn2_backward import (
    gdn2_sequence_with_chunks,
    validate_gradient_structure,
)


class TestGDN2Chunked:
    """Chunked forward pass tests."""

    @pytest.fixture
    def synthetic_sequence(self):
        """Small synthetic sequence."""
        B, T, H, Dk, Dv = 2, 16, 2, 4, 3
        np.random.seed(42)

        queries = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        keys = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        values = mx.array(np.random.randn(B, T, H, Dv).astype(np.float32))
        decays = mx.array((0.95 + 0.05 * np.random.randn(B, T, H, Dk)).astype(np.float32))
        erases = mx.array(np.clip(np.random.randn(B, T, H, Dk), 0, 1).astype(np.float32))
        writes = mx.array(np.clip(np.random.randn(B, T, H, Dv), 0, 1).astype(np.float32))

        return queries, keys, values, decays, erases, writes

    def test_chunked_output_shape(self, synthetic_sequence):
        """Chunked forward produces correct output."""
        queries, keys, values, decays, erases, writes = synthetic_sequence
        B, T, H, Dk = queries.shape
        _, _, Dv = values.shape[-3:]

        outputs, state, chunk_states = gdn2_sequence_with_chunks(
            queries, keys, values, decays, erases, writes, chunk_size=4
        )

        assert outputs.shape == (B, T, H, Dv)
        assert state.shape == (B, H, Dv, Dk)
        assert len(chunk_states) >= 2  # Initial + at least one chunk boundary

    def test_chunk_sizes(self, synthetic_sequence):
        """Chunking works with various chunk sizes."""
        queries, keys, values, decays, erases, writes = synthetic_sequence

        for chunk_size in [1, 2, 4, 8, 16, 32]:
            outputs, state, chunk_states = gdn2_sequence_with_chunks(
                queries, keys, values, decays, erases, writes, chunk_size=chunk_size
            )

            assert outputs.shape == queries.shape[:2] + (queries.shape[2], values.shape[-1])
            assert state is not None
            assert len(chunk_states) > 0

    def test_chunk_states_progression(self, synthetic_sequence):
        """Chunk states increase in time."""
        queries, keys, values, decays, erases, writes = synthetic_sequence

        outputs, state, chunk_states = gdn2_sequence_with_chunks(
            queries, keys, values, decays, erases, writes, chunk_size=4
        )

        # All chunk states have correct shape
        for cs in chunk_states:
            assert cs.shape == state.shape

        # Final chunk state matches overall final state
        assert np.allclose(np.array(chunk_states[-1]), np.array(state), atol=1e-5)

    def test_no_nan_chunked_long_sequence(self):
        """Long sequence with chunking produces no NaNs."""
        B, T, H, Dk, Dv = 1, 256, 1, 2, 2

        np.random.seed(0)
        queries = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32) * 0.1)
        keys = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32) * 0.1)
        values = mx.array(np.random.randn(B, T, H, Dv).astype(np.float32) * 0.1)
        decays = mx.array(
            np.clip(0.95 + 0.02 * np.random.randn(B, T, H, Dk), 0.9, 1.0).astype(
                np.float32
            )
        )
        erases = mx.array(np.clip(np.random.randn(B, T, H, Dk), 0, 1).astype(np.float32))
        writes = mx.array(np.clip(np.random.randn(B, T, H, Dv), 0, 1).astype(np.float32))

        outputs, state, chunk_states = gdn2_sequence_with_chunks(
            queries, keys, values, decays, erases, writes, chunk_size=64
        )

        assert not np.any(np.isnan(np.array(outputs)))
        assert not np.any(np.isnan(np.array(state)))


class TestGDN2GradientFlow:
    """Gradient computation and validation."""

    def test_mlx_autodiff_gradients(self):
        """MLX autodiff produces valid gradients."""
        B, T, H, Dk, Dv = 1, 8, 1, 2, 2

        np.random.seed(0)
        queries = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        keys = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        values = mx.array(np.random.randn(B, T, H, Dv).astype(np.float32))
        decays = mx.array((0.95 + 0.05 * np.random.randn(B, T, H, Dk)).astype(np.float32))
        erases = mx.array(np.clip(np.random.randn(B, T, H, Dk), 0, 1).astype(np.float32))
        writes = mx.array(np.clip(np.random.randn(B, T, H, Dv), 0, 1).astype(np.float32))

        def loss_fn(q):
            outputs, _, _ = gdn2_sequence_with_chunks(
                q, keys, values, decays, erases, writes
            )
            return mx.mean(outputs)

        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(queries)

        # Gradients should be finite
        grad_arr = np.array(grads)
        assert not np.any(np.isnan(grad_arr))
        assert not np.any(np.isinf(grad_arr))
        # Gradients should not all be zero
        assert np.any(grad_arr != 0)

    def test_gradient_through_decay(self):
        """Gradients flow through decay parameters."""
        B, T, H, Dk, Dv = 1, 4, 1, 2, 2

        np.random.seed(0)
        queries = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        keys = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        values = mx.array(np.random.randn(B, T, H, Dv).astype(np.float32))
        decays = mx.array((0.95 + 0.05 * np.random.randn(B, T, H, Dk)).astype(np.float32))
        erases = mx.array(np.clip(np.random.randn(B, T, H, Dk), 0, 1).astype(np.float32))
        writes = mx.array(np.clip(np.random.randn(B, T, H, Dv), 0, 1).astype(np.float32))

        def loss_fn(d):
            outputs, _, _ = gdn2_sequence_with_chunks(
                queries, keys, values, d, erases, writes
            )
            return mx.mean(outputs ** 2)

        grad_fn = mx.grad(loss_fn)
        decay_grads = grad_fn(decays)

        decay_grad_arr = np.array(decay_grads)
        assert not np.any(np.isnan(decay_grad_arr))
        assert np.any(decay_grad_arr != 0), "No gradient through decay"

    def test_gradient_through_write_gate(self):
        """Gradients flow through write gates."""
        B, T, H, Dk, Dv = 1, 4, 1, 2, 2

        np.random.seed(0)
        queries = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        keys = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        values = mx.array(np.random.randn(B, T, H, Dv).astype(np.float32))
        decays = mx.array((0.95 + 0.05 * np.random.randn(B, T, H, Dk)).astype(np.float32))
        erases = mx.array(np.clip(np.random.randn(B, T, H, Dk), 0, 1).astype(np.float32))
        writes = mx.array(np.clip(np.random.randn(B, T, H, Dv), 0, 1).astype(np.float32))

        def loss_fn(w):
            outputs, _, _ = gdn2_sequence_with_chunks(
                queries, keys, values, decays, erases, w
            )
            return mx.mean(outputs)

        grad_fn = mx.grad(loss_fn)
        write_grads = grad_fn(writes)

        write_grad_arr = np.array(write_grads)
        assert not np.any(np.isnan(write_grad_arr))
        assert np.any(write_grad_arr != 0), "No gradient through write"

    def test_no_nan_in_gradients_long_sequence(self):
        """Long sequence gradients remain finite."""
        B, T, H, Dk, Dv = 1, 128, 1, 2, 2

        np.random.seed(0)
        queries = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32) * 0.1)
        keys = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32) * 0.1)
        values = mx.array(np.random.randn(B, T, H, Dv).astype(np.float32) * 0.1)
        decays = mx.array(
            np.clip(0.95 + 0.02 * np.random.randn(B, T, H, Dk), 0.9, 1.0).astype(
                np.float32
            )
        )
        erases = mx.array(np.clip(np.random.randn(B, T, H, Dk), 0, 1).astype(np.float32))
        writes = mx.array(np.clip(np.random.randn(B, T, H, Dv), 0, 1).astype(np.float32))

        def loss_fn(q):
            outputs, _, _ = gdn2_sequence_with_chunks(
                q, keys, values, decays, erases, writes, chunk_size=32
            )
            return mx.mean(outputs)

        for i in range(5):
            grad_fn = mx.grad(loss_fn)
            grads = grad_fn(queries)
            grad_arr = np.array(grads)
            assert not np.any(np.isnan(grad_arr)), f"NaN in gradients at iteration {i}"
            assert not np.any(np.isinf(grad_arr)), f"Inf in gradients at iteration {i}"


class TestChunkedMemoryEfficiency:
    """Verify memory benefits of chunking."""

    def test_chunk_states_are_sparse(self):
        """Fewer chunk states saved than full sequence length."""
        B, T, H, Dk, Dv = 1, 256, 1, 2, 2

        np.random.seed(0)
        queries = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        keys = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        values = mx.array(np.random.randn(B, T, H, Dv).astype(np.float32))
        decays = mx.array(np.ones((B, T, H, Dk), dtype=np.float32) * 0.95)
        erases = mx.array(np.zeros((B, T, H, Dk), dtype=np.float32))
        writes = mx.array(np.ones((B, T, H, Dv), dtype=np.float32))

        chunk_size = 64
        _, _, chunk_states = gdn2_sequence_with_chunks(
            queries, keys, values, decays, erases, writes, chunk_size=chunk_size
        )

        # Should have ~T/chunk_size + 1 states, not T states
        expected_max = (T // chunk_size) + 2
        assert len(chunk_states) <= expected_max
        assert len(chunk_states) < T  # Significantly less than full sequence
