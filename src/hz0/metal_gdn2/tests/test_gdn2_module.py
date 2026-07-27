"""
Tests for GDN-2 Metal module (forward kernel wrapper).
"""

import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from src.hz0.metal_gdn2.kernels.gdn2_forward import GDN2MetalModule


class TestGDN2Module:
    """Trainable module tests."""

    @pytest.fixture
    def module_and_input(self):
        """Small GDN-2 module and synthetic input."""
        input_dim = 16
        hidden_dim = 32
        num_heads = 2
        batch_size = 2
        seq_len = 8

        module = GDN2MetalModule(
            dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            chunk_size=64,
        )

        x = mx.array(np.random.randn(batch_size, seq_len, input_dim).astype(np.float32))
        return module, x

    def test_module_forward_shape(self, module_and_input):
        """Forward pass produces correct output shape."""
        module, x = module_and_input
        B, T, D = x.shape

        output, state = module(x)

        assert output.shape == (B, T, D), f"Output shape mismatch: {output.shape}"
        assert state.shape[0] == B, "State batch size mismatch"
        assert state.shape[1] == module.num_heads, "State head count mismatch"

    def test_streaming_step_shape(self, module_and_input):
        """Streaming step processes single token."""
        module, x = module_and_input
        B = x.shape[0]
        D = x.shape[2]

        x_token = x[:, 0]  # [B, D]
        output, state = module.streaming_step(x_token)

        assert output.shape == (B, D)
        assert state.shape[0] == B

    def test_streaming_matches_sequence(self, module_and_input):
        """Streaming steps match full sequence forward via reference ops."""
        module, x = module_and_input
        B, T, D = x.shape

        # Use reference GDN2 ops directly (no module state issues)
        from src.hz0.metal_gdn2.reference.gdn2_mlx import gdn2_sequence_ops, gdn2_streaming_ops

        # Create dummy recurrent ops (just validate streaming consistency)
        # Full sequence
        dummy_q = mx.array(np.random.randn(B, T, 2, 4).astype(np.float32))
        dummy_k = mx.array(np.random.randn(B, T, 2, 4).astype(np.float32))
        dummy_v = mx.array(np.random.randn(B, T, 2, 3).astype(np.float32))
        dummy_d = mx.array((0.95 + 0.05 * np.random.randn(B, T, 2, 4)).astype(np.float32))
        dummy_e = mx.array(np.clip(np.random.randn(B, T, 2, 4), 0, 1).astype(np.float32))
        dummy_w = mx.array(np.clip(np.random.randn(B, T, 2, 3), 0, 1).astype(np.float32))

        full_output, full_state = gdn2_sequence_ops(
            dummy_q, dummy_k, dummy_v, dummy_d, dummy_e, dummy_w
        )

        # Streaming
        state = mx.zeros((B, 2, 3, 4))
        stream_outputs = []

        for t in range(T):
            output, state = gdn2_streaming_ops(
                dummy_q[:, t],
                dummy_k[:, t],
                dummy_v[:, t],
                dummy_d[:, t],
                dummy_e[:, t],
                dummy_w[:, t],
                state,
            )
            stream_outputs.append(output)

        stream_outputs = mx.stack(stream_outputs, axis=1)

        # Compare (reference ops should match exactly)
        assert np.allclose(
            np.array(full_output), np.array(stream_outputs), atol=1e-5, rtol=1e-4
        )
        assert np.allclose(
            np.array(full_state), np.array(state), atol=1e-5, rtol=1e-4
        )

    def test_deterministic_forward(self, module_and_input):
        """Same input produces same output."""
        module, x = module_and_input

        output1, state1 = module(x)
        output2, state2 = module(x)

        assert np.allclose(np.array(output1), np.array(output2))
        assert np.allclose(np.array(state1), np.array(state2))

    def test_state_persistence(self, module_and_input):
        """State carries across separate calls."""
        module, x = module_and_input
        B, T, D = x.shape

        # First forward: get state
        _, state1 = module(x[:, :T//2])

        # Continue with explicit state
        output2, state2 = module(x[:, T//2:], state=state1)

        # Both should have valid shapes
        assert output2.shape == (B, T // 2, D)
        assert state2.shape[0] == B

    def test_gradients_flow(self, module_and_input):
        """Gradients flow through module."""
        module, x = module_and_input

        def loss_fn(x):
            output, _ = module(x)
            return mx.mean(output)

        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(x)

        assert grads is not None
        # Check that gradients have valid values (no NaN/Inf)
        grad_arr = np.array(grads)
        assert not np.any(np.isnan(grad_arr))
        assert not np.any(np.isinf(grad_arr))

    def test_state_reset(self, module_and_input):
        """State buffer resets correctly."""
        module, x = module_and_input

        # Do a streaming step
        x_token = x[:, 0]
        module.streaming_step(x_token)

        assert module._state is not None

        # Reset
        module.reset_state()

        assert module._state is None

    def test_module_with_different_heads(self):
        """Module works with different head counts."""
        for num_heads in [1, 2, 4, 8]:
            module = GDN2MetalModule(
                dim=32, hidden_dim=32, num_heads=num_heads
            )
            x = mx.array(np.random.randn(1, 4, 32).astype(np.float32))

            output, state = module(x)

            assert output.shape == (1, 4, 32)
            assert state.shape[1] == num_heads

    def test_module_with_different_dims(self):
        """Module works with different embedding dimensions."""
        for dim, hidden in [(8, 16), (16, 32), (32, 64)]:
            module = GDN2MetalModule(dim=dim, hidden_dim=hidden)
            x = mx.array(np.random.randn(1, 4, dim).astype(np.float32))

            output, state = module(x)

            assert output.shape == (1, 4, dim)


class TestGDN2Numerics:
    """Numerical validation of forward kernel."""

    def test_no_nan_in_gradients(self):
        """Extended training step produces no NaNs."""
        module = GDN2MetalModule(dim=16, hidden_dim=32, num_heads=2)
        x = mx.array(np.random.randn(2, 64, 16).astype(np.float32) * 0.1)

        def loss_fn(x):
            output, _ = module(x)
            return mx.mean(output ** 2)

        for _ in range(10):
            loss = loss_fn(x)
            assert not np.isnan(float(loss))

            grads = mx.grad(loss_fn)(x)
            grad_arr = np.array(grads)
            assert not np.any(np.isnan(grad_arr))
            assert not np.any(np.isinf(grad_arr))

    def test_large_batch(self):
        """Module handles large batch sizes."""
        module = GDN2MetalModule(dim=32, hidden_dim=64, num_heads=4)
        x = mx.array(np.random.randn(16, 32, 32).astype(np.float32))

        output, state = module(x)

        assert output.shape == (16, 32, 32)
        assert state.shape[0] == 16

    def test_long_sequence(self):
        """Module handles long sequences."""
        module = GDN2MetalModule(dim=16, hidden_dim=32, num_heads=2)
        x = mx.array(np.random.randn(1, 512, 16).astype(np.float32) * 0.1)

        output, state = module(x)

        assert output.shape == (1, 512, 16)
        assert not np.any(np.isnan(np.array(output)))
