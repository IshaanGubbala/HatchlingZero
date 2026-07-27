"""
Tests for MLX GDN-2 reference implementation.
"""

import pytest
import numpy as np
import mlx.core as mx
from src.hz0.metal_gdn2.reference.gdn2_mlx import (
    gdn2_step,
    gdn2_sequence_ops,
    gdn2_streaming_ops,
)
from src.hz0.metal_gdn2.reference.gdn2_numpy import gdn2_sequence_numpy


def numerical_gradient(
    fn,
    state: mx.array,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Finite-difference gradient via MLX autodiff.
    """
    grad_fn = mx.grad(fn)
    return np.array(grad_fn(state))


class TestGDN2Reference:
    """Core reference implementation tests."""

    @pytest.fixture
    def synthetic_sequence(self):
        """Small synthetic sequence for testing."""
        B, T, H, Dk, Dv = 2, 8, 2, 4, 3
        np.random.seed(42)

        queries = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        keys = mx.array(np.random.randn(B, T, H, Dk).astype(np.float32))
        values = mx.array(np.random.randn(B, T, H, Dv).astype(np.float32))
        decays = mx.array(0.95 + 0.05 * np.random.randn(B, T, H, Dk).astype(np.float32))
        erases = mx.array(np.clip(np.random.randn(B, T, H, Dk), 0, 1).astype(np.float32))
        writes = mx.array(np.clip(np.random.randn(B, T, H, Dv), 0, 1).astype(np.float32))

        return queries, keys, values, decays, erases, writes

    def test_step_output_shape(self, synthetic_sequence):
        """Output has correct shape."""
        queries, keys, values, decays, erases, writes = synthetic_sequence
        B, T, H, Dk = queries.shape[:2] + (queries.shape[2], queries.shape[3])
        _, _, Dv = values.shape[-3:]

        state = mx.zeros((B, H, Dv, Dk), dtype=queries.dtype)

        q_0 = queries[:, 0]
        k_0 = keys[:, 0]
        v_0 = values[:, 0]
        d_0 = decays[:, 0]
        e_0 = erases[:, 0]
        w_0 = writes[:, 0]

        new_state, output = gdn2_step(state, q_0, k_0, v_0, d_0, e_0, w_0)

        assert new_state.shape == (B, H, Dv, Dk)
        assert output.shape == (B, H, Dv)

    def test_sequence_vs_streaming(self, synthetic_sequence):
        """Full sequence and streaming produce same output."""
        queries, keys, values, decays, erases, writes = synthetic_sequence
        B, T, H, Dk = queries.shape
        _, _, Dv = values.shape[-3:]

        # Full sequence
        full_outputs, full_state = gdn2_sequence_ops(
            queries, keys, values, decays, erases, writes
        )

        # Streaming
        state = mx.zeros((B, H, Dv, Dk), dtype=queries.dtype)
        stream_outputs = []

        for t in range(T):
            output, state = gdn2_streaming_ops(
                queries[:, t],
                keys[:, t],
                values[:, t],
                decays[:, t],
                erases[:, t],
                writes[:, t],
                state,
            )
            stream_outputs.append(output)

        stream_outputs = mx.stack(stream_outputs, axis=1)

        # Compare
        assert np.allclose(
            np.array(full_outputs), np.array(stream_outputs), atol=1e-5
        ), "Full sequence != streaming output"
        assert np.allclose(
            np.array(full_state), np.array(state), atol=1e-5
        ), "Full state != streaming state"

    def test_initial_state_zero(self, synthetic_sequence):
        """Explicit zero initial state matches default."""
        queries, keys, values, decays, erases, writes = synthetic_sequence
        B, T, H, Dk = queries.shape
        _, _, Dv = values.shape[-3:]

        zero_state = mx.zeros((B, H, Dv, Dk), dtype=queries.dtype)

        out_default, state_default = gdn2_sequence_ops(
            queries, keys, values, decays, erases, writes, initial_state=None
        )
        out_explicit, state_explicit = gdn2_sequence_ops(
            queries, keys, values, decays, erases, writes, initial_state=zero_state
        )

        assert np.allclose(np.array(out_default), np.array(out_explicit), atol=1e-6)
        assert np.allclose(np.array(state_default), np.array(state_explicit), atol=1e-6)

    @pytest.mark.parametrize("seq_len", [1, 2, 63, 64, 65])
    def test_variable_sequence_lengths(self, seq_len):
        """Handle various sequence lengths."""
        B, H, Dk, Dv = 1, 1, 2, 2

        queries = mx.array(np.random.randn(B, seq_len, H, Dk).astype(np.float32))
        keys = mx.array(np.random.randn(B, seq_len, H, Dk).astype(np.float32))
        values = mx.array(np.random.randn(B, seq_len, H, Dv).astype(np.float32))
        decays = mx.array(np.ones((B, seq_len, H, Dk), dtype=np.float32) * 0.9)
        erases = mx.array(np.zeros((B, seq_len, H, Dk), dtype=np.float32))
        writes = mx.array(np.ones((B, seq_len, H, Dv), dtype=np.float32))

        outputs, state = gdn2_sequence_ops(
            queries, keys, values, decays, erases, writes
        )

        assert outputs.shape == (B, seq_len, H, Dv)
        assert state.shape == (B, H, Dv, Dk)

    def test_no_writes_zero_erase(self, synthetic_sequence):
        """No writes + zero erase = only decay."""
        queries, keys, values, decays, erases, writes = synthetic_sequence
        B, T, H, Dk = queries.shape
        _, _, Dv = values.shape[-3:]

        # Zero writes, zero erases
        zero_writes = mx.zeros_like(writes)
        zero_erases = mx.zeros_like(erases)

        # Initialize state to known value
        initial_state = mx.ones((B, H, Dv, Dk), dtype=queries.dtype)

        outputs, final_state = gdn2_sequence_ops(
            queries, keys, values, decays, zero_erases, zero_writes, initial_state
        )

        # With only decay and no write/erase, state should approach zero
        expected_state = initial_state
        for t in range(T):
            expected_state = expected_state * decays[:, t][:, :, None, :]

        assert np.allclose(
            np.array(final_state), np.array(expected_state), atol=1e-5
        ), "Decay-only evolution incorrect"

    def test_identity_decay_no_erase_write(self):
        """Identity decay (1.0), no erase, full write → state accumulated."""
        B, T, H, Dk, Dv = 1, 3, 1, 2, 2
        np.random.seed(0)

        queries = mx.array(np.ones((B, T, H, Dk), dtype=np.float32))
        keys = mx.array(np.ones((B, T, H, Dk), dtype=np.float32))
        values = mx.array(np.ones((B, T, H, Dv), dtype=np.float32))
        decays = mx.array(np.ones((B, T, H, Dk), dtype=np.float32))
        erases = mx.array(np.zeros((B, T, H, Dk), dtype=np.float32))
        writes = mx.array(np.ones((B, T, H, Dv), dtype=np.float32))

        outputs, final_state = gdn2_sequence_ops(
            queries, keys, values, decays, erases, writes
        )

        # Each step: state += 1 * 1 * 1 = state + 1
        # After 3 steps, state[v, k] should be 3
        expected = np.ones((B, H, Dv, Dk), dtype=np.float32) * T

        assert np.allclose(
            np.array(final_state), expected, atol=1e-5
        ), "Accumulation incorrect"

    def test_persistent_memory_across_calls(self):
        """State carries across multiple forward passes (streaming mode)."""
        B, H, Dk, Dv = 1, 1, 2, 2
        np.random.seed(0)

        # First call: write specific value
        q1 = mx.array(np.ones((B, H, Dk), dtype=np.float32))
        k1 = mx.array(np.ones((B, H, Dk), dtype=np.float32))
        v1 = mx.array(np.array([[[1.0, 2.0]]], dtype=np.float32))
        decay1 = mx.array(np.ones((B, H, Dk), dtype=np.float32))
        erase1 = mx.array(np.zeros((B, H, Dk), dtype=np.float32))
        write1 = mx.array(np.ones((B, H, Dv), dtype=np.float32))

        state = mx.zeros((B, H, Dv, Dk), dtype=mx.float32)
        output1, state = gdn2_streaming_ops(q1, k1, v1, decay1, erase1, write1, state)

        # Second call: read same key
        q2 = mx.array(np.ones((B, H, Dk), dtype=np.float32))
        k2 = mx.array(np.ones((B, H, Dk), dtype=np.float32))
        v2 = mx.array(np.zeros((B, H, Dv), dtype=np.float32))
        decay2 = mx.array(np.ones((B, H, Dk), dtype=np.float32))
        erase2 = mx.array(np.zeros((B, H, Dk), dtype=np.float32))
        write2 = mx.array(np.zeros((B, H, Dv), dtype=np.float32))

        output2, state2 = gdn2_streaming_ops(q2, k2, v2, decay2, erase2, write2, state)

        # output2 should reflect written value from step 1
        assert output2.shape == (B, H, Dv)
        # With identity decay/query and stored v1, output should carry signal
        assert np.any(np.array(output2) != 0), "No memory retention"


class TestGDN2Numerics:
    """Numerical stability and correctness tests."""

    def test_mlx_vs_numpy_small_sequence(self):
        """MLX FP32 vs NumPy FP64 on small sequence."""
        B, T, H, Dk, Dv = 1, 4, 1, 2, 2
        np.random.seed(0)

        # Create FP32 and FP64 versions
        queries_f32 = np.random.randn(B, T, H, Dk).astype(np.float32)
        keys_f32 = np.random.randn(B, T, H, Dk).astype(np.float32)
        values_f32 = np.random.randn(B, T, H, Dv).astype(np.float32)
        decays_f32 = (0.9 + 0.05 * np.random.randn(B, T, H, Dk)).astype(np.float32)
        erases_f32 = np.clip(np.random.randn(B, T, H, Dk), 0, 1).astype(np.float32)
        writes_f32 = np.clip(np.random.randn(B, T, H, Dv), 0, 1).astype(np.float32)

        queries_f64 = queries_f32.astype(np.float64)
        keys_f64 = keys_f32.astype(np.float64)
        values_f64 = values_f32.astype(np.float64)
        decays_f64 = decays_f32.astype(np.float64)
        erases_f64 = erases_f32.astype(np.float64)
        writes_f64 = writes_f32.astype(np.float64)

        # MLX
        mlx_out, mlx_state = gdn2_sequence_ops(
            mx.array(queries_f32),
            mx.array(keys_f32),
            mx.array(values_f32),
            mx.array(decays_f32),
            mx.array(erases_f32),
            mx.array(writes_f32),
        )

        # NumPy
        numpy_out, numpy_state = gdn2_sequence_numpy(
            queries_f64, keys_f64, values_f64, decays_f64, erases_f64, writes_f64
        )

        # Compare (allow larger tolerance due to FP32 vs FP64)
        assert np.allclose(
            np.array(mlx_out), numpy_out, atol=1e-3, rtol=1e-3
        ), "MLX and NumPy outputs diverge"
        assert np.allclose(
            np.array(mlx_state), numpy_state, atol=1e-3, rtol=1e-3
        ), "MLX and NumPy states diverge"

    def test_no_nan_long_sequence(self):
        """1000-step sequence produces no NaNs."""
        B, H, Dk, Dv = 1, 1, 2, 2

        np.random.seed(0)
        queries = mx.array(np.random.randn(B, 1000, H, Dk).astype(np.float32) * 0.1)
        keys = mx.array(np.random.randn(B, 1000, H, Dk).astype(np.float32) * 0.1)
        values = mx.array(np.random.randn(B, 1000, H, Dv).astype(np.float32) * 0.1)
        decays = mx.array(
            np.clip(0.95 + 0.02 * np.random.randn(B, 1000, H, Dk), 0.9, 1.0).astype(
                np.float32
            )
        )
        erases = mx.array(np.clip(np.random.randn(B, 1000, H, Dk), 0, 1).astype(np.float32))
        writes = mx.array(np.clip(np.random.randn(B, 1000, H, Dv), 0, 1).astype(np.float32))

        outputs, state = gdn2_sequence_ops(
            queries, keys, values, decays, erases, writes
        )

        assert not np.any(np.isnan(np.array(outputs))), "NaN in outputs"
        assert not np.any(np.isnan(np.array(state))), "NaN in state"
