"""
Tests for MLX-native GDN-2 models (Step 12: Port to MLX).
"""

import pytest
import numpy as np
import mlx.core as mx
from src.hz0.model_port.mlx_gdn2_lm import (
    GDN2LanguageModel,
    create_hz_36m_mlx,
    create_hz_110m_mlx,
    count_parameters,
)


class TestGDN2LanguageModel:
    """GDN-2 language model tests."""

    @pytest.fixture
    def small_model(self):
        """Small model for testing."""
        return GDN2LanguageModel(
            vocab_size=256,
            model_dim=64,
            num_layers=4,
            num_heads=2,
            gdn2_every=2,
        )

    def test_model_forward_shape(self, small_model):
        """Forward pass produces correct logits."""
        B, T = 2, 16
        vocab_size = 256

        input_ids = mx.array(np.random.randint(0, vocab_size, (B, T), dtype=np.int32))

        logits, memory = small_model(input_ids)

        assert logits.shape == (B, T, vocab_size)
        assert memory is not None  # Recurrent memory returned

    def test_model_forward_no_nan(self, small_model):
        """Forward pass produces valid outputs."""
        B, T = 1, 8
        vocab_size = 256

        input_ids = mx.array(np.random.randint(0, vocab_size, (B, T), dtype=np.int32))

        logits, _ = small_model(input_ids)

        logits_arr = np.array(logits)
        assert not np.any(np.isnan(logits_arr))
        assert not np.any(np.isinf(logits_arr))

    def test_recurrent_memory_shape(self, small_model):
        """Memory state has correct shape."""
        B, T = 2, 8
        vocab_size = 256

        input_ids = mx.array(np.random.randint(0, vocab_size, (B, T), dtype=np.int32))

        logits, memory = small_model(input_ids)

        if memory is not None:
            # Should be [B, H, Dv, Dk] or similar
            assert len(memory.shape) in [3, 4]

    def test_gradient_flow(self, small_model):
        """Gradients flow through model."""
        B, T = 1, 8
        vocab_size = 256

        input_ids = mx.array(np.random.randint(0, vocab_size, (B, T), dtype=np.int32))

        def loss_fn(x):
            logits, _ = small_model(x)
            return mx.mean(logits)

        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(input_ids)

        # Gradients should exist (embeddings are differentiable)
        assert grads is not None

    def test_hz_36m_creation(self):
        """Create HZ-36M model."""
        model = create_hz_36m_mlx()

        assert model.model_dim == 576
        assert model.num_layers == 24
        assert model.num_heads == 9

    def test_hz_110m_creation(self):
        """Create HZ-110M model."""
        model = create_hz_110m_mlx()

        assert model.model_dim == 768
        assert model.num_layers == 32
        assert model.num_heads == 12

    def test_hz_36m_forward(self):
        """HZ-36M can run forward pass."""
        model = create_hz_36m_mlx()

        input_ids = mx.array(np.random.randint(0, 32768, (1, 16), dtype=np.int32))

        logits, memory = model(input_ids)

        assert logits.shape == (1, 16, 32768)

    def test_hz_110m_forward(self):
        """HZ-110M can run forward pass."""
        model = create_hz_110m_mlx()

        input_ids = mx.array(np.random.randint(0, 32768, (1, 8), dtype=np.int32))

        logits, memory = model(input_ids)

        assert logits.shape == (1, 8, 32768)

    def test_model_layers_structure(self, small_model):
        """Model has correct layer structure."""
        assert len(small_model.layers) == small_model.num_layers

        # Check that attention layers exist
        num_attention = sum(1 for l in small_model.layers if hasattr(l, 'qkv'))
        expected_attention = small_model.num_layers // small_model.gdn2_every
        assert num_attention == expected_attention

    def test_layer_normalization(self, small_model):
        """Layer norms are present."""
        assert hasattr(small_model, 'norm')

        for layer in small_model.layers:
            assert hasattr(layer, 'norm')

    def test_long_sequence(self):
        """Handle longer sequences."""
        model = GDN2LanguageModel(
            vocab_size=256,
            model_dim=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=512,
        )

        input_ids = mx.array(np.random.randint(0, 256, (1, 128), dtype=np.int32))

        logits, memory = model(input_ids)

        assert logits.shape == (1, 128, 256)

    def test_batch_processing(self, small_model):
        """Process multiple examples in batch."""
        for batch_size in [1, 2, 4, 8]:
            input_ids = mx.array(
                np.random.randint(0, 256, (batch_size, 16), dtype=np.int32)
            )

            logits, _ = small_model(input_ids)

            assert logits.shape == (batch_size, 16, 256)
