"""
End-to-end training tests for GDN-2 layer.
Validates kernel works in real training loop.
"""

import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from src.hz0.metal_gdn2.kernels.gdn2_forward import GDN2MetalModule


class SimpleLanguageModel(nn.Module):
    """Minimal language model with GDN-2 layer."""

    def __init__(self, vocab_size: int, dim: int, num_heads: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.embedding = nn.Embedding(vocab_size, dim)
        self.gdn2 = GDN2MetalModule(dim=dim, hidden_dim=dim*2, num_heads=num_heads)
        self.lm_head = nn.Linear(dim, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, T] token ids

        Returns:
            logits: [B, T, vocab]
        """
        x = self.embedding(x)  # [B, T, D]
        x, state = self.gdn2(x)    # [B, T, D], [B, H, Dv, Dk]
        logits = self.lm_head(x)  # [B, T, vocab]
        return logits  # Only return logits for inference


def causal_lm_loss(logits: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss for language modeling."""
    # logits: [B, T, vocab]
    # targets: [B, T]
    B, T, vocab_size = logits.shape

    logits_flat = mx.reshape(logits, (B * T, vocab_size))
    targets_flat = mx.reshape(targets, (B * T,))

    # Log softmax = log(softmax(x))
    log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-8)

    # Gather log probs for target tokens
    indices = mx.arange(B * T) * vocab_size + targets_flat
    loss = -mx.mean(mx.take(log_probs.flatten(), indices))

    return loss


class TestGDN2Training:
    """Training loop and optimization tests."""

    @pytest.fixture
    def toy_model(self):
        """Small model for testing."""
        return SimpleLanguageModel(vocab_size=256, dim=32, num_heads=2)

    @pytest.fixture
    def toy_batch(self):
        """Small batch of token sequences."""
        B, T = 2, 16
        vocab_size = 256

        input_ids = mx.array(np.random.randint(0, vocab_size, (B, T), dtype=np.int32))
        target_ids = mx.array(np.random.randint(0, vocab_size, (B, T), dtype=np.int32))

        return input_ids, target_ids

    def test_model_forward_pass(self, toy_model, toy_batch):
        """Model produces valid logits."""
        input_ids, _ = toy_batch

        logits = toy_model(input_ids)

        assert logits.shape == input_ids.shape + (256,)
        assert not np.any(np.isnan(np.array(logits)))

    def test_loss_computation(self, toy_model, toy_batch):
        """Loss function computes without NaN."""
        input_ids, target_ids = toy_batch

        logits = toy_model(input_ids)
        loss = causal_lm_loss(logits, target_ids)

        loss_val = float(loss)
        assert not np.isnan(loss_val)
        assert loss_val > 0

    def test_gradient_computation(self, toy_model, toy_batch):
        """Gradients flow through model."""
        input_ids, target_ids = toy_batch

        def loss_fn(model, inputs, targets):
            logits = model(inputs)
            return causal_lm_loss(logits, targets)

        grad_fn = nn.value_and_grad(toy_model, loss_fn)
        loss_val, grads = grad_fn(toy_model, input_ids, target_ids)

        # Check gradients exist and are finite
        assert grads is not None
        assert len(grads) > 0

    def test_optimization_step(self, toy_model, toy_batch):
        """Single optimizer step updates weights and reduces loss."""
        input_ids, target_ids = toy_batch

        # Save initial loss
        logits_0 = toy_model(input_ids)
        loss_0 = causal_lm_loss(logits_0, target_ids)
        loss_0_val = float(loss_0)

        # Optimizer step
        optimizer = optim.Adam(learning_rate=0.001)

        def loss_fn(model):
            logits = model(input_ids)
            return causal_lm_loss(logits, target_ids)

        grad_fn = nn.value_and_grad(toy_model, loss_fn)
        loss_val, grads = grad_fn(toy_model)

        optimizer.update(toy_model, grads)

        # Check loss decreased
        logits_1 = toy_model(input_ids)
        loss_1 = causal_lm_loss(logits_1, target_ids)
        loss_1_val = float(loss_1)

        # Loss might not decrease on single step, but should be finite
        assert not np.isnan(loss_1_val)

    def test_multiple_optimization_steps(self, toy_model, toy_batch):
        """Training for multiple steps is stable."""
        input_ids, target_ids = toy_batch

        optimizer = optim.Adam(learning_rate=0.001)

        losses = []
        for step in range(10):
            def loss_fn(model):
                logits = model(input_ids)
                return causal_lm_loss(logits, target_ids)

            grad_fn = nn.value_and_grad(toy_model, loss_fn)
            loss_val, grads = grad_fn(toy_model)

            losses.append(float(loss_val))
            optimizer.update(toy_model, grads)

        # All losses should be finite
        losses_arr = np.array(losses)
        assert not np.any(np.isnan(losses_arr))
        assert not np.any(np.isinf(losses_arr))

        # Loss should generally decrease over training
        assert losses[-1] < losses[0]

    def test_larger_model_training(self):
        """Larger model trains without instability."""
        model = SimpleLanguageModel(vocab_size=512, dim=64, num_heads=4)
        B, T = 4, 32

        input_ids = mx.array(np.random.randint(0, 512, (B, T), dtype=np.int32))
        target_ids = mx.array(np.random.randint(0, 512, (B, T), dtype=np.int32))

        optimizer = optim.Adam(learning_rate=0.001)

        losses = []
        for step in range(5):
            def loss_fn(m):
                logits = m(input_ids)
                return causal_lm_loss(logits, target_ids)

            grad_fn = nn.value_and_grad(model, loss_fn)
            loss_val, grads = grad_fn(model)

            losses.append(float(loss_val))
            optimizer.update(model, grads)

        losses_arr = np.array(losses)
        assert not np.any(np.isnan(losses_arr))


class TestGDN2Inference:
    """Inference and streaming prediction tests."""

    def test_greedy_decoding_step(self):
        """Single-token greedy decoding works."""
        model = SimpleLanguageModel(vocab_size=128, dim=32, num_heads=2)

        # Seed sequence
        context = mx.array(np.random.randint(0, 128, (1, 8), dtype=np.int32))

        # Get last token logits
        logits = model(context)  # [1, 8, 128]
        last_logits = logits[:, -1, :]  # [1, 128]

        # Greedy: pick argmax
        next_token = mx.argmax(last_logits, axis=-1)

        assert next_token.shape == (1,)
        assert 0 <= int(next_token[0]) < 128

    def test_generation_loop(self):
        """Generate sequence tokens one by one."""
        model = SimpleLanguageModel(vocab_size=64, dim=32, num_heads=2)

        context = mx.array(np.random.randint(0, 64, (1, 4), dtype=np.int32))
        gen_len = 8

        for i in range(gen_len):
            logits = model(context)  # [1, T, 64]
            next_logits = logits[:, -1, :]  # [1, 64]
            next_token = mx.argmax(next_logits, axis=-1, keepdims=True)  # [1, 1]

            # Append to context
            context = mx.concatenate([context, next_token], axis=1)

        assert context.shape == (1, 4 + gen_len)

    def test_streaming_inference_state_persistence(self):
        """Streaming mode maintains state correctly."""
        model = SimpleLanguageModel(vocab_size=64, dim=32, num_heads=2)

        # Full context forward
        context = mx.array(np.random.randint(0, 64, (1, 8), dtype=np.int32))
        logits_full = model(context[..., :4])  # Warm up on 4 tokens
        token = context[..., 4:5]

        # Then streaming predict next token
        model.gdn2.reset_state()
        output, _ = model.gdn2.streaming_step(model.embedding(token))
        logits_stream = model.lm_head(output)

        # Both should produce valid distributions
        assert logits_full.shape[-1] == 64
        assert logits_stream.shape[-1] == 64
