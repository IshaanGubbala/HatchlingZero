"""Single-layer fast weights prototype for test-time adaptation."""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional


class FastWeightLinear(nn.Module):
    """Linear layer with session-local fast weights.

    Base weights stay frozen. Fast weights (delta) are learned per-session
    during inference via gradient steps on context examples.
    """

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        # Base weights (frozen during inference)
        self.weight = mx.random.normal((input_dim, output_dim)) / (input_dim ** 0.5)
        if use_bias:
            self.bias = mx.zeros((output_dim,))

        # Fast weights (delta, session-local)
        self.fast_weight = mx.zeros((input_dim, output_dim))
        if use_bias:
            self.fast_bias = mx.zeros((output_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: y = x @ (W_base + W_fast) + (b_base + b_fast)"""
        # Effective weight = base + fast
        w_eff = self.weight + self.fast_weight
        out = mx.matmul(x, w_eff)

        if self.use_bias:
            b_eff = self.bias + self.fast_bias
            out = out + b_eff

        return out

    def reset_fast_weights(self):
        """Reset fast weights to zero (new session)."""
        self.fast_weight = mx.zeros_like(self.fast_weight)
        if self.use_bias:
            self.fast_bias = mx.zeros_like(self.fast_bias)

    def get_fast_weight_params(self):
        """Return dict of fast-weight parameters for gradient computation."""
        params = {"fast_weight": self.fast_weight}
        if self.use_bias:
            params["fast_bias"] = self.fast_bias
        return params

    def update_fast_weights(self, grads: dict, lr: float = 0.01):
        """Apply gradient step to fast weights."""
        if "fast_weight" in grads:
            self.fast_weight = self.fast_weight - lr * grads["fast_weight"]
        if "fast_bias" in grads:
            self.fast_bias = self.fast_bias - lr * grads["fast_bias"]


class FastWeightAttention(nn.Module):
    """Attention layer with fast-weight Q, K, V projections."""

    def __init__(self, d_model: int, num_heads: int, use_fast_weights: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q, K, V projections with optional fast weights
        if use_fast_weights:
            self.q_proj = FastWeightLinear(d_model, d_model)
            self.k_proj = FastWeightLinear(d_model, d_model)
            self.v_proj = FastWeightLinear(d_model, d_model)
        else:
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.use_fast_weights = use_fast_weights

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """Single-token forward pass (for streaming decode)."""
        B, D = x.shape  # [1, d_model] for streaming

        Q = self.q_proj(x)  # [1, d_model]
        K = self.k_proj(x)  # [1, d_model]
        V = self.v_proj(x)  # [1, d_model]

        # Reshape to [1, num_heads, head_dim]
        Q = Q.reshape(B, self.num_heads, self.head_dim)
        K = K.reshape(B, self.num_heads, self.head_dim)
        V = V.reshape(B, self.num_heads, self.head_dim)

        # Scaled dot-product (single-token, uses KV cache externally)
        scores = mx.matmul(Q, K.transpose(0, 2, 1)) / (self.head_dim ** 0.5)
        attn = mx.softmax(scores, axis=-1)

        out = mx.matmul(attn, V)  # [1, num_heads, head_dim]
        out = out.reshape(B, D)   # [1, d_model]
        out = self.out_proj(out)

        return out

    def reset_fast_weights(self):
        """Reset all fast-weight projections."""
        if self.use_fast_weights:
            self.q_proj.reset_fast_weights()
            self.k_proj.reset_fast_weights()
            self.v_proj.reset_fast_weights()

    def get_fast_weight_params(self) -> dict:
        """Collect all fast-weight parameters."""
        params = {}
        if self.use_fast_weights:
            params["q"] = self.q_proj.get_fast_weight_params()
            params["k"] = self.k_proj.get_fast_weight_params()
            params["v"] = self.v_proj.get_fast_weight_params()
        return params
