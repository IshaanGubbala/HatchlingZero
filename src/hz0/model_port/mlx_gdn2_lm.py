"""
MLX-native GDN-2 language models.

Ports HZ-36M and HZ-110M from PyTorch to MLX.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from src.hz0.metal_gdn2.kernels.gdn2_forward import GDN2MetalModule


class GDN2LanguageModel(nn.Module):
    """
    Dense GDN-2 hybrid language model.

    Architecture:
    - GDN-2 recurrent layers
    - Periodic exact attention layers
    - Dense SwiGLU MLPs
    - Layer normalization
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        model_dim: int = 768,
        num_layers: int = 24,
        num_heads: int = 12,
        gdn2_every: int = 3,  # Attention every N layers
        ffn_hidden: Optional[int] = None,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gdn2_every = gdn2_every
        self.ffn_hidden = ffn_hidden or model_dim * 4
        self.max_seq_len = max_seq_len

        # Embedding
        self.embedding = nn.Embedding(vocab_size, model_dim)

        # Layers
        self.layers = []
        for i in range(num_layers):
            is_attention = (i + 1) % gdn2_every == 0
            if is_attention:
                layer = AttentionBlock(model_dim, num_heads)
            else:
                layer = GDN2Block(model_dim, num_heads)
            self.layers.append(layer)

        # Final norm
        self.norm = nn.LayerNorm(model_dim)

        # LM head
        self.lm_head = nn.Linear(model_dim, vocab_size)

    def __call__(self, x: mx.array, memory: Optional[mx.array] = None) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Forward pass.

        Args:
            x: [B, T] token ids
            memory: [B, H, Dv, Dk] optional recurrent state

        Returns:
            logits: [B, T, vocab]
            memory: updated recurrent state (if applicable)
        """
        # Embed
        x = self.embedding(x)  # [B, T, D]

        # Process layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GDN2Block):
                x, memory = layer(x, memory)
            else:
                x = layer(x)

        # Final norm
        x = self.norm(x)

        # Logits
        logits = self.lm_head(x)

        return logits, memory


class GDN2Block(nn.Module):
    """GDN-2 recurrent block."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(dim)
        self.gdn2 = GDN2MetalModule(dim=dim, hidden_dim=dim, num_heads=num_heads)
        self.mlp = SwiGLUMLP(dim, dim * 4)

    def __call__(self, x: mx.array, memory: Optional[mx.array] = None) -> Tuple[mx.array, Optional[mx.array]]:
        """GDN-2 + MLP block."""
        # Norm + recurrence
        x_norm = self.norm(x)
        y, memory = self.gdn2(x_norm, state=memory)

        # Residual
        x = x + y

        # MLP
        mlp_out = self.mlp(self.norm(x))
        x = x + mlp_out

        return x, memory


class AttentionBlock(nn.Module):
    """Periodic exact attention block."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.mlp = SwiGLUMLP(dim, dim * 4)

    def __call__(self, x: mx.array) -> mx.array:
        """Multi-head causal attention + MLP."""
        B, T, D = x.shape
        H = self.num_heads

        # Norm + qkv
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        qkv = mx.reshape(qkv, (B, T, 3, H, self.head_dim))
        q, k, v = mx.split(qkv, 3, axis=2)
        q = mx.squeeze(q, axis=2)  # [B, T, H, Dk]
        k = mx.squeeze(k, axis=2)
        v = mx.squeeze(v, axis=2)

        # Attention: q[B,T,H,Dk] @ k[B,T,H,Dk].T -> [B,T,H,T]
        # Reshape for matmul: [B,T,H,Dk] @ [B,T,Dk,T]
        k_t = mx.transpose(k, axes=(0, 2, 3, 1))  # [B, H, Dk, T]
        q_reshaped = mx.transpose(q, axes=(0, 2, 1, 3))  # [B, H, T, Dk]
        scores = mx.matmul(q_reshaped, k_t) / mx.sqrt(mx.array(self.head_dim))  # [B, H, T, T]

        # Causal mask
        mask = mx.tril(mx.ones((T, T))) - 1
        mask = mask * -1e9
        scores = scores + mask[None, None, :, :]  # [B, H, T, T]

        attn = mx.softmax(scores, axis=-1)  # [B, H, T, T]

        # Output: [B, H, T, T] @ [B, H, T, Dv] -> [B, H, T, Dv]
        v_reshaped = mx.transpose(v, axes=(0, 2, 1, 3))  # [B, H, T, Dv]
        out = mx.matmul(attn, v_reshaped)  # [B, H, T, Dv]
        out = mx.transpose(out, axes=(0, 2, 1, 3))  # [B, T, H, Dv]
        out = mx.reshape(out, (B, T, D))
        out = self.out(out)

        # Residual + MLP
        x = x + out
        mlp_out = self.mlp(self.norm(x))
        x = x + mlp_out

        return x


class SwiGLUMLP(nn.Module):
    """Dense SwiGLU feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        """SwiGLU: (xW1 * SiLU) ⊙ (xW3) W2"""
        x = self.norm(x)
        gate = nn.SiLU()(self.w1(x))
        value = self.w3(x)
        x = gate * value
        x = self.w2(x)
        return x


def create_hz_36m_mlx() -> GDN2LanguageModel:
    """Create HZ-36M in MLX."""
    return GDN2LanguageModel(
        vocab_size=32768,
        model_dim=576,
        num_layers=24,
        num_heads=9,
        gdn2_every=3,
    )


def create_hz_110m_mlx() -> GDN2LanguageModel:
    """Create HZ-110M in MLX."""
    return GDN2LanguageModel(
        vocab_size=32768,
        model_dim=768,
        num_layers=32,
        num_heads=12,
        gdn2_every=3,
    )


def count_parameters(model: GDN2LanguageModel) -> int:
    """Count model parameters."""
    def count_params_recursive(module):
        total = 0
        if hasattr(module, 'parameters'):
            for param in module.parameters():
                if hasattr(param, 'size'):
                    total += param.size
                elif hasattr(param, 'shape'):
                    s = 1
                    for d in param.shape:
                        s *= d
                    total += s
        return total

    # Traverse model structure
    total = 0
    for module in [model.embedding, model.lm_head] + model.layers:
        total += count_params_recursive(module)
    return total
