from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reference.hz0a_gdn2_reference import CausalSelfAttention, RMSNorm, SwiGLU, cross_entropy_loss


@dataclass
class TransformerBlock:
    norm1: RMSNorm
    norm2: RMSNorm
    attention: CausalSelfAttention
    mlp: SwiGLU

    @classmethod
    def init(cls, rng: np.random.Generator, d_model: int, num_heads: int, d_ff: int) -> "TransformerBlock":
        return cls(
            norm1=RMSNorm.init(d_model),
            norm2=RMSNorm.init(d_model),
            attention=CausalSelfAttention.init(rng, d_model, num_heads),
            mlp=SwiGLU.init(rng, d_model, d_ff),
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


@dataclass
class TinyTransformerModel:
    vocab_size: int
    d_model: int
    embedding: np.ndarray
    blocks: list[TransformerBlock]
    final_norm: RMSNorm

    @classmethod
    def init(cls, rng_seed: int, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int) -> "TinyTransformerModel":
        rng = np.random.default_rng(rng_seed)
        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            embedding=rng.normal(0.0, 0.02, size=(vocab_size, d_model)).astype(np.float32),
            blocks=[TransformerBlock.init(rng, d_model, num_heads, d_ff) for _ in range(num_layers)],
            final_norm=RMSNorm.init(d_model),
        )

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        x = self.embedding[token_ids]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return np.einsum("btd,vd->btv", x, self.embedding)

    def loss(self, token_ids: np.ndarray, targets: np.ndarray) -> float:
        return cross_entropy_loss(self(token_ids), targets)
