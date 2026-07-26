from __future__ import annotations

import torch
from torch import nn

from .backends import BackendUnavailableError, UpstreamGDN2Mixer, gdn2_is_available
from .blocks import AnchorAttentionBlock, FeedForward, RMSNorm, RecurrentMixerBlock


class HybridLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        use_attention: bool,
        mixer_backend: str,
    ) -> None:
        super().__init__()
        self.mixer = build_mixer(mixer_backend, d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.attention = AnchorAttentionBlock(d_model, n_heads, dropout) if use_attention else None
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        if self.attention is not None:
            x = self.attention(x)
        return self.ffn(x)


class HybridLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        mixer_backend: str,
        attention_every: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [
                HybridLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_attention=((idx + 1) % attention_every == 0),
                    mixer_backend=mixer_backend,
                )
                for idx in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq = tokens.shape
        positions = torch.arange(seq, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def build_mixer(mixer_backend: str, d_model: int, n_heads: int, dropout: float) -> nn.Module:
    backend = mixer_backend.lower()
    if backend == "fallback":
        return RecurrentMixerBlock(d_model, dropout)
    if backend == "gdn2":
        return UpstreamGDN2Mixer(d_model=d_model, n_heads=n_heads)
    if backend == "auto":
        available, _ = gdn2_is_available()
        if available:
            return UpstreamGDN2Mixer(d_model=d_model, n_heads=n_heads)
        return RecurrentMixerBlock(d_model, dropout)
    raise BackendUnavailableError(f"Unknown mixer backend: {mixer_backend}")
