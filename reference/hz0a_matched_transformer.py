"""Parameter-matched causal transformer for the HZ-0A comparison protocol."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from reference.hz0a_torch_model import _make_linear


class MatchedTransformerConfig:
    def __init__(self, values: dict):
        self.use_bitlinear = False  # HZ-0H T-lane default; see docs/restart/hz0h_ternary_training_design.md
        # Off by default -- this class backs many established HZ-0A/HZ-0H
        # results (T1/T2 ternary comparisons, HZ-0A hybrid-vs-transformer
        # controls) that were run and reported WITHOUT any positional
        # encoding at all. Adding RoPE unconditionally would silently
        # change what every one of those historical results measured.
        # Opt in explicitly per run instead -- see
        # docs/restart/hz0h_initial_bdh_vs_transformer_pilot_results.md's
        # "real, current gap" finding for why this was added.
        self.use_rope = False
        self.__dict__.update(values)

    @classmethod
    def from_json(cls, path: str | Path) -> "MatchedTransformerConfig":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


class BiasFreeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


def _rope_cos_sin(seq_len: int, head_dim: int, device, dtype, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard (Llama/GPT-NeoX/Qwen-style) "rotate-half" RoPE, computed
    fresh per forward call rather than cached in a buffer -- this
    project's sequence lengths are small (<=1024) so the recompute cost
    is negligible, and it sidesteps any buffer-resize/device/dtype-cast
    bookkeeping entirely. Deliberately NOT reusing
    reference/hz0h_bdh_torch.py's Attention.phases_cos_sin: that formula
    is BDH's own real, verified-against-upstream RoPE variant (quantized
    frequency bins, cycles-to-radians wrap) -- this is a standard,
    independent implementation for the Transformer baseline, not a port
    of BDH's version, so this comparison isn't "BDH's RoPE vs no RoPE"
    but "BDH's RoPE vs a normal modern Transformer's RoPE."
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, freqs)  # (T, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)  # (T, head_dim)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, heads, T, head_dim); cos/sin: (T, head_dim), broadcast over B/heads.
    return x * cos + _rotate_half(x) * sin


class MatchedTransformerBlock(nn.Module):
    def __init__(self, config: MatchedTransformerConfig):
        super().__init__()
        d, ff = config.d_model, config.d_ff
        use_bitlinear = getattr(config, "use_bitlinear", False)
        self.norm1, self.norm2 = BiasFreeRMSNorm(d), BiasFreeRMSNorm(d)
        # HZ-0H T-lane (docs/restart/hz0h_ternary_training_design.md): same
        # absmean-ternary-STE `_make_linear`/`BitLinear` swap already used by
        # reference/hz0a_torch_model.py's hybrid model -- reused directly
        # rather than duplicated, per the design memo's own "same
        # _make_linear-style swap as the hybrid model" contract row.
        # `norm1`/`norm2`/embedding/final_norm stay full precision either way.
        self.qkv = _make_linear(d, 3 * d, use_bitlinear)
        self.attn_out = _make_linear(d, d, use_bitlinear)
        self.gate = _make_linear(d, ff, use_bitlinear)
        self.up = _make_linear(d, ff, use_bitlinear)
        self.down = _make_linear(ff, d, use_bitlinear)
        self.heads, self.head_dim = config.num_heads, config.head_dim
        self.use_rope = getattr(config, "use_rope", False)

    def forward(self, x: torch.Tensor, activation_sparsity_out: list | None = None) -> torch.Tensor:
        bsz, steps, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(bsz, steps, self.heads, 3 * self.head_dim).chunk(3, dim=-1)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        if self.use_rope:
            cos, sin = _rope_cos_sin(steps, self.head_dim, x.device, q.dtype)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        mixed = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_out(mixed.transpose(1, 2).reshape(bsz, steps, dim))
        y = self.norm2(x)
        gated = torch.nn.functional.silu(self.gate(y))
        if activation_sparsity_out is not None:
            # SwiGLU's silu is never exactly zero (unlike BDH's ReLU, which
            # produces true sparse-positive activations) -- "near-zero"
            # (|.| < 0.05) is a real, disclosed, DIFFERENT metric standing
            # in for the same comparison slot, not a directly-comparable
            # number to BDH's exact-zero fraction. See
            # reference/hz0h_bdh_torch.py's compute_activation_and_state_diagnostics
            # for the BDH side.
            activation_sparsity_out.append(float((gated.abs() < 0.05).float().mean()))
        return x + self.down(gated * self.up(y))


class MatchedTransformerLM(nn.Module):
    def __init__(self, config: MatchedTransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Same fix as reference/hz0a_torch_model.py's HZ0AModel (found 2026-08-01,
        # see that file's comment for the full derivation): PyTorch's nn.Embedding
        # default init is N(0, 1), ~28x larger than MLX's own `sqrt(1/dims)`
        # default -- left uncorrected here, it produces the same ~500+ initial
        # loss / 300-1000+ early gradient norms as the HZ0AModel bug did, not a
        # depth-scaling quirk of this architecture as originally assumed.
        nn.init.normal_(self.embedding.weight, std=math.sqrt(1.0 / config.d_model))
        self.blocks = nn.ModuleList(MatchedTransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = BiasFreeRMSNorm(config.d_model)

    def forward(self, token_ids: torch.Tensor, activation_sparsity_out: list | None = None) -> torch.Tensor:
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, activation_sparsity_out=activation_sparsity_out)
        return torch.einsum("btd,vd->btv", self.final_norm(x), self.embedding.weight)


def parameter_count(config: MatchedTransformerConfig) -> int:
    return sum(parameter.numel() for parameter in MatchedTransformerLM(config).parameters())


def compute_activation_diagnostics(model: "MatchedTransformerLM", token_ids: torch.Tensor) -> dict:
    """Phase 1 metric (plans/HatchlingZero_Reality_Plan.md): per-layer
    SwiGLU-gate near-zero fraction, read-only (no_grad, restores the
    caller's train/eval mode). NOT directly comparable to BDH's exact-zero
    ReLU sparsity (reference/hz0h_bdh_torch.py's
    compute_activation_and_state_diagnostics) -- a real architecture
    difference (dense SwiGLU vs. true sparse-positive ReLU activations),
    not a bug. This model has no persistent/recurrent state, so there is
    no analogous state_norms metric to report here."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        activation_sparsity: list = []
        model(token_ids, activation_sparsity_out=activation_sparsity)
    if was_training:
        model.train()
    return {
        "gate_near_zero_fraction": activation_sparsity,
        "mean_activation_sparsity": sum(activation_sparsity) / len(activation_sparsity),
        "state_norms": None,
    }
