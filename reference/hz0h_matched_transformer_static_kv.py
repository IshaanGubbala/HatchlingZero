"""Production-style static-KV-cache decode path for MatchedTransformerLM.

Tier 0 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md:
the existing measure_transformer_decode_kv_cache
(scripts/hz0h_inference_benchmark.py) is real and correct, but its cache
grows via `torch.cat([kv_cache["k"], k], dim=2)` every single decode step
-- reallocating and copying the ENTIRE past K/V tensor on every new token,
not the "no per-token torch.cat" production pattern real serving stacks
use. That's real, avoidable per-step overhead that scales with context
length, exactly the kind of asymmetry that made the earlier BDH-vs-
Transformer long-context decode crossover provisional (BDH's decode path
got far more engineering attention than the Transformer's).

This module preallocates a fixed-size (B, heads, max_seq_len, head_dim)
buffer per layer once, and each forward call writes only the NEW tokens
into their slice via an in-place assignment -- no reallocation, no copy
of prior tokens. Attention reads a prefix VIEW of the buffer (cheap,
no copy). Reuses MatchedTransformerConfig/BiasFreeRMSNorm/RoPE helpers
from reference/hz0a_matched_transformer.py verbatim -- this is a new
cache mechanism, not a new architecture, so the actual math must stay
identical (verified bit-exact against the cat-based path in
tests/reference/test_matched_transformer_static_kv.py-style checks
before this file's benchmark numbers are trusted).
"""
from __future__ import annotations

import torch
from torch import nn

from reference.hz0a_matched_transformer import (
    BiasFreeRMSNorm,
    MatchedTransformerConfig,
    _apply_rope,
    _rope_cos_sin,
)


class StaticKVCache:
    """One fixed-size (B, heads, max_seq_len, head_dim) k/v buffer pair
    per layer, preallocated once. `length` tracks how many positions are
    currently valid (Python int, not a tensor -- host-side bookkeeping
    only, same as the cat-based cache's dict-presence check)."""

    def __init__(self, num_layers: int, batch_size: int, num_heads: int, max_seq_len: int, head_dim: int, device, dtype):
        self.length = 0
        self.max_seq_len = max_seq_len
        self.k = [torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype) for _ in range(num_layers)]
        self.v = [torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype) for _ in range(num_layers)]

    def write(self, layer_index: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Writes k_new/v_new (B, heads, steps, head_dim) into this
        layer's buffer at [length : length+steps] via in-place slice
        assignment, returns (k_view, v_view, past_length) where the views
        cover [0 : length+steps] -- the valid prefix for attention. Only
        layer 0's call should advance self.length (all layers see the
        same number of new tokens per forward pass); callers advance
        length once per model-level forward via `advance`."""
        steps = k_new.shape[2]
        past_length = self.length
        end = past_length + steps
        if end > self.max_seq_len:
            raise ValueError(f"StaticKVCache overflow: tried to write through position {end}, max_seq_len={self.max_seq_len}")
        self.k[layer_index][:, :, past_length:end, :] = k_new
        self.v[layer_index][:, :, past_length:end, :] = v_new
        return self.k[layer_index][:, :, :end, :], self.v[layer_index][:, :, :end, :], past_length

    def advance(self, steps: int) -> None:
        self.length += steps


def _static_kv_attention_mask(past_length: int, new_length: int, device) -> torch.Tensor:
    """Identical semantics to hz0a_matched_transformer._kv_cache_attention_mask
    (True = attend); duplicated here rather than imported because that
    function is private to a module documented as backing many historical
    results that should not casually change behavior."""
    key_positions = torch.arange(past_length + new_length, device=device).view(1, -1)
    query_positions = torch.arange(new_length, device=device).view(-1, 1) + past_length
    return key_positions <= query_positions


class StaticKVMatchedTransformerBlock(nn.Module):
    def __init__(self, config: MatchedTransformerConfig, block: nn.Module):
        """Wraps an EXISTING MatchedTransformerBlock's parameters (qkv,
        attn_out, gate, up, down, norm1, norm2) rather than reinitializing
        them -- lets a benchmark load one set of weights and run both the
        cat-based and static-buffer decode paths for a bit-exactness
        check, and keeps this file from silently drifting out of sync
        with MatchedTransformerBlock's own linear/norm construction
        (use_bitlinear, etc)."""
        super().__init__()
        self.norm1, self.norm2 = block.norm1, block.norm2
        self.qkv, self.attn_out = block.qkv, block.attn_out
        self.gate, self.up, self.down = block.gate, block.up, block.down
        self.heads, self.head_dim = config.num_heads, config.head_dim
        self.use_rope = getattr(config, "use_rope", False)

    def forward(self, x: torch.Tensor, cache: StaticKVCache, layer_index: int) -> torch.Tensor:
        bsz, steps, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(bsz, steps, self.heads, 3 * self.head_dim).chunk(3, dim=-1)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))

        past_length = cache.length
        if self.use_rope:
            cos, sin = _rope_cos_sin(steps, self.head_dim, x.device, q.dtype, start_position=past_length)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        k_view, v_view, past_length = cache.write(layer_index, k, v)
        if past_length == 0:
            # Full prefill (nothing cached yet, query length == key length):
            # mathematically identical to the explicit mask below, but
            # is_causal=True lets SDPA dispatch to the memory-efficient
            # flash-attention kernel instead of materializing an explicit
            # (steps, steps) boolean mask -- at context=65536 the explicit
            # path OOM'd (tried to allocate 8 GiB for one op) purely from
            # forcing the non-fused backend, not from genuine VRAM pressure.
            mixed = torch.nn.functional.scaled_dot_product_attention(q, k_view, v_view, is_causal=True)
        else:
            mask = _static_kv_attention_mask(past_length, steps, x.device)
            mixed = torch.nn.functional.scaled_dot_product_attention(q, k_view, v_view, attn_mask=mask)
        x = x + self.attn_out(mixed.transpose(1, 2).reshape(bsz, steps, dim))
        y = self.norm2(x)
        gated = torch.nn.functional.silu(self.gate(y))
        return x + self.down(gated * self.up(y))


class StaticKVMatchedTransformerLM(nn.Module):
    """Wraps an existing, already-constructed MatchedTransformerLM --
    same weights, same embedding, same final_norm -- swapping only the
    attention cache mechanism. `from_matched` is the intended
    constructor; building this directly with a bare config would
    duplicate initialization and defeat the point of testing the SAME
    weights under two cache strategies."""

    def __init__(self, matched_model: nn.Module):
        super().__init__()
        self.config = matched_model.config
        self.embedding = matched_model.embedding
        self.final_norm = matched_model.final_norm
        self.blocks = nn.ModuleList(
            StaticKVMatchedTransformerBlock(matched_model.config, block) for block in matched_model.blocks
        )

    def new_cache(self, batch_size: int, max_seq_len: int, device, dtype) -> StaticKVCache:
        return StaticKVCache(
            num_layers=len(self.blocks), batch_size=batch_size, num_heads=self.config.num_heads,
            max_seq_len=max_seq_len, head_dim=self.config.head_dim, device=device, dtype=dtype,
        )

    def forward(self, token_ids: torch.Tensor, cache: StaticKVCache) -> torch.Tensor:
        x = self.embedding(token_ids)
        for layer_index, block in enumerate(self.blocks):
            x = block(x, cache, layer_index)
        cache.advance(token_ids.shape[1])
        return torch.einsum("btd,vd->btv", self.final_norm(x), self.embedding.weight)
