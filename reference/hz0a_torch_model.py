"""Config-driven PyTorch HZ-0A reference model with recurrent state carry."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class HZ0AConfig:
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_k: int
    d_v: int
    d_ff: int
    attention_layer_indices: tuple[int, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "HZ0AConfig":
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(spec["vocab_size"], spec["d_model"], spec["num_layers"], spec["num_heads"], spec["head_dim_qk"], spec["head_dim_v"], spec["d_ff"], tuple(spec["attention_layer_indices"]))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d_model, d_ff), nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class GDN2Mixer(nn.Module):
    def __init__(self, c: HZ0AConfig):
        super().__init__()
        self.c = c
        width = c.num_heads * (4 * c.d_k + 2 * c.d_v)
        self.in_proj, self.out_proj = nn.Linear(c.d_model, width), nn.Linear(c.num_heads * c.d_v, c.d_model)
        start = c.num_heads * (2 * c.d_k + c.d_v)
        self.in_proj.bias.data[start:start + c.num_heads * c.d_k].fill_(4.59512)
        self.in_proj.bias.data[start + c.num_heads * c.d_k:start + 2 * c.num_heads * c.d_k].fill_(-4.59512)
        self.in_proj.bias.data[start + 2 * c.num_heads * c.d_k:].fill_(-4.59512)

    def forward(self, x, state):
        c, bsz, steps = self.c, x.shape[0], x.shape[1]
        p = self.in_proj(x).view(bsz, steps, c.num_heads, 4 * c.d_k + 2 * c.d_v)
        q, k, v = p[..., :c.d_k], p[..., c.d_k:2*c.d_k], p[..., 2*c.d_k:2*c.d_k+c.d_v]
        offset = 2 * c.d_k + c.d_v
        decay, erase, write = torch.sigmoid(p[..., offset:offset+c.d_k]), torch.sigmoid(p[..., offset+c.d_k:offset+2*c.d_k]), torch.sigmoid(p[..., offset+2*c.d_k:])
        outputs = []
        for t in range(steps):
            state = decay[:, t, :, None, :] * (1 - erase[:, t, :, None, :]) * state + write[:, t, :, :, None] * v[:, t, :, :, None] * k[:, t, :, None, :]
            outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, t]))
        return self.out_proj(torch.stack(outputs, dim=1).reshape(bsz, steps, c.num_heads * c.d_v)), state


class CausalAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.qkv, self.out = nn.Linear(c.d_model, 3 * c.num_heads * c.d_k), nn.Linear(c.num_heads * c.d_k, c.d_model)

    def forward(self, x):
        c, bsz, steps = self.c, x.shape[0], x.shape[1]
        q, k, v = self.qkv(x).view(bsz, steps, c.num_heads, 3 * c.d_k).chunk(3, dim=-1)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(bsz, steps, c.num_heads * c.d_k))


class HZ0ABlock(nn.Module):
    def __init__(self, c, attention):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.mixer = CausalAttention(c) if attention else GDN2Mixer(c)
        self.mlp, self.attention = SwiGLU(c.d_model, c.d_ff), attention

    def forward(self, x, state):
        mixed = self.mixer(self.norm1(x)) if self.attention else self.mixer(self.norm1(x), state)
        if self.attention:
            mixed, next_state = mixed, None
        else:
            mixed, next_state = mixed
        x = x + mixed
        return x + self.mlp(self.norm2(x)), next_state


class HZ0AModel(nn.Module):
    def __init__(self, c: HZ0AConfig):
        super().__init__()
        self.config, self.embedding = c, nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(HZ0ABlock(c, i in c.attention_layer_indices) for i in range(c.num_layers))
        self.final_norm = RMSNorm(c.d_model)

    def init_states(self, batch_size, device=None, dtype=None):
        c = self.config
        return [None if block.attention else torch.zeros(batch_size, c.num_heads, c.d_v, c.d_k, device=device, dtype=dtype or self.embedding.weight.dtype) for block in self.blocks]

    def forward(self, token_ids, states=None):
        x = self.embedding(token_ids)
        states = self.init_states(token_ids.shape[0], token_ids.device) if states is None else states
        next_states = []
        for block, state in zip(self.blocks, states):
            x, state = block(x, state)
            next_states.append(state)
        return torch.einsum("btd,vd->btv", self.final_norm(x), self.embedding.weight), next_states


def parameter_count(config: HZ0AConfig) -> int:
    return sum(parameter.numel() for parameter in HZ0AModel(config).parameters())
