"""HZ-CQ-v1 Mainline Phase 1, STEP 1: real fixed-size persistent task
memory S, plans/HatchlingZero — Mainline Research Plan.md section 6.1.

    S_t = U_theta(S_{t-1}, E(D_t)),   S in R^{M_S x D}, M_S ~= 4-16

Deliberately boring per section 6.4 -- no LoRA, no halting, no
verification, nothing beyond S itself. Real design choices, each
justified against the plan's own locked KEEP/DEAD lessons (section 2/3):

- D (memory width) is kept EQUAL to the base model's n_embd, never a
  separate low-dimensional bottleneck -- directly enforces "do not
  force BDH through a new low-dimensional latent coordinate system"
  (DEAD: compressed BDH-Delta belief/workspace).
- The cross-attention Q/K/V projections are full-rank dense Linears,
  no low-rank factoring -- "KEEP: exact/high-fidelity addressing,
  do not approximate the addressing side without overwhelming evidence".
- The write path (the value side of this read/write) is where
  compression would be tolerated per "KEEP: value/output-side
  compression", but is left dense here too since M_S is already tiny
  (4-16 slots) -- there is no parameter-count problem to solve on this
  path, so there is nothing to compress.
- The gated write reuses the EXACT validated adaptive-gate design from
  reference/hz0h_bdh_adaptive_gate_torch.py (same q-feature shape, same
  protected zero-init W2 + logit(g_init) b2, same tiny 2-layer MLP) --
  "controlled state-dependent writes improve and stabilize recurrence"
  is the strongest validated recurrent-dynamics result this project
  has, and section 6.3 explicitly says to reuse it rather than invent
  a new controller.

This module takes demo hidden states as a plain tensor (already
encoded elsewhere, e.g. by running demo bytes through the shared BDH
round mechanism) rather than raw bytes -- keeps S itself cheap to unit
test in isolation (Phase 1A, section 7: "Before spending meaningful
GPU money, v1 must pass cheap correctness tests") without needing a
full model forward pass per test case. A later integration module
wires a real byte encoder in front of this.

Real, load-bearing property this design is built to satisfy (section
7's "Query-time memory cost does not grow with raw demo length"): after
`update()` returns, nothing about the input hidden states is retained
by S itself -- S's own memory footprint is exactly `M_S x D` regardless
of how long or how many demos were fed in one at a time. The CALLER
must not retain the raw demo sequence either (this module does not
enforce that on its own -- it is a real integration responsibility,
tested in Phase 1A's test 4).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-5


class HZCQPersistentMemoryConfig:
    def __init__(self, n_embd: int, memory_slots: int = 8, gate_hidden: int = 16, g_init: float = 0.58):
        if not (4 <= memory_slots <= 16):
            raise ValueError(f"memory_slots should be in the plan's stated 4-16 range, got {memory_slots}")
        self.n_embd = n_embd
        self.memory_slots = memory_slots
        self.gate_hidden = gate_hidden
        self.g_init = g_init


def _rms(t: torch.Tensor) -> torch.Tensor:
    """eps INSIDE the sqrt -- same real fix as the validated adaptive
    gate (h - h_prev, here S_prev - delta_S's own zero-init case, can be
    exactly zero at the first ever call)."""
    return (t.pow(2).mean(dim=-1, keepdim=True) + _EPS).sqrt()


class HZCQPersistentMemory(nn.Module):
    """Fixed-size persistent task memory S, updated by sequential
    demonstration ingestion via exact cross-attention + a validated
    state-dependent gate. S never grows -- every call to `update`
    returns a tensor of the same (B, memory_slots, n_embd) shape."""

    def __init__(self, config: HZCQPersistentMemoryConfig):
        super().__init__()
        self.config = config
        D = config.n_embd
        M_S = config.memory_slots

        # Real init: small random S_0, not exactly zero -- an all-zero
        # S_0 would make the very first cross-attention read's queries
        # identical across all M_S slots (Q = 0 @ q_proj = 0 regardless
        # of q_proj), collapsing every slot to the same read before any
        # training signal could differentiate them.
        self.S_init = nn.Parameter(torch.randn(M_S, D) * 0.02)

        # Exact (dense, full-rank) cross-attention -- addressing side,
        # per the KEEP rule above.
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.write_proj = nn.Linear(D, D, bias=False)
        self.ln_read = nn.LayerNorm(D)
        self.ln_state = nn.LayerNorm(D)

        # Gate controller: EXACT same shape/init discipline as
        # reference/hz0h_bdh_adaptive_gate_torch.py's add_adaptive_gate.
        # q-features here: RMS(S_prev), RMS(delta_S), cos(S_prev,
        # delta_S), RMS(S_prev averaged - delta_S averaged) -- the
        # closest faithful adaptation of that file's [RMS(h), RMS(y),
        # cos(h,y), RMS(h-h_prev), d_r] to a memory-write context (no
        # separate "evidence" tensor here the way BDH's e_r is; the
        # cross-attention read itself IS the evidence, so d_r's slot is
        # dropped rather than faked -- q_dim=4, not 5).
        q_dim = 4
        self.gate_w1 = nn.Parameter(torch.zeros(q_dim, config.gate_hidden).normal_(std=0.02))
        self.gate_b1 = nn.Parameter(torch.zeros(config.gate_hidden))
        self.gate_w2 = nn.Parameter(torch.zeros(config.gate_hidden, 1))  # protected zero init
        logit = math.log(config.g_init / (1.0 - config.g_init))
        self.gate_b2 = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        S0 = self.S_init.to(device=device or self.S_init.device, dtype=dtype or self.S_init.dtype)
        return S0.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def _gate(self, S_prev: torch.Tensor, delta_S: torch.Tensor) -> torch.Tensor:
        q = torch.cat([
            _rms(S_prev), _rms(delta_S),
            F.cosine_similarity(S_prev, delta_S, dim=-1).unsqueeze(-1),
            _rms(S_prev - delta_S),
        ], dim=-1)
        hid = F.silu(q @ self.gate_w1 + self.gate_b1)
        g_logit = hid @ self.gate_w2 + self.gate_b2
        return torch.sigmoid(g_logit)

    def update(self, S_prev: torch.Tensor, demo_hidden: torch.Tensor,
               demo_mask: torch.Tensor | None = None) -> torch.Tensor:
        """S_prev: (B, M_S, D). demo_hidden: (B, T_demo, D), already
        encoded elsewhere -- discarded by the caller after this returns,
        never retained by S. demo_mask: optional (B, T_demo) bool,
        True = real token (for batched demos of different real
        lengths padded to a common T_demo)."""
        Q = self.q_proj(S_prev)
        K = self.k_proj(demo_hidden)
        V = self.v_proj(demo_hidden)
        scale = 1.0 / math.sqrt(Q.size(-1))
        scores = torch.matmul(Q, K.transpose(-1, -2)) * scale  # (B, M_S, T_demo)
        if demo_mask is not None:
            scores = scores.masked_fill(~demo_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        read = torch.matmul(attn, V)  # (B, M_S, D)
        delta_S = self.ln_read(self.write_proj(read))
        g = self._gate(S_prev, delta_S)
        S_new = self.ln_state(S_prev + g * delta_S)
        return S_new

    def update_sequence(self, batch_size: int, demo_hiddens: list[torch.Tensor],
                         demo_masks: list[torch.Tensor | None] | None = None,
                         device=None, dtype=None) -> torch.Tensor:
        """Convenience: ingest a REAL SEQUENCE of demonstrations one at a
        time (order matters -- each update conditions on the previous
        S, not on the raw history), returning only the final S. This is
        the actual real-usage pattern: `data D_1..D_k` seen one at a
        time, never concatenated into one growing sequence -- directly
        the property that makes HZ-CQ-v0 NOT faithful (plan section
        4)."""
        S = self.init_state(batch_size, device=device, dtype=dtype)
        masks = demo_masks or [None] * len(demo_hiddens)
        for demo_hidden, mask in zip(demo_hiddens, masks):
            S = self.update(S, demo_hidden, mask)
        return S
