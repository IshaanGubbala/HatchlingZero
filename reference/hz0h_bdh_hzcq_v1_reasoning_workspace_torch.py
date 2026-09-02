"""HZ-CQ-v1 Mainline Phase 1, STEP 3/4: real fixed-size recurrent
reasoning workspace H, plans/HatchlingZero — Mainline Research Plan.md
section 6.2/6.3.

    H_r in R^{M_H x D},   M_H = 4 or 8
    Delta H_r = F_theta(H_r, S, x)
    g_r = C_phi(H_r, Delta H_r, S)
    H_{r+1} = LN(H_r + g_r * Delta H_r)

The SAME M_H slots evolve every round -- no new sequence positions are
ever created. This is the direct fix for the plan's diagnosed HZ-CQ-v0
Problem 1 (section 4): "reasoning is actually sequence growth". Here,
R applications of `step` always return a (B, M_H, D) tensor -- R=16
means sixteen real applications of one operator, never sixteen new
token positions, so R is finally interpretable as latent compute
depth rather than a confound with sequence length/positional geometry.

Real design choices:

- H reads from TWO sources each round: the persistent task memory S
  (exact cross-attention, same discipline as
  reference/hz0h_bdh_hzcq_v1_persistent_memory_torch.py) and the query's
  own encoded hidden states x (same exact cross-attention pattern,
  separate weights -- S answers "what's the rule", x answers "what am
  I looking at right now", and H's job is to combine both).
- Both cross-attention pathways are full-rank dense (KEEP: exact/high-
  fidelity addressing).
- The gated residual write reuses the exact validated adaptive-gate
  design (KEEP: controlled state-dependent writes improve and
  stabilize recurrence) -- same q-feature shape (adapted for an H/S
  context instead of h/y/e), same protected zero-init W2 + logit(g_init)
  b2.
- Weights are TIED across every round (KEEP: weight tying) -- `step` is
  called repeatedly with the SAME parameters, never a per-round copy.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-5


class HZCQReasoningWorkspaceConfig:
    def __init__(self, n_embd: int, workspace_slots: int = 8, gate_hidden: int = 16, g_init: float = 0.58):
        if workspace_slots not in (4, 8):
            raise ValueError(f"plan section 6.2 specifies M_H = 4 or 8, got {workspace_slots}")
        self.n_embd = n_embd
        self.workspace_slots = workspace_slots
        self.gate_hidden = gate_hidden
        self.g_init = g_init


def _rms(t: torch.Tensor) -> torch.Tensor:
    return (t.pow(2).mean(dim=-1, keepdim=True) + _EPS).sqrt()


class _ExactCrossAttention(nn.Module):
    """One dense, full-rank cross-attention read: queries from H,
    keys/values from some source tensor. Factored out since H needs
    two of these (one for S, one for x) with independent weights."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.scale = 1.0 / math.sqrt(n_embd)

    def forward(self, H: torch.Tensor, source: torch.Tensor, source_mask: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.q_proj(H)
        K = self.k_proj(source)
        V = self.v_proj(source)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        if source_mask is not None:
            scores = scores.masked_fill(~source_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)


class HZCQReasoningWorkspace(nn.Module):
    """Fixed-size recurrent reasoning workspace H. `step` is the tied
    per-round operator F_theta; `run` applies it R times, returning
    only the final H (shape never changes across R)."""

    def __init__(self, config: HZCQReasoningWorkspaceConfig):
        super().__init__()
        self.config = config
        D = config.n_embd
        M_H = config.workspace_slots

        self.H_init = nn.Parameter(torch.randn(M_H, D) * 0.02)
        self.read_s = _ExactCrossAttention(D)
        self.read_x = _ExactCrossAttention(D)
        self.write_proj = nn.Linear(2 * D, D, bias=False)
        self.ln_read = nn.LayerNorm(D)
        self.ln_state = nn.LayerNorm(D)

        # Gate controller: q-features adapted from the validated
        # adaptive gate's [RMS(h), RMS(y), cos(h,y), RMS(h-h_prev), d_r]
        # to this context -- d_r (evidence-disagreement) here is a real,
        # direct analogue: cos(H_prev, S-summary), matching the
        # original's cos(h_r, e_r) "evidence disagreement" role exactly
        # (S is this module's evidence, the same way e_r was BDH's).
        q_dim = 5
        self.gate_w1 = nn.Parameter(torch.zeros(q_dim, config.gate_hidden).normal_(std=0.02))
        self.gate_b1 = nn.Parameter(torch.zeros(config.gate_hidden))
        self.gate_w2 = nn.Parameter(torch.zeros(config.gate_hidden, 1))  # protected zero init
        logit = math.log(config.g_init / (1.0 - config.g_init))
        self.gate_b2 = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        H0 = self.H_init.to(device=device or self.H_init.device, dtype=dtype or self.H_init.dtype)
        return H0.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def _gate(self, H_prev: torch.Tensor, delta_H: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        s_summary = S.mean(dim=1, keepdim=True)  # (B, 1, D), broadcasts against (B, M_H, D)
        q = torch.cat([
            _rms(H_prev), _rms(delta_H),
            F.cosine_similarity(H_prev, delta_H, dim=-1).unsqueeze(-1),
            _rms(H_prev - delta_H),
            F.cosine_similarity(H_prev, s_summary.expand_as(H_prev), dim=-1).unsqueeze(-1),
        ], dim=-1)
        hid = F.silu(q @ self.gate_w1 + self.gate_b1)
        g_logit = hid @ self.gate_w2 + self.gate_b2
        return torch.sigmoid(g_logit)

    def step(self, H_prev: torch.Tensor, S: torch.Tensor, x_hidden: torch.Tensor,
              x_mask: torch.Tensor | None = None) -> torch.Tensor:
        """One real application of the tied reasoning operator F_theta.
        H_prev, S: (B, M_*, D). x_hidden: (B, T_query, D). Returns
        (B, M_H, D) -- exactly H_prev's shape, always."""
        read_from_s = self.read_s(H_prev, S)
        read_from_x = self.read_x(H_prev, x_hidden, x_mask)
        delta_H = self.ln_read(self.write_proj(torch.cat([read_from_s, read_from_x], dim=-1)))
        g = self._gate(H_prev, delta_H, S)
        H_new = self.ln_state(H_prev + g * delta_H)
        return H_new

    def run(self, batch_size: int, S: torch.Tensor, x_hidden: torch.Tensor, n_rounds: int,
            x_mask: torch.Tensor | None = None, device=None, dtype=None) -> torch.Tensor:
        """Apply the SAME tied operator n_rounds times. Real, direct
        test of section 6.2's core claim -- call this with different
        n_rounds and verify (Phase 1A test 1-4) that only the CONTENT
        of the returned H changes, never its shape, and that no new
        sequence positions or growing tensors appear anywhere in the
        process."""
        H = self.init_state(batch_size, device=device, dtype=dtype)
        for _ in range(n_rounds):
            H = self.step(H, S, x_hidden, x_mask)
        return H
