r"""HZ-CQ-v1 PAPER-4 -- fast scratch / slow integrator, plan's
"Paper-Derived Reasoning Upgrades" queue, fourth architecture ablation.
Motivated by real, distinct failures of the first three single-state
residual variants (PAPER-2: unbounded accumulation -> gate hard-
collapse, -4.98pp; PAPER-3: bounded but fixed-anchor -> no real
accumulation, -4.89pp; PAPER-3b: bounded AND accumulating -> WORSE
than either, -7.14pp, real evidence that per-round boundedness alone
can't keep the gate open once genuine accumulation is allowed). All
three shared one property: a single state \(H_r\) has to be both the
volatile per-round scratchpad AND the thing that accumulates useful
information across rounds. PAPER-4 tests whether separating those two
roles into two distinct full-dimensional states helps.

    H_fast in R^{M_H x D}   -- temporary, fully overwritten every round
    H_slow in R^{M_H x D}   -- persistent, gated-residual-accumulated

    H_fast_{r+1} = LN(write_fast(read_s(H_fast_r,S), read_x(H_fast_r,x),
                                  read_slow(H_fast_r,H_slow_r)))
    H_slow_{r+1} = H_slow_r + alpha * g_r * write_slow(read_fast(H_slow_r,H_fast_{r+1}))

H_fast is fully renormalized (LN'd) every round -- same discipline as
the ORIGINAL default single-state design, deliberately, since that's
the one variant of the four tested so far that never showed gate
collapse. H_slow gets the same gated-residual write as the original
design's H_r, but its ONLY input is a read against H_fast (not S/x
directly) -- the hypothesis being that keeping H_slow's own update
small and mediated through a renormalized scratch state, rather than
letting it read raw evidence and drift unboundedly like PAPER-2's H_r
did, avoids the compounding-drift problem all three residual variants
ran into.

No compressed low-dimensional belief state (both states stay full D).
Weight tying preserved: the same modules are reused every round, never
copied. `run()` returns H_slow only (shape (B, M_H, D)) so the FSM
harness's existing readout code needs zero changes."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-5


def _rms(t: torch.Tensor) -> torch.Tensor:
    return (t.pow(2).mean(dim=-1, keepdim=True) + _EPS).sqrt()


class _ExactCrossAttention(nn.Module):
    """Same discipline as the single-state module: dense, full-rank,
    exact addressing -- KEEP per section 2."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.scale = 1.0 / math.sqrt(n_embd)

    def forward(self, Q_src: torch.Tensor, KV_src: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.q_proj(Q_src)
        K = self.k_proj(KV_src)
        V = self.v_proj(KV_src)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)


class HZCQFastSlowConfig:
    def __init__(self, n_embd: int, workspace_slots: int = 32, gate_hidden: int = 16,
                 g_init: float = 0.58, alpha_init: float = 0.1, allow_ablation_slots: bool = True):
        allowed = (4, 8, 16, 32, 64) if allow_ablation_slots else (4, 8)
        if workspace_slots not in allowed:
            raise ValueError(f"workspace_slots must be one of {allowed}, got {workspace_slots}")
        self.n_embd = n_embd
        self.workspace_slots = workspace_slots
        self.gate_hidden = gate_hidden
        self.g_init = g_init
        self.alpha_init = alpha_init


class HZCQReasoningWorkspaceFastSlow(nn.Module):
    """Two full-dimensional state roles. `step` is the tied per-round
    operator; `run` applies it R times and returns H_slow only."""

    def __init__(self, config: HZCQFastSlowConfig):
        super().__init__()
        self.config = config
        D = config.n_embd
        M_H = config.workspace_slots

        self.H_fast_init = nn.Parameter(torch.randn(M_H, D) * 0.02)
        self.H_slow_init = nn.Parameter(torch.randn(M_H, D) * 0.02)

        # F_fast reads THREE sources: S, x, and the current H_slow.
        self.fast_read_s = _ExactCrossAttention(D)
        self.fast_read_x = _ExactCrossAttention(D)
        self.fast_read_slow = _ExactCrossAttention(D)
        self.fast_write = nn.Linear(3 * D, D, bias=False)
        self.fast_ln = nn.LayerNorm(D)

        # F_slow reads only H_fast (the just-updated scratch).
        self.slow_read_fast = _ExactCrossAttention(D)
        self.slow_write = nn.Linear(D, D, bias=False)
        self.slow_ln_read = nn.LayerNorm(D)

        # Same validated adaptive-gate design, gating the SLOW write only.
        q_dim = 5
        self.gate_w1 = nn.Parameter(torch.zeros(q_dim, config.gate_hidden).normal_(std=0.02))
        self.gate_b1 = nn.Parameter(torch.zeros(config.gate_hidden))
        self.gate_w2 = nn.Parameter(torch.zeros(config.gate_hidden, 1))  # protected zero init
        logit = math.log(config.g_init / (1.0 - config.g_init))
        self.gate_b2 = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

        self.alpha = nn.Parameter(torch.tensor(config.alpha_init, dtype=torch.float32))

    def init_state(self, batch_size: int, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
        Hf0 = self.H_fast_init.to(device=device or self.H_fast_init.device, dtype=dtype or self.H_fast_init.dtype)
        Hs0 = self.H_slow_init.to(device=device or self.H_slow_init.device, dtype=dtype or self.H_slow_init.dtype)
        return (Hf0.unsqueeze(0).expand(batch_size, -1, -1).clone(),
                Hs0.unsqueeze(0).expand(batch_size, -1, -1).clone())

    def _gate(self, H_slow_prev: torch.Tensor, delta_slow: torch.Tensor, s_summary: torch.Tensor) -> torch.Tensor:
        q = torch.cat([
            _rms(H_slow_prev), _rms(delta_slow),
            F.cosine_similarity(H_slow_prev, delta_slow, dim=-1).unsqueeze(-1),
            _rms(H_slow_prev - delta_slow),
            F.cosine_similarity(H_slow_prev, s_summary.expand_as(H_slow_prev), dim=-1).unsqueeze(-1),
        ], dim=-1)
        hid = F.silu(q @ self.gate_w1 + self.gate_b1)
        g_logit = hid @ self.gate_w2 + self.gate_b2
        return torch.sigmoid(g_logit)

    def step(self, H_fast_prev: torch.Tensor, H_slow_prev: torch.Tensor, S: torch.Tensor,
              x_hidden: torch.Tensor, x_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        rs = self.fast_read_s(H_fast_prev, S)
        rx = self.fast_read_x(H_fast_prev, x_hidden, x_mask)
        r_slow = self.fast_read_slow(H_fast_prev, H_slow_prev)
        H_fast_new = self.fast_ln(self.fast_write(torch.cat([rs, rx, r_slow], dim=-1)))

        r_fast = self.slow_read_fast(H_slow_prev, H_fast_new)
        delta_slow = self.slow_ln_read(self.slow_write(r_fast))
        s_summary = S.mean(dim=1, keepdim=True)
        g = self._gate(H_slow_prev, delta_slow, s_summary)
        H_slow_new = H_slow_prev + self.alpha * g * delta_slow

        return H_fast_new, H_slow_new

    def run(self, batch_size: int, S: torch.Tensor, x_hidden: torch.Tensor, n_rounds: int,
            x_mask: torch.Tensor | None = None, device=None, dtype=None) -> torch.Tensor:
        """Applies the tied fast/slow operator n_rounds times, returns
        H_slow only -- (B, M_H, D), same contract as the single-state
        HZCQReasoningWorkspace.run(), so the FSM harness's readout code
        needs zero changes to consume this."""
        H_fast, H_slow = self.init_state(batch_size, device=device, dtype=dtype)
        for _ in range(n_rounds):
            H_fast, H_slow = self.step(H_fast, H_slow, S, x_hidden, x_mask)
        return H_slow
