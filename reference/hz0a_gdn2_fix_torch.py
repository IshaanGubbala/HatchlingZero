"""Torch oracle for the exact vector-gated HZ-0A GDN-2 fix."""

from __future__ import annotations

import torch
from torch import nn


def gdn2_fix_step(state, query, key, value, alpha, erase, write, *, normalize_key=True):
    if normalize_key:
        key = key / key.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    decayed = state * alpha[:, :, None, :]
    old_value = (decayed * (erase * key)[:, :, None, :]).sum(dim=-1)
    next_state = decayed + (write * value - old_value)[:, :, :, None] * key[:, :, None, :]
    return (next_state * query[:, :, None, :]).sum(dim=-1), next_state


def gdn2_fix_scan(query, key, value, alpha, erase, write, initial_state=None, *, normalize_key=True):
    batch, steps, heads, d_k = query.shape
    d_v = value.shape[-1]
    state = torch.zeros(batch, heads, d_v, d_k, dtype=value.dtype, device=value.device) if initial_state is None else initial_state
    outputs = []
    for t in range(steps):
        output, state = gdn2_fix_step(state, query[:, t], key[:, t], value[:, t], alpha[:, t], erase[:, t], write[:, t], normalize_key=normalize_key)
        outputs.append(output)
    return torch.stack(outputs, dim=1), state


def _gdn2_fix_step_prenormalized(state, query, key, value, alpha, erase, write):
    """Same math as `gdn2_fix_step`, but assumes `key` (and `query`, though
    unused here) are ALREADY normalized -- normalization is hoisted out of
    the loop in `_gdn2_fix_sequential` below (it doesn't depend on `state`,
    so it's wasteful to redo it every single step)."""
    decayed = state * alpha[:, :, None, :]
    old_value = (decayed * (erase * key)[:, :, None, :]).sum(dim=-1)
    next_state = decayed + (write * value - old_value)[:, :, :, None] * key[:, :, None, :]
    return next_state, (next_state * query[:, :, None, :]).sum(dim=-1)


def _gdn2_fix_sequential(state, query, key, value, alpha, erase, write, *, normalize_key=True):
    """Whole-chunk-loop version of `gdn2_fix_scan`, restructured to match
    `reference/hz0a_torch_model.py`'s `_gdn2_sequential` /
    `reference/hz0a_gdn3_candidate_mixer_torch.py`'s `_gdn3_sequential`
    convention: `(state, decay/gate..., ...) -> (state, stacked_outputs)`,
    with the per-timestep loop moved inside one function so it can be
    `torch.compile`d as a single graph (see those two docstrings for why
    compiling the whole per-chunk loop -- not the whole model, and not one
    timestep at a time -- is the fast-and-still-exact choice for this
    family of recurrence). Purely a refactor: same math as `gdn2_fix_scan`,
    verified 0.0 diff against it (query/key normalization hoisted out of
    the loop since it doesn't depend on the running state).
    """
    if normalize_key:
        key = key / key.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    outputs = []
    for t in range(query.shape[1]):
        state, out_t = _gdn2_fix_step_prenormalized(state, query[:, t], key[:, t], value[:, t], alpha[:, t], erase[:, t], write[:, t])
        outputs.append(out_t)
    return state, torch.stack(outputs, dim=1)


class GDN2FixMixer(nn.Module):
    """Reference mixer with independent vector erase/write gates."""

    _seq_fn = staticmethod(_gdn2_fix_sequential)

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.in_proj = nn.Linear(dim, 6 * dim)
        self.out_proj = nn.Linear(dim, dim)
        # Choose a small positive log-decay scale so the new parameterization
        # starts near the old ~0.99 retention regime.
        # One shared learned scale keeps the successor parameter count
        # comparable to the frozen baseline while retaining log-decay a.
        self.decay_a = nn.Parameter(torch.full((1,), -6.13))
        with torch.no_grad():
            self.in_proj.bias[: 3 * dim].zero_()
            self.in_proj.bias[3 * dim:4 * dim].fill_(4.59512)
            self.in_proj.bias[4 * dim:5 * dim].fill_(-4.59512)
            self.in_proj.bias[5 * dim:].fill_(-4.59512)

    def forward(self, x, state=None):
        batch, steps, _ = x.shape
        projected = self.in_proj(x).view(batch, steps, 6, self.heads, self.head_dim)
        query, key, value, alpha, erase, write = projected.unbind(dim=2)
        query = query / query.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(x.dtype)
        key = key / key.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(x.dtype)
        decay_rate = torch.exp(self.decay_a).view(1, 1, 1, 1)
        alpha = torch.exp(-decay_rate * torch.nn.functional.softplus(alpha.float())).to(x.dtype)
        erase = torch.sigmoid(erase.float()).to(x.dtype)
        write = torch.sigmoid(write.float()).to(x.dtype)
        if state is None:
            state = torch.zeros(batch, self.heads, self.head_dim, self.head_dim, dtype=value.dtype, device=value.device)
        state, output = type(self)._seq_fn(state, query, key, value, alpha, erase, write, normalize_key=False)
        return self.out_proj(output.reshape(batch, steps, self.dim)), state
