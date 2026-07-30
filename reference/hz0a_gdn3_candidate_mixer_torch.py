"""Torch/CUDA port of `reference/hz0a_gdn3_candidate_mixer.py` -- the MLX
version only runs on Apple/Metal, same wall this project already hit with
the main training runner (`docs/rtx3060_windows_setup.md`), so the GDN-3
candidate investigation needs its own torch path to run on the RTX 3060
(or any CUDA/CPU machine) at all.

Mirrors `reference/hz0a_torch_model.py`'s `GDN2Mixer` conventions exactly
(same `in_proj`/state shape/forward signature style) so the two are a fair
side-by-side comparison in torch, the same way the MLX versions are.
Device-agnostic (works on cpu/mps/cuda -- whatever tensors are already on).
"""
from __future__ import annotations

import torch
from torch import nn


class GDN3CandidateMixerTorch(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int | None = None):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.head_dim = head_dim if head_dim is not None else dim // heads
        width = heads * self.head_dim
        # dim -> 6*width, matching reference/hz0a_torch_model.py's GDN2Mixer
        # in_proj sizing convention (heads * (4*d_k + 2*d_v), here with
        # d_k==d_v==head_dim) -- q,k,v,decay,beta,unused_padding. The padding
        # slot exists ONLY for parameter-count parity with GDN2Mixer (which
        # has 6 slots: q,k,v,decay,erase,write) -- see the MLX version's
        # docstring for why this parity matters (an earlier, unmatched
        # version gave a confounded, misleading comparison result).
        self.in_proj = nn.Linear(dim, 6 * width)
        self.out_proj = nn.Linear(width, dim)
        with torch.no_grad():
            self.in_proj.bias[:3 * width].zero_()
            self.in_proj.bias[3 * width:4 * width].fill_(4.59512)   # decay -> sigmoid ~0.99, retain by default
            self.in_proj.bias[4 * width:5 * width].fill_(-4.59512)  # beta -> sigmoid ~0.01, small write strength by default
            self.in_proj.bias[5 * width:].zero_()                   # unused padding

    def forward(self, x, state):
        c, bsz, steps = self, x.shape[0], x.shape[1]
        p = self.in_proj(x).view(bsz, steps, 6, c.heads, c.head_dim)
        q, k, v, decay_logit, beta_logit = p[..., 0, :, :], p[..., 1, :, :], p[..., 2, :, :], p[..., 3, :, :], p[..., 4, :, :]
        if state is None:
            state = torch.zeros(bsz, c.heads, c.head_dim, c.head_dim, device=x.device, dtype=x.dtype)
        decay = torch.sigmoid(decay_logit)
        beta = torch.sigmoid(beta_logit)
        # Same fix as the MLX version, same reason: (I - beta*k*k^T) is
        # only a proper projection for unit-norm k -- an unnormalized
        # learned k can blow the recurrence up (verified in the MLX port;
        # not re-derived here, this is a direct, deliberate match).
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        outputs = []
        for t in range(steps):
            decayed = decay[:, t, :, None, :] * state
            old_retrieved = (decayed * k[:, t, :, None, :]).sum(dim=-1)
            correction = beta[:, t, :, :, None] * (v[:, t] - old_retrieved)[:, :, :, None] * k[:, t, :, None, :]
            state = decayed + correction
            outputs.append((state * q[:, t, :, None, :]).sum(dim=-1))
        return self.out_proj(torch.stack(outputs, dim=1).reshape(bsz, steps, c.heads * c.head_dim)), state
