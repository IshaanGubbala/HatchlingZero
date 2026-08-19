"""Tests the middle ground between full depth-tying and full depth-
untying (`reference/hz0h_bdh_depth_untied_torch.py`) that motivated this
follow-up: does each recurrent group really need its OWN full-size
encoder/encoder_v/decoder, or can it share one big base set and only pay
for a small per-group correction?

Part 4c (`docs/restart/hz0h_inherited_choices_audit_results.md`) found
that full-capacity untying (every group gets its own FULL-size weights)
matches or beats tied quality at real depth (`n_layer=8`), but costs up
to 7.72x the parameters at `groups=8` -- an expensive way to buy that
win. `AdapterDepthBDH` tests whether most of that win is available for
much less: keep ONE shared full-size weight set per matrix, and give
each group only a small low-rank correction on top,

    W_group = W_shared + A_group @ B_group

with `A_group`/`B_group` rank `rank << N` (`rank` is a fraction of the
matrix's own inner dimension, so the extra parameters per group are
`O(rank)` instead of `O(N)`). This is the standard LoRA-style
factorization, applied here across RECURRENT GROUPS instead of across
layers of a stacked network.

`B_group` is zero-initialized (`A_group` is not) -- the standard trick
for this factorization -- so `A_group @ B_group` is EXACTLY zero at
construction and every group starts out computing the identical tied
forward pass. This gives a strong, bit-exact correctness gate (see
`tests/reference/test_hz0h_bdh_depth_adapter_torch.py`): only after
training does divergence between groups appear, and only from the small
`A`/`B` factors, not from any hidden asymmetry.

Never modifies `reference/hz0h_bdh_torch.py` (read-only upstream
oracle).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, BDHConfig


class AdapterDepthBDH(nn.Module):
    """`embed`/`ln`/`attn`/`lm_head` and one shared `encoder`/`encoder_v`/
    `decoder` set (all borrowed from a fresh `BDH(config)`, the oracle's
    own init) are shared across every recurrent group, exactly like the
    tied oracle. Each of `groups` independent groups additionally gets a
    small rank-`rank` correction per matrix, applied at forward time as
    `W_shared + A_group @ B_group`. `depth // groups` adjacent levels
    share one group's adapter, same `group_of` convention as
    `DepthUntiedBDH`.
    """

    def __init__(self, config: BDHConfig, depth: int, groups: int, rank: int):
        super().__init__()
        self.config = config
        self.depth = depth
        self.groups = groups
        self.rank = rank
        assert 1 <= groups <= depth
        assert rank >= 1

        shared = BDH(config)
        self.embed = shared.embed
        self.ln = shared.ln
        self.attn = shared.attn
        self.drop = shared.drop
        self.lm_head = shared.lm_head
        self.shared_encoder = shared.encoder      # (nh, D, N)
        self.shared_encoder_v = shared.encoder_v  # (nh, D, N)
        self.shared_decoder = shared.decoder      # (nh*N, D)

        nh = config.n_head
        D = config.n_embd
        N = D * config.mlp_internal_dim_multiplier // nh

        self.enc_A = nn.ParameterList()
        self.enc_B = nn.ParameterList()
        self.encv_A = nn.ParameterList()
        self.encv_B = nn.ParameterList()
        self.dec_A = nn.ParameterList()
        self.dec_B = nn.ParameterList()
        for _ in range(groups):
            self.enc_A.append(nn.Parameter(torch.zeros(nh, D, rank).normal_(std=0.02)))
            self.enc_B.append(nn.Parameter(torch.zeros(nh, rank, N)))
            self.encv_A.append(nn.Parameter(torch.zeros(nh, D, rank).normal_(std=0.02)))
            self.encv_B.append(nn.Parameter(torch.zeros(nh, rank, N)))
            self.dec_A.append(nn.Parameter(torch.zeros(nh * N, rank).normal_(std=0.02)))
            self.dec_B.append(nn.Parameter(torch.zeros(rank, D)))

    def group_of(self, level: int) -> int:
        return level * self.groups // self.depth

    def encoder_for(self, group: int) -> torch.Tensor:
        return self.shared_encoder + self.enc_A[group] @ self.enc_B[group]

    def encoder_v_for(self, group: int) -> torch.Tensor:
        return self.shared_encoder_v + self.encv_A[group] @ self.encv_B[group]

    def decoder_for(self, group: int) -> torch.Tensor:
        return self.shared_decoder + self.dec_A[group] @ self.dec_B[group]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, depth: int | None = None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        depth = self.depth if depth is None else depth

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)

        for level in range(depth):
            group = self.group_of(level)
            encoder = self.encoder_for(group)
            encoder_v = self.encoder_v_for(group)
            decoder = self.decoder_for(group)

            x_latent = x @ encoder
            x_sparse = F.relu(x_latent)

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            y_latent = yKV @ encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            nh = C.n_head
            N = D * C.mlp_internal_dim_multiplier // nh
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def adapter_parameter_count(self) -> int:
        """Real count of JUST the low-rank correction parameters (A/B
        pairs across all groups), excluding the shared base weights --
        this is the actual param overhead of untying via this method,
        the number to compare against DepthUntiedBDH's full per-group
        matrices."""
        return sum(
            p.numel() for group in range(self.groups)
            for p in (self.enc_A[group], self.enc_B[group],
                      self.encv_A[group], self.encv_B[group],
                      self.dec_A[group], self.dec_B[group])
        )
