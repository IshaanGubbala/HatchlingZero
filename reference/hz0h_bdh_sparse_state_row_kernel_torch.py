"""Tier 3 item 20 of plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md
(Phase SR-C): a real, GPU-parallel (no Python loop over the batch)
implementation of item 18's correctness-proven sparse-state-row oracle,
gated behind the real 2.101x ceiling measured for this path (go/no-go
note under Phase SR-B in the plan).

Unlike item 17's batched exact-skip (padded over B*T=2048 tokens, which
OOM'd and was 126x slower in the real GPU test), THIS kernel pads over
just the decode BATCH dimension B (typically 1-32 for real serving, not
2048) -- the specific failure mode that killed item 17 (one worst-case
token inflating padding for an entire large token batch) is structurally
much less severe here, per the plan-update note recorded alongside the
ceiling check. Still benchmarked for real wall-clock speed before being
treated as a win -- a passing ceiling does not guarantee a passing real
implementation, exactly the lesson item 17 already taught.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_sparse_state_row_oracle_torch import active_pair_mask


def bdh_stream_step_sparse_row_batched(model, states: list, idx_token: torch.Tensor, position: int) -> tuple[list, torch.Tensor]:
    """Single-token (L=1) decode step, batched over B with no Python loop
    over batch or head -- only a real loop over `nh` (typically 8), same
    as item 17's precedent (looping over heads, not tokens/batch, is
    cheap and was never the bottleneck there)."""
    c = model.config
    B, L = idx_token.shape
    assert L == 1, "scoped to single-token decode steps"
    D = c.n_embd
    nh = c.n_head
    N = D * c.mlp_internal_dim_multiplier // nh
    device = idx_token.device

    x = model.ln(model.embed(idx_token).unsqueeze(1))
    r_phases = torch.tensor([[position]], device=device, dtype=model.attn.freqs.dtype).view(1, 1, 1, 1) * model.attn.freqs

    new_states = []
    for level in range(c.n_layer):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)  # (B, nh, 1, N)
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        V = x  # (B, 1, 1, D)

        prefix_state = states[level]  # (B, nh, N, D)
        pair_active = active_pair_mask(x_sparse)  # (B, nh, 1, N//2)

        yKV = torch.zeros(B, nh, 1, D, dtype=x.dtype, device=device)
        write_state = prefix_state.clone()

        for h in range(nh):  # real loop is only over heads, not batch or tokens
            row_active = pair_active[:, h, 0].repeat_interleave(2, dim=-1)  # (B, N) -- expand pair mask to row mask
            support_count = row_active.sum(dim=-1)  # (B,)
            max_k = int(support_count.max().item())
            if max_k == 0:
                continue

            sorted_idx = torch.argsort(row_active.int(), dim=-1, descending=True, stable=True)  # (B, N)
            gathered_idx = sorted_idx[:, :max_k]  # (B, max_k)
            valid = torch.arange(max_k, device=device).unsqueeze(0) < support_count.unsqueeze(1)  # (B, max_k)

            QR_h = QR[:, h, 0]  # (B, N)
            state_h = prefix_state[:, h]  # (B, N, D)
            QR_gathered = torch.gather(QR_h, 1, gathered_idx) * valid  # (B, max_k)
            state_gathered = torch.gather(state_h, 1, gathered_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, max_k, D)
            yKV[:, h, 0] = torch.einsum("bk,bkd->bd", QR_gathered, state_gathered)

            KR_h = KR[:, h, 0]
            KR_gathered = torch.gather(KR_h, 1, gathered_idx) * valid  # (B, max_k)
            V_h = V[:, 0, 0]  # (B, D)
            update = torch.einsum("bk,bd->bkd", KR_gathered, V_h)  # (B, max_k, D) -- outer product per active row
            write_state[:, h].scatter_add_(1, gathered_idx.unsqueeze(-1).expand(-1, -1, D), update)

        yKV = model.ln(yKV)
        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, 1, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

        new_states.append(write_state)

    logits = x.view(B, 1, D) @ model.lm_head
    return new_states, logits
