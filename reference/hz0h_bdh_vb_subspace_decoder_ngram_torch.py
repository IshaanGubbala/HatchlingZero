"""N-gram / hashed lexical memory for BDHVBSubspaceDecoder, Phase 3 of
plans/HatchlingZero_Qwen_Integration_Plan_2026-08-26.md#7.

Real, deliberately minimal first version of Qwen3.8-Flash-Next's n-gram
embedding idea (their real version: a 20,000,000-entry bigram/trigram
table contributing 51B of their 180B total params, host-memory resident
with async prefetch). This version: a single hashed embedding table for
one fixed n-gram order, GPU-resident (no host-memory/prefetch machinery
-- that's only worth building if the core mechanism shows a real
quality win first), injected additively into the input embedding once,
before the recurrent round loop, gated by one learnable scalar that
starts near zero so the injection can't hurt at initialization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from reference.hz0h_bdh_vb_subspace_decoder_checkpointed_torch import _vb_subspace_decoder_checkpoint_iteration
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder


def ngram_hash(idx: torch.Tensor, order: int, table_size: int, base: int = 131) -> torch.Tensor:
    """Real vectorized polynomial rolling hash over the last `order` bytes
    ending at each position (idx is byte-level, values 0-255; base=131 and
    order<=4 keeps every intermediate product well inside int64 range, no
    modular reduction needed until the final mod table_size). Left-padded
    with zero bytes so early-sequence positions (fewer than `order` real
    bytes of history) get a defined, if degenerate, hash instead of
    wrapping garbage -- a real limitation (no true history at window
    start), disclosed rather than hidden, and irreducible without
    cross-window context the data pipeline doesn't carry."""
    B, T = idx.shape
    padded = F.pad(idx, (order - 1, 0), value=0)
    windows = padded.unfold(1, order, 1)  # (B, T, order)
    mults = base ** torch.arange(order, device=idx.device, dtype=torch.int64)
    h = (windows.to(torch.int64) * mults).sum(dim=-1)
    return h % table_size


class NgramHashedMemory(nn.Module):
    def __init__(self, table_size: int, d_embd: int, order: int, alpha_init: float = 0.01,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(table_size, d_embd, dtype=dtype, device=device).normal_(std=0.02))
        # Real subtlety: alpha=0.0 EXACTLY would zero the gradient reaching
        # `table` too (d(alpha*e)/de = alpha), so the table would never
        # learn from a hard-zero start. "Near zero", per the plan, not
        # zero -- both alpha and table get real signal from step 1.
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32, device=device))
        self.order = order
        self.table_size = table_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        h = ngram_hash(idx, self.order, self.table_size)
        return self.table[h] * self.alpha


def bdh_vb_subspace_decoder_forward_ngram_checkpointed(
    model: BDHVBSubspaceDecoder,
    idx: torch.Tensor,
    n_iterations: int,
    targets: torch.Tensor | None = None,
):
    """model.ngram_memory must already be attached (see add_ngram_memory)
    -- accessed via attribute, same convention as the MTP heads, so this
    function's signature matches the plain checkpointed forward's shape
    (model, idx, n_iterations, targets) and can be swapped in per-arm."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)
    x = x + model.ngram_memory(idx).unsqueeze(1)

    for _iteration in range(n_iterations):
        x = torch.utils.checkpoint.checkpoint(
            _vb_subspace_decoder_checkpoint_iteration, x, model, B, T, D, nh, N, use_reentrant=False,
        )

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss


def add_ngram_memory(model: BDHVBSubspaceDecoder, table_params: int, order: int) -> NgramHashedMemory:
    """Registers ngram_memory as a real submodule (plain attribute
    assignment on an nn.Module -- same mechanism P/O and the MTP heads
    already use), so model.parameters()/named_parameters() and
    model.state_dict() pick it up automatically, no base-class changes."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    table_size = max(1, table_params // model.config.n_embd)
    ngram = NgramHashedMemory(table_size, model.config.n_embd, order, dtype=dtype, device=device)
    model.ngram_memory = ngram
    print(f"[ngram] order={order} table_size={table_size} real_params={table_size * model.config.n_embd}", flush=True)
    return ngram
