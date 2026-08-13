"""HZ Next-Phase Plan Phase I4.1 (`plans/HatchlingZero_Next_Phase_Plan.md`,
"Selective BlockBDH -- continuous content gating before hard sparsity"):
the dense-gated phase, real first step before any soft-to-hard annealing
or hard inference. Every block of `encoder`'s N-dimension is ALWAYS
computed (no `index_select`, no skipped compute -- unlike
`reference/hz0h_bdh_blocksparse_torch.py`'s real sparse execution path)
but a small, real, LEARNED per-block gate scales each block's
contribution before it flows into the rest of BDH's computation.

Real motivation, per the plan: BlockBDH's hard top-k router (hard from
step 0) showed real training instability (router lock-in, three
mechanistically distinct fixes failed -- see
`docs/restart/hz0h_phase4_blocksparse_results.md` Updates 6-8). This is
a DIFFERENT mechanism, not a fourth tweak on that same family: instead
of a hard, non-differentiable top-k selection from the start, every
block gets a continuous, fully differentiable gate in [0, 1], so
gradient reaches every block every step (unlike hard top-k's exactly-
zero gradient to unselected blocks) -- the real hypothesis is that
learning WHICH blocks matter under a soft, always-differentiable signal
is more stable than being forced to commit to a hard selection
immediately. Only after this dense-gated phase is validated do later
phases (I4.2 soft-to-sparse annealing, I4.3 hard inference) consider
converting learned gate preferences into real skipped compute.

Real, explicit scope, matching the plan's own text: no full
input-dependent encoder-matrix generation (Mamba-style), no slow/fast
hierarchy -- one small, shared (tied across iterations, matching BDH's
own weight-sharing convention), real learned gate parameter per model,
its OUTPUT varying every iteration because the input `x` varies, not
because the gate's own weights do.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import Attention, BDHConfig


@dataclasses.dataclass
class BDHBlockGatedConfig(BDHConfig):
    block_size: int = 4  # must be even (RoPE pairs adjacent indices), matching BlockBDH's own constraint


class BDHBlockGated(torch.nn.Module):
    def __init__(self, config: BDHBlockGatedConfig):
        super().__init__()
        assert config.vocab_size is not None
        if config.block_size % 2 != 0:
            raise ValueError(f"block_size must be even (RoPE pairs adjacent indices) -- got {config.block_size}")
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        if N % config.block_size != 0:
            raise ValueError(f"N ({N}) must be divisible by block_size ({config.block_size})")
        n_blocks = N // config.block_size
        self.n_blocks = n_blocks

        self.decoder = torch.nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = torch.nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = Attention(config)
        self.ln = torch.nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = torch.nn.Embedding(config.vocab_size, D)
        self.drop = torch.nn.Dropout(config.dropout)
        self.encoder_v = torch.nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.lm_head = torch.nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))

        # The one real addition vs plain BDH: a small, shared (tied
        # across every iteration, same convention as encoder/encoder_v/
        # decoder) gate that maps the CURRENT residual stream to a
        # per-block gate logit. Its output varies every iteration
        # because x does; its own weights do not change across
        # iterations.
        self.gate = torch.nn.Parameter(torch.zeros((D, n_blocks)).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        return bdh_block_gated_forward(self, idx, targets=targets)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def compute_block_gate(model: BDHBlockGated, x: torch.Tensor) -> torch.Tensor:
    """Real, differentiable per-block gate in (0, 1), computed fresh
    from the CURRENT residual stream every call -- `x` has shape
    (B, 1, T, D); returns (B, T, n_blocks)."""
    return torch.sigmoid(x.squeeze(1) @ model.gate)


def bdh_block_gated_forward(model: BDHBlockGated, idx: torch.Tensor, targets: torch.Tensor | None = None, return_gates: bool = False):
    """Dense computation -- every block of `encoder`'s N-dimension is
    ALWAYS computed, no `index_select`, no skipped FLOPs (unlike
    `bdh_blocksparse_forward`'s real sparse execution). The gate scales
    `x_sparse` block-wise BEFORE it feeds attention (Q=K=gated
    x_sparse) -- this is the one, single gating point; y_sparse and
    xy_sparse are not gated a second time, since they already inherit
    the gate's effect through `yKV` (which depends on the gated Q/K)."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    n_blocks = model.n_blocks
    block_size = C.block_size

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    all_gates = [] if return_gates else None

    for _level in range(C.n_layer):
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)  # (B, nh, T, N)

        gate = compute_block_gate(model, x)  # (B, T, n_blocks)
        if return_gates:
            all_gates.append(gate)
        gate_expanded = gate.unsqueeze(1).unsqueeze(-1).expand(B, nh, T, n_blocks, block_size).reshape(B, nh, T, N)
        x_sparse = x_sparse * gate_expanded

        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    if return_gates:
        return logits, loss, all_gates
    return logits, loss


# --- Phase I4.2: soft-to-sparse annealing ----------------------------------
#
# I4.1's real result (docs/restart/hz0h_phase_i4_block_gated_results.md
# Update 1): a diversity-anchored continuous gate is stable and BEATS the
# original hard top-k baseline (mean 0.93 vs 0.868 on the reassignment
# task, 5 seeds) -- but it's still dense, no real FLOP savings. This
# section converts a trained gate's preferences into real skipped
# compute, gradually, rather than imposing a hard cutoff from step 0
# (the exact mistake that made the original BlockBDH router unstable).


def compute_active_blocks_by_gate(model: "BDHBlockGated", idx: torch.Tensor, active_fraction: float) -> torch.Tensor:
    """Real, cheap router for the WHOLE call: ranks blocks by the
    LEARNED gate's value (not raw activation magnitude, unlike
    `reference/hz0h_bdh_blocksparse_torch.py`'s `compute_active_blocks`)
    -- computed once from the embedding-layer state, matching that
    function's own call-level (not per-token) granularity."""
    with torch.no_grad():
        n_blocks = model.n_blocks
        n_active = max(1, round(n_blocks * active_fraction))
        x = model.ln(model.embed(idx).unsqueeze(1))
        gate = compute_block_gate(model, x)  # (B, T, n_blocks)
        block_scores = gate.mean(dim=(0, 1))
        active_blocks = torch.topk(block_scores, n_active).indices.sort().values
        return active_blocks


def bdh_block_gated_annealed_forward(model: "BDHBlockGated", idx: torch.Tensor, active_fraction: float, targets: torch.Tensor | None = None):
    """`active_fraction >= 1.0` -> identical to `bdh_block_gated_forward`
    (dense, soft-gated, nothing skipped). `active_fraction < 1.0` -> real
    hard selection of the top-`(active_fraction * n_blocks)` blocks,
    ranked by the LEARNED gate (not raw activation magnitude), with real
    `index_select` compute savings on the unselected blocks -- same real
    RoPE-frequency-gathering fix `bdh_blocksparse_forward` needed
    (`model.attn.freqs` must be gathered with the SAME column indices as
    the activations). The continuous gate is NOT discarded once hard
    selection kicks in -- recomputed fresh each iteration, restricted to
    the selected columns, and still applied as a soft weight on top of
    the hard selection."""
    if active_fraction >= 1.0:
        return bdh_block_gated_forward(model, idx, targets=targets)

    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    block_size = C.block_size

    active_blocks = compute_active_blocks_by_gate(model, idx, active_fraction)
    n_active_blocks = active_blocks.numel()
    column_indices = (active_blocks.view(-1, 1) * block_size + torch.arange(block_size, device=idx.device)).reshape(-1)
    n_active_cols = column_indices.numel()

    encoder_sparse = model.encoder.index_select(2, column_indices)
    encoder_v_sparse = model.encoder_v.index_select(2, column_indices)
    decoder_sparse = model.decoder.reshape(nh, N, D).index_select(1, column_indices).reshape(nh * n_active_cols, D)
    sparse_freqs = model.attn.freqs.index_select(-1, column_indices)

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _level in range(C.n_layer):
        x_latent = x @ encoder_sparse
        x_sparse = F.relu(x_latent)

        full_gate = compute_block_gate(model, x)  # (B, T, n_blocks)
        selected_gate = full_gate.index_select(-1, active_blocks)  # (B, T, n_active_blocks)
        gate_expanded = selected_gate.unsqueeze(1).unsqueeze(-1).expand(B, nh, T, n_active_blocks, block_size).reshape(B, nh, T, n_active_cols)
        x_sparse = x_sparse * gate_expanded

        r_phases = torch.arange(0, T, device=idx.device, dtype=sparse_freqs.dtype).view(1, 1, -1, 1) * sparse_freqs
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        scores = (QR @ KR.mT).tril(diagonal=-1)
        yKV = scores @ x
        yKV = model.ln(yKV)

        y_latent = yKV @ encoder_v_sparse
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, n_active_cols * nh) @ decoder_sparse
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
