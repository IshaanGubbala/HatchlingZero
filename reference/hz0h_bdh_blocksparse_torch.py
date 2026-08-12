"""HZ Phase 4 (`plans/HatchlingZero_Reality_Plan.md`): BlockBDH -- turn
BDH's neuronal sparsity into REAL skipped computation, not just a
theoretical FLOP reduction.

BDH already produces sparse positive activations (ReLU) -- measured real
in Phase 1 (`docs/restart/hz0h_phase1_...` activation-sparsity work,
~13-53% zero depending on layer/path). Phase 4's actual question is
narrower and harder: does SKIPPING the FLOPs for the inactive neurons
(not just zeroing them after computing them anyway) produce a REAL
wall-clock or energy win on real hardware, or does gather/scatter
overhead eat the theoretical saving -- exactly the risk
`plans/HatchlingZero_Reality_Plan.md`'s own "Risk 2" names explicitly
("Modern hardware is extremely efficient at dense GEMM... Never claim a
speedup from FLOP reduction alone").

Design here, matching the plan's own "Implementation Priorities" (large
fixed blocks, deterministic capacity, avoid fine-grained sparsity until
it proves a real win): a CHEAP, COARSE router picks the top-k active
BLOCKS of the encoder's N-dimension for an entire forward call (not
per-token -- per-token routing would need grouped/sorted dispatch
machinery this first experiment deliberately doesn't build yet, per the
plan's own "avoid arbitrary fine-grained sparsity until it produces real
wall-clock wins"). Only the SELECTED columns of `encoder`/`encoder_v`/
`decoder` are ever multiplied -- a real, smaller matmul via
`index_select`, not a full computation with a mask applied after.

This is a real, explicit divergence from exact BDH (a coarser router
than the plan's own eventual per-token design), evaluated honestly
against dense compute at several active fractions -- if it's a net loss
at every fraction tried, that is exactly the kind of result Phase 4's
own exit gate ("promote only if it produces real end-to-end
speed/energy improvements") exists to catch, and will be reported as
such, not hidden.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, BDHConfig


def compute_active_blocks(model: BDH, idx: torch.Tensor, block_size: int, active_fraction: float, exploration_noise: float = 0.0) -> torch.Tensor:
    """Cheap router: one forward-ish pass computing x_sparse for the
    FIRST layer only (reusing the model's own encoder, no extra learned
    router parameters -- the cheapest possible real router), aggregates
    mean activation magnitude per block across the whole batch/sequence,
    and returns the indices of the top-k active blocks. Real, coarse,
    call-level granularity (not per-token) -- see module docstring.

    `exploration_noise`: real, disclosed fix for a diagnosed failure
    mode (`docs/restart/hz0h_phase4_blocksparse_results.md`'s Update 5):
    with `exploration_noise=0` (the original, default behavior), the
    router is a deterministic top-k on activation magnitude -- a "rich
    get richer" dynamic where whichever blocks happen to score slightly
    higher early in training get selected, receive all the gradient,
    grow further, and lock in permanently (observed directly: bad-seed
    training runs freeze their block SET by step ~200 and then the loss
    plateaus for the rest of training, while good-seed runs keep
    exploring different block sets until step ~600 and keep improving
    long after). Setting `exploration_noise > 0` adds real Gumbel-style
    noise to the block scores before top-k (`score + noise *
    -log(-log(uniform))`), giving every block a real, decaying-over-
    training-if-scheduled-externally chance of selection instead of a
    fully deterministic, self-reinforcing choice."""
    with torch.no_grad():
        C = model.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        n_blocks = N // block_size
        n_active = max(1, round(n_blocks * active_fraction))

        x = model.ln(model.embed(idx).unsqueeze(1))
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)  # (B, nh, T, N)
        block_scores = x_sparse.reshape(*x_sparse.shape[:-1], n_blocks, block_size).abs().mean(dim=(0, 1, 2, 4))  # (n_blocks,)
        if exploration_noise > 0:
            uniform = torch.rand_like(block_scores).clamp_min(1e-9)
            gumbel = -torch.log(-torch.log(uniform))
            block_scores = block_scores + exploration_noise * gumbel
        active_blocks = torch.topk(block_scores, n_active).indices.sort().values
        return active_blocks


def block_balance_loss(model: BDH, idx: torch.Tensor, block_size: int) -> torch.Tensor:
    """Real load-balancing AUXILIARY loss (`docs/restart/hz0h_phase4_blocksparse_results.md`
    Update 6's proposed fix, replacing the failed `exploration_noise`
    attempt above): the router in `compute_active_blocks` has NO learned
    parameters of its own -- it just reads mean activation magnitude of
    `model.encoder`'s own output. So the real lever for fixing the
    "rich get richer" lock-in isn't the router, it's `encoder` itself:
    this loss pushes `encoder` (via a real gradient, unlike
    `compute_active_blocks` which runs under `torch.no_grad()`) to
    produce block activation magnitudes that are MORE evenly spread,
    so the (still deterministic, still un-learned) top-k selection
    naturally has less of a permanent few-blocks-dominate attractor.

    `p = softmax(block_scores)`; loss = `n_blocks * sum(p_i^2)`, the
    standard squared-coefficient-of-variation-style load-balancing
    loss (Shazeer et al. sparse-MoE lineage) -- minimized exactly at
    uniform `p` (loss=1), maximized when one block captures all mass
    (loss=n_blocks). Combine as `total_loss = lm_loss + lambda_balance *
    block_balance_loss(...)` in the training loop; this function computes
    ONLY the auxiliary term, with gradient enabled (no `torch.no_grad()`),
    since blocking gradient here would make it a no-op identical to the
    noise-injection attempt it's replacing."""
    C = model.config
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    n_blocks = N // block_size

    x = model.ln(model.embed(idx).unsqueeze(1))
    x_latent = x @ model.encoder
    x_sparse = F.relu(x_latent)
    block_scores = x_sparse.reshape(*x_sparse.shape[:-1], n_blocks, block_size).abs().mean(dim=(0, 1, 2, 4))
    p = F.softmax(block_scores, dim=0)
    return n_blocks * (p * p).sum()


def bdh_blocksparse_forward(model: BDH, idx: torch.Tensor, active_blocks: torch.Tensor, block_size: int, targets: torch.Tensor | None = None):
    """Same real per-layer computation as `BDH.forward`, except
    `encoder`/`encoder_v`/`decoder` are only ever multiplied through
    their SELECTED columns (`index_select`, a real smaller matmul) --
    inactive blocks contribute exactly zero, not approximately zero.

    Real correctness subtlety, found while building this: RoPE's
    frequency buffer (`model.attn.freqs`, shape `(1,1,1,N)`) has one
    fixed frequency PER N-index -- gathering a subset of N-columns for
    `x_sparse` without gathering the SAME subset of frequencies would
    silently rotate each surviving column by the WRONG phase (and even
    crash on a shape mismatch, since the gathered Q no longer has width
    N). Bypasses `model.attn(...)`'s black-box call (which always uses
    the model's own full-width `self.freqs`) and reimplements the
    attention math directly with a correspondingly-gathered
    `sparse_freqs`. `block_size` must be even -- RoPE's rotation pairs
    adjacent indices `(2i, 2i+1)`, and since blocks are contiguous
    (gathered, not interleaved), an odd block_size would split a
    rotation pair across a block boundary."""
    if block_size % 2 != 0:
        raise ValueError(f"block_size must be even (RoPE pairs adjacent indices) -- got {block_size}")

    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    column_indices = (active_blocks.view(-1, 1) * block_size + torch.arange(block_size, device=idx.device)).reshape(-1)  # (n_active*block_size,)
    n_active_cols = column_indices.numel()

    encoder_sparse = model.encoder.index_select(2, column_indices)  # (nh, D, n_active_cols)
    encoder_v_sparse = model.encoder_v.index_select(2, column_indices)
    decoder_sparse = model.decoder.reshape(nh, N, D).index_select(1, column_indices).reshape(nh * n_active_cols, D)
    sparse_freqs = model.attn.freqs.index_select(-1, column_indices)  # (1,1,1,n_active_cols)

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _level in range(C.n_layer):
        x_latent = x @ encoder_sparse  # (B, nh, T, n_active_cols) -- real smaller matmul
        x_sparse = F.relu(x_latent)

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
