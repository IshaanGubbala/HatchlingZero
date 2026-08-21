"""Hard domain-block masking for BDH's latent width -- the foundational
mechanism for the structural specialization design proposed 2026-08-21,
directly motivated by Part 11's real finding (this project's own audit,
`docs/restart/hz0h_inherited_choices_audit_results.md`): ordinary
training does NOT make BDH's neurons specialize by domain on their own
(within/across-domain support-overlap ratio stayed flat at 1.03x-1.18x
at both 2M and 10M real tokens). If specialization is wanted, it has to
be forced structurally, not hoped for from the loss alone.

The design: partition the per-head latent width `N` into a SHARED block
plus one block per named domain (sizes configurable, real default
matches the proposal: 25% shared, 15% each for code/math/prose/
reasoning/tools). For a given training batch's domain label, only the
shared block plus that domain's own block may contribute to the
forward pass, and -- the real, load-bearing property, not just a
forward-pass gate -- ONLY those blocks' encoder/encoder_v/decoder
weight columns receive any gradient at all.

That gradient-isolation property is exact, not approximate, and needs
no custom backward: masking `u=x_sparse` and `v=y_sparse` (both real
ReLU outputs) with a 0/1 mask, in that order, gives d(loss)/d(u[n]) =
d(loss)/d(u_masked[n]) * mask[n] = 0 for any masked-out `n`, by the
plain chain rule through the elementwise multiply -- which in turn
zeroes the gradient to `encoder`'s column `n` (since d(loss)/d(z[n])
before the ReLU is also exactly 0 whenever `u[n]`'s downstream
gradient is 0). The SAME argument applies to `y_sparse`/`encoder_v`.
`decoder`'s masked ROWS get zero gradient for free, with no separate
masking needed: `g = u * v` is already exactly zero at any index where
EITHER `u` or `v` was masked, so `decoder`'s gradient contribution
(`g_flat^T @ dyMLP`, summed over batch/time) is naturally zero at those
rows too. Proven, not asserted -- see
`tests/reference/test_hz0h_bdh_domain_masked_torch.py`'s
`test_masked_block_receives_exactly_zero_gradient`.

Real, deliberate scope for this first version: the CORE masking
mechanism only -- hard block routing via an explicit domain label
(no learned/soft router yet), no auxiliary losses (cross-domain
suppression, orthogonality, capacity balancing, expert dropout, mixed-
domain blending) which the full 2026-08-21 proposal also calls for.
Those are real, disclosed follow-up work, deliberately NOT built here
until this foundational mechanism itself is validated (matches this
project's own "measure before building further" discipline) -- a hard-
masking design with no specialization-preserving pressure at all could
just as easily let every domain's block collapse toward the shared
block's role, which is exactly the kind of question a real training
run (not just this correctness gate) needs to answer.

Same real per-layer computation as `bdh_variable_depth_forward` when
`domain_mask` is `None` (or all-ones) -- zero change to BDH's math in
that case, only an added elementwise mask when a real mask is supplied.
Never modifies `reference/hz0h_bdh_torch.py` or
`reference/hz0h_bdh_variable_depth_torch.py`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH


def domain_block_layout(n_per_head: int, domain_names: list[str], shared_fraction: float = 0.25) -> dict[str, torch.Tensor]:
    """Real, deterministic partition of the per-head latent width `N`
    into a SHARED index range plus one contiguous range per domain,
    covering `N` exactly once each (no overlap, no gap) -- returns a
    dict mapping `"shared"` and each domain name to a 1-D LongTensor of
    the indices it owns. Real, disclosed rounding behavior: domain
    blocks are sized equally from whatever's left after the shared
    fraction; any leftover from integer rounding is folded into the
    LAST domain's block so every index is owned by exactly one name."""
    if not (0.0 <= shared_fraction < 1.0):
        raise ValueError(f"shared_fraction must be in [0, 1), got {shared_fraction}")
    if not domain_names:
        raise ValueError("domain_names must be non-empty")
    shared_size = int(round(n_per_head * shared_fraction))
    remaining = n_per_head - shared_size
    per_domain = remaining // len(domain_names)
    layout: dict[str, torch.Tensor] = {"shared": torch.arange(0, shared_size)}
    cursor = shared_size
    for i, name in enumerate(domain_names):
        size = per_domain if i < len(domain_names) - 1 else (remaining - per_domain * (len(domain_names) - 1))
        layout[name] = torch.arange(cursor, cursor + size)
        cursor += size
    assert cursor == n_per_head, f"layout covers {cursor} of {n_per_head} indices -- real bug if this fires"
    return layout


def build_domain_mask(layout: dict[str, torch.Tensor], active_domain: str, n_per_head: int,
                       device=None, dtype=torch.float32) -> torch.Tensor:
    """Real 0/1 mask, shape `(N,)`: 1 at every index owned by `"shared"`
    or by `active_domain`, 0 everywhere else. `active_domain=None`
    returns an all-ones mask (dense/no masking -- the real regression
    check that this file's math matches the oracle exactly in that
    case)."""
    mask = torch.zeros(n_per_head, device=device, dtype=dtype)
    mask[layout["shared"]] = 1.0
    if active_domain is not None:
        if active_domain not in layout:
            raise ValueError(f"unknown domain {active_domain!r}, layout has {sorted(layout.keys())}")
        mask[layout[active_domain]] = 1.0
    return mask


def bdh_domain_masked_forward(
    model: BDH,
    idx: torch.Tensor,
    n_iterations: int,
    domain_mask: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
):
    """Same real per-layer computation as `bdh_variable_depth_forward`,
    with `x_sparse` (`u`) and `y_sparse` (`v`) both multiplied by
    `domain_mask` (shape `(N,)`, broadcasts over `B, n_head, T`) right
    after their ReLUs -- see this module's own docstring for the real,
    exact (not approximate) gradient-isolation property this gives.
    `domain_mask=None` skips masking entirely (identical to
    `bdh_variable_depth_forward`)."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    N = D * C.mlp_internal_dim_multiplier // C.n_head

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _iteration in range(n_iterations):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        if domain_mask is not None:
            x_sparse = x_sparse * domain_mask

        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        if domain_mask is not None:
            y_sparse = y_sparse * domain_mask
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        nh = C.n_head
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
