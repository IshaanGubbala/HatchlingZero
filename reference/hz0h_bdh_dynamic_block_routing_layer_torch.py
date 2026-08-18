"""Real, one-recurrent-layer BDH forward using per-token dynamic block
routing for the encoder projection, feeding into the UNMODIFIED oracle
`Attention` module.

Builds directly on `reference/hz0h_bdh_dynamic_block_routing_torch.py`
(routing + capacity-limited gather/scatter, already correctness- and
gradient-tested in isolation). This file adds the two real pieces
needed to actually use that mechanism inside a real BDH layer:

1. A real, CHEAP router. The isolated routing module took per-token
   block scores as an external input; a genuine router must produce
   those scores WITHOUT first computing the expensive full encoder
   projection (otherwise nothing is saved -- you did the expensive part
   just to decide what to skip). `DynamicBlockRouter` is a small,
   separate `(D, n_blocks)` gate per head, the real MoE pattern (a cheap
   router deciding before the expensive expert compute runs), not
   derived from `x @ encoder` itself.

2. Real, verified wiring into the oracle's own `Attention` module,
   UNCHANGED. Earlier framing in this session's own prior commit
   overstated attention as a hard, unsolved blocker for this mechanism
   -- that was wrong, corrected here with a real, load-bearing test:
   `Attention.forward` only computes dot products / matmuls on whatever
   tensor it's given. Since `dynamic_block_encoder_forward` already
   returns a full-width tensor with real, exact zeros in unrouted
   positions, feeding that directly into the unmodified oracle Attention
   is mathematically exact for what it computes -- a token's
   contribution to any block it wasn't routed to is genuinely absent,
   the same real semantics ReLU-induced sparsity already has in the
   dense oracle. There is no attention-specific code in this file at
   all; `model.attn` is called exactly as the oracle calls it.

Real, disclosed scope limit: `encoder_v` and `decoder` stay DENSE in
this first version (matching `PackedBlockBDH`'s own incremental
precedent of building one real, tested piece at a time) -- extending
dynamic routing to those steps is a real, separate follow-up, not done
here. `dynamic_block_routing_forward` wires the single-layer function
through the full `model.config.n_layer` recurrent loop with the SAME
tied router and weights every iteration (matching the oracle's own
real shared-weight convention), producing real logits/loss -- a
complete, trainable-shaped forward for the encoder-routed, dense-
encoder_v/decoder BDH variant. No CUDA execution, no real speed/quality
training run, and no end-to-end optimizer-step test exist yet -- those
remain real, disclosed, not-yet-done follow-ups.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_dynamic_block_routing_torch import (
    RoutingResult,
    dynamic_block_encoder_forward,
    route_tokens_to_blocks,
)
from reference.hz0h_bdh_torch import BDH


def init_dynamic_block_router(n_head: int, n_embd: int, n_blocks: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Real, cheap per-head router weight, `(n_head, n_embd, n_blocks)`
    -- small relative to the real encoder (`n_embd x N`, `N = n_blocks *
    block_size`), matching real MoE gating-network scale."""
    weight = torch.empty(n_head, n_embd, n_blocks)
    if generator is not None:
        weight.normal_(mean=0.0, std=0.02, generator=generator)
    else:
        weight.normal_(mean=0.0, std=0.02)
    return weight


def dynamic_block_routing_layer_forward(
    model: BDH,
    x: torch.Tensor,
    router: torch.Tensor,
    *,
    block_size: int,
    top_k: int,
    capacity_factor: float,
    apply_gate: bool = True,
) -> tuple[torch.Tensor, list[RoutingResult]]:
    """Real, one-recurrent-layer BDH forward: cheap router -> real
    per-token dynamic block routing -> gathered/scattered encoder
    projection -> ReLU -> UNMODIFIED oracle `Attention`. `encoder_v`/
    `decoder` stay dense (see module docstring's disclosed scope limit).

    `x`: `(B, 1, T, D)`, the oracle's own layer-input convention.
    `router`: `(nh, D, n_blocks)` from `init_dynamic_block_router`.
    `apply_gate`: default True -- multiplies each served block's output
    by its real, differentiable softmax gate weight, the only pathway
    gradient has back into `router` (see
    `dynamic_block_encoder_forward`'s own docstring: the discrete
    routing SELECTION is not differentiable, only this gate is). Real,
    disclosed tradeoff: with gating on, this does NOT reduce exactly to
    the oracle's own math even when every block is served (see that same
    docstring) -- pass `apply_gate=False` to get the exact-in-the-
    limiting-case behavior instead, at the cost of the router itself
    never receiving gradient (only useful with precomputed/static
    scores, not a jointly-trained router).
    Returns `(x_next, routing_results)` -- `x_next` is this layer's real
    output (`B, 1, T, D`, ready to feed into the next layer exactly like
    the oracle's own loop body), `routing_results` is one
    `RoutingResult` per head, for real inspection/diagnostics (e.g.
    measuring real drop rates).
    """
    C = model.config
    B, one, T, D = x.shape
    if one != 1:
        raise ValueError(f"BDH's own convention: x's head dim must be 1, got {x.shape}")
    nh = C.n_head
    encoder = model._w(model.encoder)  # (nh, D, N)
    N = encoder.shape[-1]
    if N % block_size:
        raise ValueError(f"N={N} must be divisible by block_size={block_size}")
    n_blocks = N // block_size
    if router.shape != (nh, D, n_blocks):
        raise ValueError(f"router shape {tuple(router.shape)} must be (nh={nh}, D={D}, n_blocks={n_blocks})")

    x_flat = x.reshape(B * T, D)  # same token data broadcast to every head, matching the oracle
    per_head_outputs = []
    routing_results: list[RoutingResult] = []
    for head in range(nh):
        scores = x_flat @ router[head]  # (B*T, n_blocks), the real cheap routing decision
        routing = route_tokens_to_blocks(scores, top_k=top_k, capacity_factor=capacity_factor)
        head_output = dynamic_block_encoder_forward(x_flat, encoder[head], routing, block_size, apply_gate=apply_gate)  # (B*T, N)
        per_head_outputs.append(head_output.reshape(B, T, N))
        routing_results.append(routing)

    x_latent = torch.stack(per_head_outputs, dim=1)  # (B, nh, T, N)
    x_sparse = F.relu(x_latent)

    yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x))  # real, UNMODIFIED oracle Attention

    y_latent = yKV @ model._w(model.encoder_v)  # dense, disclosed scope limit
    y_sparse = F.relu(y_latent)
    xy_sparse = model.drop(x_sparse * y_sparse)

    heads = C.n_head
    width = D * C.mlp_internal_dim_multiplier // heads
    projected = (
        xy_sparse.transpose(1, 2).reshape(B, 1, T, width * heads)
        @ model._w(model.decoder)
    )  # dense, disclosed scope limit
    y = model.ln(projected)
    x_next = model.ln(x + y)
    return x_next, routing_results


def dynamic_block_routing_forward(
    model: BDH,
    idx: torch.Tensor,
    router: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    block_size: int,
    top_k: int,
    capacity_factor: float,
    apply_gate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, list[list[RoutingResult]]]:
    """Real, full BDH forward using dynamic block routing at every one of
    `model.config.n_layer` recurrent iterations -- the SAME `router` and
    SAME model weights are reused every iteration, matching the oracle's
    own real tied-weight convention (`encoder`/`encoder_v`/`decoder` are
    literally one set of parameters reused every recurrent round, not
    per-layer instances -- see `reference/hz0h_bdh_torch.py`'s own module
    docstring). Routing is re-decided fresh each iteration from that
    iteration's own current `x`, not fixed once at the start.

    Returns `(logits, loss, routing_results_per_layer)` --
    `routing_results_per_layer[level]` is that layer iteration's own
    list of per-head `RoutingResult`s (from
    `dynamic_block_routing_layer_forward`), for real per-layer diagnostics
    (e.g. does the real drop rate change with depth).
    """
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    x = model.ln(model.embed(idx).unsqueeze(1))

    routing_results_per_layer: list[list[RoutingResult]] = []
    for _level in range(C.n_layer):
        x, routing_results = dynamic_block_routing_layer_forward(
            model, x, router, block_size=block_size, top_k=top_k,
            capacity_factor=capacity_factor, apply_gate=apply_gate,
        )
        routing_results_per_layer.append(routing_results)

    logits = x.view(B, T, D) @ model.lm_head
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return logits, loss, routing_results_per_layer
