"""Real correctness tests for
reference/hz0h_bdh_dynamic_block_routing_layer_torch.py: does the
dynamic-routing layer reduce EXACTLY to the real oracle in the limiting
case (every block served, no drops), and does it stay correct AND
gradient-clean when real drops actually occur? CPU-testable throughout."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_dynamic_block_routing_layer_torch import (
    dynamic_block_routing_layer_forward,
    init_dynamic_block_router,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _oracle_one_layer(model: BDH, x: torch.Tensor) -> torch.Tensor:
    """Real, direct transcription of one iteration of BDH.forward's own
    loop body (reference/hz0h_bdh_torch.py lines 176-198) -- used as the
    real, load-bearing comparison target, not re-derived math."""
    C = model.config
    B, _, T, D = x.shape
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    x_latent = x @ model._w(model.encoder)
    x_sparse = F.relu(x_latent)
    yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
    yKV = model.ln(yKV)
    y_latent = yKV @ model._w(model.encoder_v)
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse
    xy_sparse = model.drop(xy_sparse)
    yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
    y = model.ln(yMLP)
    return model.ln(x + y)


def test_dynamic_routing_layer_matches_oracle_exactly_when_every_block_is_served():
    """Load-bearing wiring-correctness test: top_k=n_blocks (every block
    nominally selected by every token) + a huge capacity_factor (no real
    drops possible) + apply_gate=False must reduce EXACTLY to the
    oracle's own real per-layer computation, regardless of the router's
    own (here, effectively irrelevant since everything is selected)
    scores. apply_gate=False is required here: WITH the real,
    differentiable softmax gate (the default, needed for the router to
    be trainable), the result does NOT reduce exactly to the oracle even
    when everything is selected -- a real, disclosed, expected property
    of gated MoE-style routing, not a bug (see
    dynamic_block_encoder_forward's own docstring). This test validates
    the underlying gather/scatter WIRING is exact; the gated path's own
    correctness is validated separately below, against a matching gated
    slow reference, and by the dedicated gradient test."""
    torch.manual_seed(0)
    config = BDHConfig(n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, n_layer=1, vocab_size=64, dropout=0.0)
    model = BDH(config).eval()
    block_size = 4  # N = 32*8/4 = 64 per head -> n_blocks = 16
    n_blocks = (config.n_embd * config.mlp_internal_dim_multiplier // config.n_head) // block_size

    x = torch.randn(2, 1, 9, config.n_embd)
    router = init_dynamic_block_router(config.n_head, config.n_embd, n_blocks, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        oracle_out = _oracle_one_layer(model, x)
        routed_out, routing_results = dynamic_block_routing_layer_forward(
            model, x, router, block_size=block_size, top_k=n_blocks, capacity_factor=100.0, apply_gate=False,
        )

    for routing in routing_results:
        assert routing.tokens_dropped == 0, "test setup must guarantee zero drops for this to be a valid exactness check"

    max_diff = (oracle_out - routed_out).abs().max().item()
    assert torch.allclose(oracle_out, routed_out, atol=1e-4, rtol=1e-4), f"max diff {max_diff}"


def test_dynamic_routing_layer_cross_validates_against_a_slow_masked_reference_under_real_drops():
    """Real cross-validation for the case that actually matters (real
    routing/dropping active, real gate applied -- the default,
    trainable-router configuration): builds a slow, obviously-correct
    reference that computes the FULL dense encoder projection then
    multiplies by the REAL per-(token,block) gate weight where served,
    zero where dropped-for-capacity or never selected -- using the SAME
    routing decision, so this isolates "did the fast path implement the
    routing+gating decision correctly" from "is the routing decision
    itself reasonable"."""
    torch.manual_seed(2)
    config = BDHConfig(n_embd=24, n_head=3, mlp_internal_dim_multiplier=8, n_layer=1, vocab_size=64, dropout=0.0)
    model = BDH(config).eval()
    block_size = 4
    n_blocks = (config.n_embd * config.mlp_internal_dim_multiplier // config.n_head) // block_size

    x = torch.randn(3, 1, 11, config.n_embd)
    router = init_dynamic_block_router(config.n_head, config.n_embd, n_blocks, generator=torch.Generator().manual_seed(3))

    with torch.no_grad():
        routed_out, routing_results = dynamic_block_routing_layer_forward(
            model, x, router, block_size=block_size, top_k=2, capacity_factor=0.5,  # small -> real, guaranteed drops
        )  # apply_gate defaults to True -- the real, trainable-router configuration

        total_dropped = sum(r.tokens_dropped for r in routing_results)
        assert total_dropped > 0, "test setup must produce real drops to be a meaningful cross-validation"

        # Slow, obviously-correct reference: compute the real full dense
        # x_sparse, then apply the real per-(token,block) gate weight
        # where served, zero everywhere else.
        B, _, T, D = x.shape
        nh = config.n_head
        N = D * config.mlp_internal_dim_multiplier // nh
        x_latent_dense = x @ model._w(model.encoder)  # (B, nh, T, N), real full dense projection
        mask = torch.zeros_like(x_latent_dense)
        for head, routing in enumerate(routing_results):
            served_mask_flat = torch.zeros(B * T, N)
            for block in range(n_blocks):
                token_ids = routing.block_token_indices[block]
                valid_positions = token_ids >= 0
                valid = token_ids[valid_positions]
                gate_values = routing.block_gate_weights[block][valid_positions]
                if valid.numel():
                    served_mask_flat[valid, block * block_size:(block + 1) * block_size] = gate_values.unsqueeze(-1)
            mask[:, head, :, :] = served_mask_flat.reshape(B, T, N)
        x_latent_reference = x_latent_dense * mask
        x_sparse_reference = F.relu(x_latent_reference)
        yKV_reference = model.ln(model.attn(Q=x_sparse_reference, K=x_sparse_reference, V=x))
        y_latent_reference = yKV_reference @ model._w(model.encoder_v)
        y_sparse_reference = F.relu(y_latent_reference)
        xy_reference = model.drop(x_sparse_reference * y_sparse_reference)
        yMLP_reference = xy_reference.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        reference_out = model.ln(x + model.ln(yMLP_reference))

    max_diff = (reference_out - routed_out).abs().max().item()
    assert torch.allclose(reference_out, routed_out, atol=1e-4, rtol=1e-4), (
        f"fast gather/scatter path disagrees with the slow masked-dense reference under real drops: max diff {max_diff}"
    )


def test_dynamic_routing_layer_gradients_are_finite_and_reach_every_real_parameter():
    torch.manual_seed(4)
    config = BDHConfig(n_embd=20, n_head=2, mlp_internal_dim_multiplier=8, n_layer=1, vocab_size=64, dropout=0.0)
    model = BDH(config)
    block_size = 4
    n_blocks = (config.n_embd * config.mlp_internal_dim_multiplier // config.n_head) // block_size

    x = torch.randn(2, 1, 7, config.n_embd)
    router = init_dynamic_block_router(config.n_head, config.n_embd, n_blocks, generator=torch.Generator().manual_seed(5))
    router.requires_grad_(True)

    x_next, routing_results = dynamic_block_routing_layer_forward(
        model, x, router, block_size=block_size, top_k=2, capacity_factor=0.6,
    )
    assert sum(r.tokens_dropped for r in routing_results) > 0, "test setup must exercise real drops"
    assert torch.isfinite(x_next).all()

    x_next.sum().backward()
    assert router.grad is not None and torch.isfinite(router.grad).all()
    assert model.encoder.grad is not None and torch.isfinite(model.encoder.grad).all()
    assert model.encoder_v.grad is not None and torch.isfinite(model.encoder_v.grad).all()
    assert model.decoder.grad is not None and torch.isfinite(model.decoder.grad).all()
