"""HZ-0F: counterfactual-utility router warm-start.

E3's `supervised_warm_start` trains the router via per-BATCH cross-entropy
against a coarse `target_expert` domain-to-expert label (every token in a
"prose" batch gets the SAME label, regardless of whether that token would
actually be better served by a different expert). F1's oracle routing
audit (`reference/hz0e_f1_oracle_routing_audit.py`) already computes a
much finer-grained signal -- per-TOKEN loss under each forced expert --
and F3 confirmed real, regime-balanced headroom exists between the
router's real choices and that oracle. This module turns that oracle
into an actual warm-start supervision signal: instead of "this whole
batch is prose, use expert 0," each token gets its OWN best-expert label,
computed directly from real per-token loss, not a domain heuristic.

Labels are restricted to the router's real decision space (argmin over
the `num_experts` forced-expert candidates only, not including the dense
fallback) -- the router chooses among experts; capacity-based overflow to
the fallback is a separate, later mechanism, matching every other warm-
start/training function in this project's own convention.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from reference.hz0e_e3_routing_objectives import _moe_prefix, dict_to_params, init_moe_layer, params_to_dict
from reference.hz0e_f1_oracle_routing_audit import per_token_losses_forced_expert
from reference.hz0e_moe_contract import MoeConfig, MoeLayerParams


def compute_counterfactual_labels(model, moe_params: MoeLayerParams, config: MoeConfig, layer_index: int, tokens: mx.array) -> mx.array:
    """Real per-token best-expert label: `[batch, seq-1]` int array,
    `argmin` over the `num_experts` forced-expert per-token losses
    (computed with the CURRENT, fixed expert weights -- experts do not
    move during router-only warm-start, matching
    `supervised_warm_start`'s own contract)."""
    losses = mx.stack([
        per_token_losses_forced_expert(model, moe_params, layer_index, tokens, j)
        for j in range(config.num_experts)
    ])  # [num_experts, batch, seq-1]
    return mx.argmin(losses, axis=0).astype(mx.int32)


def counterfactual_warm_start(
    model, domain_batches: dict[str, mx.array], config: MoeConfig, *,
    layer_index: int = 27, steps: int = 20, learning_rate: float = 3e-3,
    init_seed: int = 0, start_params: MoeLayerParams | None = None,
) -> MoeLayerParams:
    """Same role and contract as `supervised_warm_start` (router-only
    training, experts/fallback untouched, `steps` real gradient steps)
    but supervised by real per-token counterfactual-utility labels
    instead of a coarse per-batch domain label."""
    initial = start_params or init_moe_layer(MoeConfig(
        dim=config.dim, dense_d_ff=config.dense_d_ff, num_experts=config.num_experts,
        expert_d_ff=config.expert_d_ff, capacity_factor=config.capacity_factor, init_seed=init_seed,
    ))
    params_dict = params_to_dict(initial)
    optimizer = optim.Adam(learning_rate=learning_rate)
    domains = list(domain_batches.keys())

    # Experts are fixed throughout router-only warm-start, so the
    # counterfactual labels are constant across steps -- compute once per
    # domain batch, not on every step.
    labels_by_domain = {}
    for domain, tokens in domain_batches.items():
        labels = compute_counterfactual_labels(model, initial, config, layer_index, tokens)
        mx.eval(labels)
        labels_by_domain[domain] = labels

    def router_loss_fn(p, tokens, labels):
        _x, ffn_input = _moe_prefix(model, tokens, layer_index)
        batch, seq, dim = ffn_input.shape
        ffn_trunc = ffn_input[:, :-1, :].reshape(batch * (seq - 1), dim)
        router_logits = ffn_trunc @ p["router_w"].T + p["router_b"]
        return mx.mean(nn.losses.cross_entropy(router_logits, labels.reshape(-1)))

    grad_fn = mx.value_and_grad(router_loss_fn, argnums=0)
    for step in range(steps):
        domain = domains[step % len(domains)]
        tokens = domain_batches[domain]
        labels = labels_by_domain[domain]
        _loss, grads = grad_fn(params_dict, tokens, labels)
        router_only_grads = {"router_w": grads["router_w"], "router_b": grads["router_b"]}
        router_only_params = {"router_w": params_dict["router_w"], "router_b": params_dict["router_b"]}
        updated = optimizer.apply_gradients(router_only_grads, router_only_params)
        params_dict["router_w"] = updated["router_w"]
        params_dict["router_b"] = updated["router_b"]
        mx.eval(params_dict)
    return dict_to_params(params_dict)
