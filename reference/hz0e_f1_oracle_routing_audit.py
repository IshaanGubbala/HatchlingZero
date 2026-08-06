"""HZ-0F F1: oracle routing audit.

HZ-0E's E10 evaluation closed with a real, disclosed, structural
tradeoff: MoE beats fair dense on per-domain (in-distribution) quality
in 6/6 trials, but loses on general (out-of-distribution) quality. The
natural next question -- proposed directly, not invented here -- is
whether that OOD loss is a ROUTING problem (the learned top-1 router
makes bad choices on OOD tokens, fixable by a smarter/more cautious
router) or an ARCHITECTURE problem (no available route -- not even the
best one -- is actually good on OOD tokens, fixable only by changing
what routes exist, e.g. a shared dense trunk).

This module answers that by computing a REAL oracle: for each held-out
token, the loss under every one of the 4 experts FORCED (bypassing the
learned router and gate entirely) plus the loss under a fair,
independently-trained dense baseline, and comparing the per-token MINIMUM
across those 5 candidates against the actual learned router's real loss.

Design choices, disclosed directly:

- This is a per-token oracle over GLOBALLY-forced runs, not a true
  combinatorial per-token oracle (that would require one forward pass
  per token per candidate, computationally infeasible). Each candidate
  run forces EVERY token in the batch through the same fixed route, then
  per-token losses are read off positionally and the minimum taken
  across the 5 runs. Because attention/recurrence mix information across
  token positions in the layers AFTER the MoE FFN, a token's measured
  loss under "expert 2 forced" is not perfectly isolated from what OTHER
  tokens in the same sequence were forced to that run -- this is a real,
  disclosed approximation, not a perfectly clean per-token oracle. It is
  the same approximation the proposal that motivated this module itself
  assumes (`for each token: evaluate expert 0 loss ... target = best
  expert`), made explicit here rather than silently accepted.
- Forced-expert output is UNSCALED (no router gate weighting applied) --
  there is no "confidence" to apply when the choice is forced, so this
  measures the expert's raw transformation quality at that token, not
  what the trained gate would have scaled it to.
- The "dense" / abstain candidate uses the SAME fairly-trained,
  matched-active dense baseline (`run_warm_dense_baseline`) every other
  comparison in this project uses -- not MoE's own internal (frozen,
  never independently curriculum-trained) fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from reference.hz0e_e6_integration import forward_e6
from reference.hz0e_moe_contract import MoeConfig, MoeLayerParams


def per_token_cross_entropy(logits: mx.array, tokens: mx.array) -> mx.array:
    """`[batch, seq]` tokens -> `[batch, seq-1]` per-position CE loss,
    same next-token-prediction convention as
    `reference/hz0e_e6_integration.py::cross_entropy_loss`, without the
    final reduction to a scalar."""
    return nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), tokens[:, 1:])


def _run_prefix_suffix(model, layer_index: int, tokens: mx.array, ffn_fn):
    """Runs every block except `layer_index` through the real model path
    unchanged; at `layer_index`, replaces the dense FFN with `ffn_fn`
    (called on the flattened `[batch*seq, dim]` post-norm2 input,
    returning the same shape). Shared by every forced-route candidate in
    this module so they all see byte-identical prefix/suffix computation
    -- only the one FFN call differs."""
    x = model.embedding(tokens)
    for index, block in enumerate(model.blocks):
        if index != layer_index:
            x, _ = block(x, None)
        else:
            mixed, _ = block.mixer(block.norm1(x), None)
            residual = x + mixed
            ffn_input = block.norm2(residual)
            b, s, d = ffn_input.shape
            out = ffn_fn(ffn_input.reshape(b * s, d)).reshape(b, s, d)
            x = residual + out
    return mx.matmul(model.final_norm(x), model.embedding.weight.T)


def per_token_losses_forced_expert(model, moe_params: MoeLayerParams, layer_index: int, tokens: mx.array, expert_index: int) -> mx.array:
    """Every token forced through `expert_index`'s SwiGLU, unscaled (no
    router gate applied)."""
    def ffn_fn(x_flat: mx.array) -> mx.array:
        gate = nn.silu(x_flat @ moe_params.expert_gate_w[expert_index].T + moe_params.expert_gate_b[expert_index])
        up = x_flat @ moe_params.expert_up_w[expert_index].T + moe_params.expert_up_b[expert_index]
        hidden = gate * up
        return hidden @ moe_params.expert_down_w[expert_index].T + moe_params.expert_down_b[expert_index]

    logits = _run_prefix_suffix(model, layer_index, tokens, ffn_fn)
    return per_token_cross_entropy(logits, tokens)


def per_token_losses_dense(model, dense_params: dict[str, mx.array], layer_index: int, tokens: mx.array) -> mx.array:
    """Every token routed to the fair, independently-trained dense
    baseline (`warm_dense_init`/`run_warm_dense_baseline`'s own
    parameter dict layout) -- the real "abstain" candidate."""
    def ffn_fn(x_flat: mx.array) -> mx.array:
        gate = nn.silu(x_flat @ dense_params["gate_w"].T + dense_params["gate_b"])
        up = x_flat @ dense_params["up_w"].T + dense_params["up_b"]
        hidden = gate * up
        return hidden @ dense_params["down_w"].T + dense_params["down_b"]

    logits = _run_prefix_suffix(model, layer_index, tokens, ffn_fn)
    return per_token_cross_entropy(logits, tokens)


def per_token_losses_actual_router(model, moe_params: MoeLayerParams, layer_index: int, tokens: mx.array) -> mx.array:
    """The real, currently-shipped learned-router path -- top-1 routing,
    real capacity/overflow, real gate scaling, via
    `reference/hz0e_e6_integration.py::forward_e6`."""
    result = forward_e6(model, tokens, moe_layers={layer_index: moe_params}, enabled=True, target_layers=(layer_index,))
    return per_token_cross_entropy(result.logits, tokens)


@dataclass(frozen=True)
class OracleAuditResult:
    actual_router_mean_loss: float
    oracle_mean_loss: float
    oracle_gap: float  # actual - oracle; positive means the real router is worse than the oracle
    candidate_win_rate: dict[str, float]  # fraction of tokens where each candidate achieves the per-token minimum
    candidate_mean_loss: dict[str, float]


def oracle_routing_audit(model, moe_params: MoeLayerParams, dense_params: dict[str, mx.array], config: MoeConfig, layer_index: int, tokens: mx.array) -> OracleAuditResult:
    """The real audit: computes the actual router's per-token loss and
    every forced-route candidate's per-token loss on the SAME held-out
    batch, then compares the actual router's mean loss against the
    per-token oracle minimum across all candidates.

    A positive `oracle_gap` means there was real, available headroom the
    router failed to reach -- evidence for "fix the router" (candidates
    #2-4 in the proposal this module tests). A near-zero `oracle_gap`
    means the router is already near-optimal given the available
    routes -- evidence that the routes THEMSELVES are insufficient
    (candidates #1, #5, #7 -- architecture changes), not the routing
    policy choosing among them.
    """
    actual = per_token_losses_actual_router(model, moe_params, layer_index, tokens)

    candidates: dict[str, mx.array] = {
        f"expert_{j}": per_token_losses_forced_expert(model, moe_params, layer_index, tokens, j)
        for j in range(config.num_experts)
    }
    candidates["dense"] = per_token_losses_dense(model, dense_params, layer_index, tokens)

    stacked = mx.stack(list(candidates.values()))  # [num_candidates, tokens]
    oracle_per_token = mx.min(stacked, axis=0)
    winners = mx.argmin(stacked, axis=0)
    names = list(candidates.keys())
    total = float(winners.size)
    win_rate = {name: float(mx.sum(winners == index)) / total for index, name in enumerate(names)}
    mean_loss = {name: float(mx.mean(values)) for name, values in candidates.items()}

    actual_mean = float(mx.mean(actual))
    oracle_mean = float(mx.mean(oracle_per_token))
    return OracleAuditResult(
        actual_router_mean_loss=actual_mean,
        oracle_mean_loss=oracle_mean,
        oracle_gap=actual_mean - oracle_mean,
        candidate_win_rate=win_rate,
        candidate_mean_loss=mean_loss,
    )
