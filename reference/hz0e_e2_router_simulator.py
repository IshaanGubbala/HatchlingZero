"""HZ-0E E2: isolated router simulator.

Per the plan's own E2 text: "Test routing on code, prose, math, JSON,
tools, mixed domains, imbalance, domain shifts, and noisy inputs.
Measure utilization, balance, overflow, entropy, collapse, and
stability. Exit gate: multiple experts remain active without
collapse."

Real corpus content, not synthetic noise, matching this project's own
established discipline (HZ-0C's C2/C3 both moved from synthetic to
real-corpus construction after finding synthetic confounds). All 5 of
the plan's named domains have a real corpus file in this project --
"tools" is matched to `terminal_and_debugging_validation.jsonl`, the
closest real available domain to "tool use," disclosed here as a
substitution rather than silently assumed identical.

The router at this stage (E1's `init_moe_layer`) is UNTRAINED --
nothing has taught it to differentiate domains yet (that is E8's
"specialization curriculum," a later phase). E2's real, honest job is
therefore MECHANISM stability, not semantic specialization: do multiple
experts stay meaningfully active (no collapse) across real domain
content, domain shifts, batch imbalance, and injected noise, with
correct utilization/entropy/overflow accounting -- not "does routing
already specialize by domain," which no untrained router could show.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from reference.hz0e_moe_contract import MoeConfig, MoeDiagnostics, MoeLayerParams, moe_ffn_forward

DOMAIN_DATA_PATHS = {
    "prose": "data/packed/repro_1024_val.jsonl",
    "code": "data/packed/external/code_validation.jsonl",
    "math": "data/packed/external/mathematical_and_structured_validation.jsonl",
    "json": "data/packed/external/json_and_configuration_validation.jsonl",
    "tools": "data/packed/external/terminal_and_debugging_validation.jsonl",  # closest real match to "tools", see module docstring
}


def collect_real_ffn_input(model, token_ids: mx.array, layer_index: int) -> mx.array:
    """Runs the REAL frozen backbone up through block `layer_index`'s
    OWN mixer and residual, then applies that block's own `norm2` --
    the EXACT real input `reference/hz0a_mlx_model.py::Block.__call__`'s
    own FFN (`self.down(silu(self.gate(normed2)) * self.up(normed2))`)
    receives. This is what a real E6 integration will feed into
    `moe_ffn_forward` in place of the block's own dense FFN call.

    A real bug lived here during development: an earlier version
    applied `block.norm2` to `x` BEFORE running that block's own mixer
    (i.e. to the block's INPUT, not its post-mixer-residual state) --
    caught by `test_collect_real_ffn_input_matches_independent_manual_replay`
    comparing against a from-scratch manual replay of
    `reference/hz0a_mlx_model.py::Block.__call__`'s own control flow,
    not by eyeballing plausible-looking output statistics (the buggy
    and correct versions have nearly identical std, `1.10273` vs
    `1.10264` -- superficially indistinguishable without an exact
    per-element comparison). Every number in
    `docs/restart/hz0e_e2_router_simulator_results.md` was measured
    AFTER this fix, not before. Returns `[batch, seq, dim]`."""
    x = model.embedding(token_ids)
    for index, block in enumerate(model.blocks):
        if index == layer_index:
            mixed, _ = block.mixer(block.norm1(x), None)
            x = x + mixed
            return block.norm2(x)
        x, _ = block(x, None)
    raise ValueError(f"layer_index {layer_index} not found among model.blocks (0..{len(model.blocks) - 1})")


@dataclass(frozen=True)
class RoutingStats:
    """Summary statistics for one `moe_ffn_forward` call's routing
    decisions -- the real, computed quantities E2's exit gate and named
    measurements ("utilization, balance, overflow, entropy, collapse,
    stability") are checked against."""

    utilization: list[float]     # fraction of SERVED (post-overflow) tokens each expert handled, sums to <= 1.0 (remainder is fallback)
    fallback_fraction: float     # fraction of tokens that overflowed to the shared fallback
    entropy_bits: float          # mean per-token entropy of the router's softmax distribution, in bits (log2) -- max possible = log2(num_experts)
    max_expert_share: float      # the single largest expert's utilization share -- the direct "collapse" signal
    num_tokens: int


def compute_routing_stats(diagnostics: MoeDiagnostics, router_probs: mx.array, num_experts: int) -> RoutingStats:
    """`router_probs`: `[N, num_experts]`, the FULL softmax distribution
    (not just the top-1 gate weight) -- needed for a real entropy
    computation, which depends on the whole distribution, not just the
    argmax value."""
    n = int(diagnostics.expert_idx.shape[0])
    counts = diagnostics.expert_counts.tolist()
    fallback = int(diagnostics.fallback_count)
    utilization = [c / n for c in counts]
    fallback_fraction = fallback / n

    log_probs = mx.log2(mx.maximum(router_probs, 1e-12))
    per_token_entropy = -mx.sum(router_probs * log_probs, axis=-1)  # [N], in bits
    mean_entropy = float(mx.mean(per_token_entropy))

    return RoutingStats(
        utilization=utilization, fallback_fraction=fallback_fraction,
        entropy_bits=mean_entropy, max_expert_share=max(utilization) if utilization else 0.0,
        num_tokens=n,
    )


def route_with_stats(x: mx.array, params: MoeLayerParams, config: MoeConfig) -> tuple[mx.array, MoeDiagnostics, RoutingStats]:
    """`moe_ffn_forward` plus the full router softmax distribution
    (recomputed identically to the internal call, since
    `moe_ffn_forward` itself only returns the top-1 gate weight, not
    the full distribution `compute_routing_stats` needs for entropy)."""
    batch, seq, dim = x.shape
    x_flat = x.reshape(batch * seq, dim)
    router_logits = x_flat @ params.router_w.T + params.router_b
    router_probs = mx.softmax(router_logits, axis=-1)
    out, diagnostics = moe_ffn_forward(x, params, config)
    stats = compute_routing_stats(diagnostics, router_probs, config.num_experts)
    return out, diagnostics, stats
