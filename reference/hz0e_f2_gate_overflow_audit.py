"""HZ-0F F2: real-router gate confidence and overflow audit.

F1's oracle audit (`reference/hz0e_f1_oracle_routing_audit.py`) found the
router's per-token SELECTION is not disproportionately worse OOD, and
that unscaled experts individually beat dense on OOD tokens MORE often
than on in-distribution tokens -- ruling against "add abstention to
dense on OOD" as the fix. F1's own writeup named two real, cheaper next
diagnostics its forced/unscaled oracle framing could not test:

1. Does the learned router's GATE CONFIDENCE (the softmax weight applied
   to the chosen expert's output) differ systematically between
   in-distribution and OOD tokens? A confident router scales its
   expert's output near 1.0; an unconfident one scales it down toward
   0 -- if OOD tokens get systematically LOWER gate confidence, the
   REAL (scaled) production path could be worse on OOD even though F1
   found the UNSCALED expert outputs are not worse OOD.
2. Does REAL capacity-based overflow send more OOD tokens to the
   internal, never-independently-curriculum-trained frozen fallback,
   versus in-distribution tokens landing more often within their
   expert's capacity?

This module measures both directly from the real, production
`forward_e6` path (real top-1 routing, real capacity, real gate
scaling) -- not a forced/oracle framing.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from reference.hz0e_e6_integration import forward_e6
from reference.hz0e_moe_contract import MoeConfig, MoeLayerParams, moe_ffn_forward


@dataclass(frozen=True)
class GateOverflowStats:
    overflow_rate: float
    mean_gate_weight_non_overflow: float
    median_gate_weight_non_overflow: float
    gate_weight_p10_non_overflow: float  # 10th percentile -- the "least confident" tail
    token_count: int


def measure_gate_and_overflow(model, moe_params: MoeLayerParams, layer_index: int, tokens: mx.array) -> GateOverflowStats:
    """Real production-path measurement: routes `tokens` through the
    actual trained router (top-1, real capacity, real gate), and reports
    the real overflow rate and the real gate-confidence distribution
    among tokens that did NOT overflow (overflowed tokens have no
    meaningful gate weight -- they use the unscaled fallback)."""
    result = forward_e6(model, tokens, moe_layers={layer_index: moe_params}, enabled=True, target_layers=(layer_index,))
    diag = result.diagnostics[layer_index]
    mx.eval(diag.overflow, diag.gate_weight)
    overflow_np = np.array(diag.overflow)
    gate_weight_np = np.array(diag.gate_weight)

    non_overflow_gates = gate_weight_np[~overflow_np]
    n = int(non_overflow_gates.size)

    return GateOverflowStats(
        overflow_rate=float(overflow_np.mean()),
        mean_gate_weight_non_overflow=float(non_overflow_gates.mean()) if n > 0 else float("nan"),
        median_gate_weight_non_overflow=float(np.median(non_overflow_gates)) if n > 0 else float("nan"),
        gate_weight_p10_non_overflow=float(np.percentile(non_overflow_gates, 10)) if n > 0 else float("nan"),
        token_count=int(overflow_np.size),
    )


def per_token_cross_entropy(logits: mx.array, tokens: mx.array) -> mx.array:
    return nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), tokens[:, 1:])


@dataclass(frozen=True)
class GateForcingResult:
    real_gated_loss: float
    gate_forced_to_one_loss: float
    delta: float  # real - forced_to_one; positive means forcing full-strength expert output HELPS


def per_token_losses_real_and_gate_forced_to_one(model, moe_params: MoeLayerParams, config: MoeConfig, layer_index: int, tokens: mx.array) -> tuple[mx.array, mx.array]:
    """Real, controlled causal test: computes the REAL router's output
    (real top-1 selection, real capacity/overflow, real gate scaling)
    and a second variant where non-overflow tokens' gate scaling is
    forced to 1.0 (full-strength expert output, undoing only the
    softmax-confidence attenuation) -- overflow tokens are already
    unscaled in both variants (the E1 contract never gates the
    fallback), so this isolates the effect of gate CONFIDENCE alone,
    holding expert selection and overflow identical. Both variants
    replay the real suffix blocks independently (the two branches
    diverge starting at the MoE layer's residual add), so both losses
    are real, complete next-token LM losses, not proxies."""
    x = model.embedding(tokens)
    for index in range(layer_index):
        x, _ = model.blocks[index](x, None)
    block = model.blocks[layer_index]
    mixed, _ = block.mixer(block.norm1(x), None)
    residual = x + mixed
    ffn_input = block.norm2(residual)
    moe_out, diag = moe_ffn_forward(ffn_input, moe_params, config)

    b, s, _d = ffn_input.shape
    gate_weight_full = diag.gate_weight.reshape(b, s, 1)
    overflow_full = diag.overflow.reshape(b, s, 1).astype(mx.float32)
    moe_out_gate1 = mx.where(overflow_full > 0.5, moe_out, moe_out / gate_weight_full)

    def _finish(x_after_moe: mx.array) -> mx.array:
        xi = x_after_moe
        for index in range(layer_index + 1, len(model.blocks)):
            xi, _ = model.blocks[index](xi, None)
        return mx.matmul(model.final_norm(xi), model.embedding.weight.T)

    real_logits = _finish(residual + moe_out)
    gate1_logits = _finish(residual + moe_out_gate1)
    return per_token_cross_entropy(real_logits, tokens), per_token_cross_entropy(gate1_logits, tokens)


def gate_forcing_audit(model, moe_params: MoeLayerParams, config: MoeConfig, layer_index: int, tokens: mx.array) -> GateForcingResult:
    real_losses, gate1_losses = per_token_losses_real_and_gate_forced_to_one(model, moe_params, config, layer_index, tokens)
    real_mean = float(mx.mean(real_losses))
    gate1_mean = float(mx.mean(gate1_losses))
    return GateForcingResult(real_gated_loss=real_mean, gate_forced_to_one_loss=gate1_mean, delta=real_mean - gate1_mean)


@dataclass(frozen=True)
class FallbackAuditResult:
    overflow_rate: float
    fallback_token_loss: float  # real router's loss, restricted to overflow (fallback-served) token positions
    dense_baseline_loss_same_positions: float  # fair, independently-trained dense baseline's loss at those SAME positions
    fallback_minus_dense_gap: float  # positive means the internal fallback is worse than the fair dense baseline on the tokens that use it


def fallback_vs_dense_audit(model, moe_params: MoeLayerParams, dense_params: dict[str, mx.array], config: MoeConfig, layer_index: int, tokens: mx.array) -> FallbackAuditResult:
    """Directly tests whether MoE's internal shared fallback -- which
    DOES receive gradient updates during curriculum training, but only
    from whichever tokens happen to overflow to it each step (a sparse,
    incidental training signal), unlike the fair dense baseline (which
    is trained on every token every step) -- underperforms specifically
    on the token positions it actually serves."""
    result = forward_e6(model, tokens, moe_layers={layer_index: moe_params}, enabled=True, target_layers=(layer_index,))
    diag = result.diagnostics[layer_index]
    real_losses = per_token_cross_entropy(result.logits, tokens)

    def dense_ffn(x_flat: mx.array) -> mx.array:
        gate = nn.silu(x_flat @ dense_params["gate_w"].T + dense_params["gate_b"])
        up = x_flat @ dense_params["up_w"].T + dense_params["up_b"]
        return (gate * up) @ dense_params["down_w"].T + dense_params["down_b"]

    x = model.embedding(tokens)
    for index, block in enumerate(model.blocks):
        if index != layer_index:
            x, _ = block(x, None)
        else:
            mixed, _ = block.mixer(block.norm1(x), None)
            residual = x + mixed
            ffn_input = block.norm2(residual)
            b, s, d = ffn_input.shape
            out = dense_ffn(ffn_input.reshape(b * s, d)).reshape(b, s, d)
            x = residual + out
    dense_logits = mx.matmul(model.final_norm(x), model.embedding.weight.T)
    dense_losses = per_token_cross_entropy(dense_logits, tokens)

    # diag.overflow is flat [N] over (batch*seq); real_losses is [batch, seq-1] over next-token pairs.
    # Align by reshaping overflow to [batch, seq] and dropping the last position (no next-token target for it).
    b, s = tokens.shape
    overflow_grid = np.array(diag.overflow).reshape(b, s)[:, :-1]
    real_np = np.array(real_losses)
    dense_np = np.array(dense_losses)

    mask = overflow_grid.astype(bool)
    n = int(mask.sum())
    fallback_loss = float(real_np[mask].mean()) if n > 0 else float("nan")
    dense_at_fallback_positions = float(dense_np[mask].mean()) if n > 0 else float("nan")

    return FallbackAuditResult(
        overflow_rate=float(mask.mean()),
        fallback_token_loss=fallback_loss,
        dense_baseline_loss_same_positions=dense_at_fallback_positions,
        fallback_minus_dense_gap=fallback_loss - dense_at_fallback_positions if n > 0 else float("nan"),
    )
