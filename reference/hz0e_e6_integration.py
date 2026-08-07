"""HZ-0E E6: frozen HZ-0A backbone with selected upper MoE FFNs."""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from reference.hz0e_moe_contract import (
    MoeConfig, MoeDiagnostics, MoeLayerParams, init_moe_layer, moe_ffn_forward,
)


TARGET_LAYERS = (27, 28, 30)


@dataclass(frozen=True)
class E6Result:
    logits: mx.array
    states: list
    diagnostics: dict[int, MoeDiagnostics]


def _copy_fallback(block, params: MoeLayerParams) -> MoeLayerParams:
    """Warm-start the shared overflow fallback from the pretrained FFN."""
    return MoeLayerParams(
        router_w=params.router_w, router_b=params.router_b,
        expert_gate_w=params.expert_gate_w, expert_gate_b=params.expert_gate_b,
        expert_up_w=params.expert_up_w, expert_up_b=params.expert_up_b,
        expert_down_w=params.expert_down_w, expert_down_b=params.expert_down_b,
        fallback_gate_w=block.gate.weight, fallback_gate_b=block.gate.bias,
        fallback_up_w=block.up.weight, fallback_up_b=block.up.bias,
        fallback_down_w=block.down.weight, fallback_down_b=block.down.bias,
    )


def init_e6_layers(model, *, seed: int = 0, target_layers: tuple[int, ...] = TARGET_LAYERS, warm_start_experts: bool = True, expert_output_scale: float = 6.0, expert_output_scales: dict[int, float] | None = None) -> dict[int, MoeLayerParams]:
    """Create external MoE weights while keeping the HZ-0A model frozen.

    Warm-starting copies a stable slice of the pretrained dense FFN into
    every expert. This avoids a random-logit shock on the first integrated
    forward. The output projection is scaled to compensate for the
    top-1 softmax gate attenuation; E8 can train the experts and router
    afterward.
    """
    layers = {}
    scales = expert_output_scales if expert_output_scales is not None else {27: 5.0, 28: 5.0, 30: 7.0}
    for offset, layer_index in enumerate(target_layers):
        config = MoeConfig(dim=model.dim, init_seed=seed + offset)
        params = _copy_fallback(model.blocks[layer_index], init_moe_layer(config))
        if warm_start_experts:
            block = model.blocks[layer_index]
            width = config.expert_d_ff
            params = MoeLayerParams(
                router_w=params.router_w, router_b=params.router_b,
                expert_gate_w=mx.broadcast_to(block.gate.weight[:width][None], params.expert_gate_w.shape),
                expert_gate_b=mx.broadcast_to(block.gate.bias[:width][None], params.expert_gate_b.shape),
                expert_up_w=mx.broadcast_to(block.up.weight[:width][None], params.expert_up_w.shape),
                expert_up_b=mx.broadcast_to(block.up.bias[:width][None], params.expert_up_b.shape),
                expert_down_w=mx.broadcast_to(block.down.weight[:, :width][None], params.expert_down_w.shape) * scales.get(layer_index, expert_output_scale),
                expert_down_b=mx.broadcast_to(block.down.bias[None], params.expert_down_b.shape),
                fallback_gate_w=params.fallback_gate_w, fallback_gate_b=params.fallback_gate_b,
                fallback_up_w=params.fallback_up_w, fallback_up_b=params.fallback_up_b,
                fallback_down_w=params.fallback_down_w, fallback_down_b=params.fallback_down_b,
            )
        layers[layer_index] = params
    return layers


def load_e6_layers(model, path, *, target_layers: tuple[int, ...] = TARGET_LAYERS) -> dict[int, MoeLayerParams]:
    """Load an exported calibration artifact and validate its tensor surface."""
    arrays = mx.load(str(path))
    expected = {f"layer_{index}_{name}" for index in target_layers for name in MoeLayerParams.__dataclass_fields__}
    if set(arrays) != expected:
        missing, extra = sorted(expected - set(arrays)), sorted(set(arrays) - expected)
        raise ValueError(f"invalid E6 artifact; missing={missing}, extra={extra}")
    layers = {}
    for index in target_layers:
        values = {name: arrays[f"layer_{index}_{name}"] for name in MoeLayerParams.__dataclass_fields__}
        config = MoeConfig(dim=model.dim)
        expected = {
            "router_w": (config.num_experts, model.dim), "router_b": (config.num_experts,),
            "expert_gate_w": (config.num_experts, config.expert_d_ff, model.dim),
            "expert_gate_b": (config.num_experts, config.expert_d_ff),
            "expert_up_w": (config.num_experts, config.expert_d_ff, model.dim),
            "expert_up_b": (config.num_experts, config.expert_d_ff),
            "expert_down_w": (config.num_experts, model.dim, config.expert_d_ff),
            "expert_down_b": (config.num_experts, model.dim),
            "fallback_gate_w": (config.dense_d_ff, model.dim), "fallback_gate_b": (config.dense_d_ff,),
            "fallback_up_w": (config.dense_d_ff, model.dim), "fallback_up_b": (config.dense_d_ff,),
            "fallback_down_w": (model.dim, config.dense_d_ff), "fallback_down_b": (model.dim,),
        }
        for name, value in values.items():
            if tuple(value.shape) != expected[name]:
                raise ValueError(f"invalid E6 tensor shape for {name}: expected {expected[name]}, got {tuple(value.shape)}")
            if not bool(mx.all(mx.isfinite(value))):
                raise ValueError(f"invalid E6 tensor values for {name}: non-finite data")
        layers[index] = MoeLayerParams(**values)
    return layers


def forward_e6(model, token_ids: mx.array, states=None, *, moe_layers=None, enabled: bool = True, target_layers: tuple[int, ...] = TARGET_LAYERS) -> E6Result:
    """Run the frozen backbone, replacing only selected FFNs when enabled.

    Mixer calls and state objects are deliberately unchanged from
    ``HZ0AMlxModel.__call__``. MoE is applied after each selected block's
    mixer+residual and RMSNorm, exactly where the original dense FFN ran.
    """
    x = model.embedding(token_ids)
    if states is None:
        states = [None] * len(model.blocks)
    next_states = []
    diagnostics: dict[int, MoeDiagnostics] = {}
    for index, (block, state) in enumerate(zip(model.blocks, states)):
        if not enabled or index not in target_layers:
            x, next_state = block(x, state)
        else:
            mixed, next_state = block.mixer(block.norm1(x), state)
            residual = x + mixed
            ffn_input = block.norm2(residual)
            config = MoeConfig(dim=model.dim)
            moe_out, diag = moe_ffn_forward(ffn_input, moe_layers[index], config)
            x = residual + moe_out
            diagnostics[index] = diag
        next_states.append(next_state)
    logits = mx.matmul(model.final_norm(x), model.embedding.weight.T)
    return E6Result(logits=logits, states=next_states, diagnostics=diagnostics)


def cross_entropy_loss(logits: mx.array, token_ids: mx.array) -> mx.array:
    return mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), token_ids[:, 1:]))
