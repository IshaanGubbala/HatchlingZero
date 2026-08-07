"""HZ-0F F4: three-arm fallback isolation experiment.

F3 found a real, growing, OOD-unfavorable fallback-vs-dense differential
at the full training scale: the shared fallback (confirmed NOT frozen --
it receives real gradients from overflow-triggered tokens) increasingly
beats the fair dense baseline in-distribution while staying flat-to-worse
OOD, as if it were quietly absorbing some of the same
specialization-costs-generality tradeoff the experts show explicitly.

This module tests three concrete fallback training policies directly,
holding routing, capacity, overflow rate, initialization, and token
order identical across arms:

1. `"current"` -- the real, existing policy: the fallback trains on
   whatever curriculum-domain gradients overflow sends it (this
   project's shipped behavior, reproduced here for a fair, identically-
   measured comparison point, not re-derived from the earlier docs).
2. `"frozen"` -- warm-started from the pretrained dense FFN (same as
   `"current"`'s own initialization), then genuinely frozen: its
   gradient is zeroed every curriculum step, so it never changes past
   its warm-start value.
3. `"broad_only"` -- the fallback receives NO gradient from curriculum
   steps (zeroed, like `"frozen"`), but DOES receive real gradient
   updates from a separate interleaved general-prose replay batch after
   every curriculum step (router and experts are zeroed-out for that
   replay step, so only the fallback moves). This is NOT step-count-
   matched to `"current"`/`"frozen"` for the fallback specifically (it
   gets one dedicated update per curriculum step, versus only whenever
   overflow happens to route a token to it) -- disclosed directly, not
   hidden, since it is the most direct real implementation of "isolate
   the fallback from curriculum-domain specialization pressure" the
   follow-up experiment asked for.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.optimizers as optim

from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS
from reference.hz0e_e3_routing_objectives import combined_loss, dict_to_params, params_to_dict, supervised_warm_start
from reference.hz0e_e6_integration import TARGET_LAYERS, cross_entropy_loss, forward_e6, init_e6_layers
from reference.hz0e_e8_curriculum import (
    DOMAIN_TO_EXPERT, LAYER, TRAIN_DOMAIN_DATA_PATHS,
    _pack_layers, _unpack_layers,
    balanced_batches, evaluate_dense_per_domain, evaluate_joint_moe_per_domain, evaluate_moe_per_domain,
    imbalanced_batches, load_domain_batches, load_replay_batches, mixed_domain_batches,
    per_domain_mean_loss, run_warm_dense_baseline, run_joint_multilayer_dense_baseline,
)
from reference.hz0e_moe_contract import MoeConfig, MoeLayerParams

FALLBACK_KEYS = {"fallback_gate_w", "fallback_gate_b", "fallback_up_w", "fallback_up_b", "fallback_down_w", "fallback_down_b"}


def train_moe_with_fallback_policy(
    model, config: MoeConfig, *, fallback_policy: str,
    balanced_steps: int = 50, mixed_steps: int = 50, imbalanced_steps: int = 50,
    warm_start_steps: int = 40, learning_rate: float = 1e-5, seed: int = 0,
) -> MoeLayerParams:
    if fallback_policy not in ("current", "frozen", "broad_only"):
        raise ValueError(f"unknown fallback_policy: {fallback_policy}")

    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    e6_layers = init_e6_layers(model, seed=seed)
    warm = supervised_warm_start(
        model, train_domains, DOMAIN_TO_EXPERT, config, layer_index=LAYER,
        steps=warm_start_steps, learning_rate=1e-3, start_params=e6_layers[LAYER], cache_backbone=True,
    )

    stage1 = balanced_batches(train_domains, balanced_steps)
    stage2 = mixed_domain_batches(train_domains, mixed_steps, seed=seed)
    stage3 = imbalanced_batches(train_domains, imbalanced_steps)
    curriculum_batches = stage1 + stage2 + stage3

    params_dict = params_to_dict(warm)
    non_fallback_keys = set(params_dict) - FALLBACK_KEYS
    replay_batches = load_replay_batches(count=len(curriculum_batches)) if fallback_policy == "broad_only" else []

    def loss_fn(p, tokens):
        return combined_loss(p, model, tokens, config, LAYER, {})[0]

    grad_fn = mx.value_and_grad(loss_fn, argnums=0)
    optimizer = optim.Adam(learning_rate=learning_rate)

    for step, tokens in enumerate(curriculum_batches):
        _loss, grads = grad_fn(params_dict, tokens)
        if fallback_policy in ("frozen", "broad_only"):
            for key in FALLBACK_KEYS:
                grads[key] = mx.zeros_like(grads[key])
        params_dict = optimizer.apply_gradients(grads, params_dict)

        if fallback_policy == "broad_only":
            replay_tokens = replay_batches[step % len(replay_batches)]
            _loss2, grads2 = grad_fn(params_dict, replay_tokens)
            for key in non_fallback_keys:
                grads2[key] = mx.zeros_like(grads2[key])
            params_dict = optimizer.apply_gradients(grads2, params_dict)
        mx.eval(params_dict)

    return dict_to_params(params_dict)


@dataclass(frozen=True)
class ArmEvaluation:
    per_domain_moe: dict[str, float]
    per_domain_dense: dict[str, float]
    domain_win_count: int  # out of 5
    moe_general_loss: float
    dense_general_loss: float


def evaluate_arm(model, moe_params: MoeLayerParams, dense_params: dict[str, mx.array], config: MoeConfig) -> ArmEvaluation:
    from reference.hz0e_e8_curriculum import make_warm_dense_loss_fn
    from scripts.hz0c_c3_trigger_simulator import load_real_sequences

    per_domain_moe = evaluate_moe_per_domain(model, moe_params, config, layer_index=LAYER)
    per_domain_dense = evaluate_dense_per_domain(model, dense_params, layer_index=LAYER)
    wins = sum(1 for name in per_domain_moe if per_domain_moe[name] < per_domain_dense[name])

    general_val = [mx.array([s[:64]]) for s in load_real_sequences("data/packed/repro_1024_val.jsonl", 10)]
    from reference.hz0e_e3_routing_objectives import lm_forward_with_moe, params_to_dict as _p2d
    moe_dict = _p2d(moe_params)
    moe_general = sum(float(lm_forward_with_moe(moe_dict, model, tb, config, LAYER)[0]) for tb in general_val) / len(general_val)
    dense_loss_fn = make_warm_dense_loss_fn(model, LAYER)
    dense_general = sum(float(dense_loss_fn(dense_params, tb)) for tb in general_val) / len(general_val)

    return ArmEvaluation(
        per_domain_moe=per_domain_moe, per_domain_dense=per_domain_dense, domain_win_count=wins,
        moe_general_loss=moe_general, dense_general_loss=dense_general,
    )


# --- Full 3-layer joint scope (27, 28, 30 trained together, matching
# E1's real contract and `run_joint_multilayer_curriculum`'s own
# approach) -- validates whether the single-layer `broad_only` finding
# generalizes, or was an artifact of the isolated single-layer setup
# every other diagnostic in this investigation (F1-F4's single-layer
# results, and E4's original single-layer risk finding) used. ---

def train_joint_moe_with_fallback_policy(
    model, config: MoeConfig, *, fallback_policy: str,
    balanced_steps: int = 50, mixed_steps: int = 50, imbalanced_steps: int = 50,
    warm_start_steps: int = 40, learning_rate: float = 1e-5, seed: int = 0,
    target_layers: tuple[int, ...] = TARGET_LAYERS,
) -> dict[int, MoeLayerParams]:
    """Same three fallback-training policies as
    `train_moe_with_fallback_policy`, applied to ALL of `target_layers`
    simultaneously (one shared gradient step per real batch across all
    3 layers, matching `run_joint_multilayer_curriculum`) -- not 3
    independent single-layer runs. Each layer's own fallback fields are
    controlled by the SAME policy, applied per layer."""
    if fallback_policy not in ("current", "frozen", "broad_only"):
        raise ValueError(f"unknown fallback_policy: {fallback_policy}")

    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    e6_layers = init_e6_layers(model, seed=seed, target_layers=target_layers)
    warmed_layers = {
        index: supervised_warm_start(model, train_domains, DOMAIN_TO_EXPERT, config, layer_index=index, steps=warm_start_steps, learning_rate=1e-3, start_params=e6_layers[index])
        for index in target_layers
    }
    flat_params = _pack_layers(warmed_layers)
    fallback_flat_keys = {f"{index}.{key}" for index in target_layers for key in FALLBACK_KEYS}
    non_fallback_flat_keys = set(flat_params) - fallback_flat_keys

    def loss_fn(flat_p: dict[str, mx.array], tokens: mx.array) -> mx.array:
        layers = _unpack_layers(flat_p, target_layers)
        result = forward_e6(model, tokens, moe_layers=layers, enabled=True, target_layers=target_layers)
        return cross_entropy_loss(result.logits, tokens)

    grad_fn = mx.value_and_grad(loss_fn, argnums=0)
    optimizer = optim.Adam(learning_rate=learning_rate)

    stage1 = balanced_batches(train_domains, balanced_steps)
    stage2 = mixed_domain_batches(train_domains, mixed_steps, seed=seed)
    stage3 = imbalanced_batches(train_domains, imbalanced_steps)
    curriculum_batches = stage1 + stage2 + stage3
    replay_batches = load_replay_batches(count=len(curriculum_batches)) if fallback_policy == "broad_only" else []

    for step, tokens in enumerate(curriculum_batches):
        _loss, grads = grad_fn(flat_params, tokens)
        if fallback_policy in ("frozen", "broad_only"):
            for key in fallback_flat_keys:
                grads[key] = mx.zeros_like(grads[key])
        flat_params = optimizer.apply_gradients(grads, flat_params)

        if fallback_policy == "broad_only":
            replay_tokens = replay_batches[step % len(replay_batches)]
            _loss2, grads2 = grad_fn(flat_params, replay_tokens)
            for key in non_fallback_flat_keys:
                grads2[key] = mx.zeros_like(grads2[key])
            flat_params = optimizer.apply_gradients(grads2, flat_params)
        mx.eval(flat_params)

    return _unpack_layers(flat_params, target_layers)


def train_joint_dense_baseline_with_general_eval(
    model, *, d_ff: int = 577, target_layers: tuple[int, ...] = TARGET_LAYERS,
    balanced_steps: int = 50, mixed_steps: int = 50, imbalanced_steps: int = 50,
    learning_rate: float = 1e-5, seed: int = 0,
) -> tuple[dict[str, float], float]:
    """Same fair, jointly-trained dense baseline as
    `run_joint_multilayer_dense_baseline`, but ALSO returns the general/
    OOD held-out loss (that function only evaluates per-domain) -- a
    small, self-contained duplicate rather than modifying the existing,
    already-tested function's return contract."""
    import mlx.nn as nn

    from scripts.hz0c_c3_trigger_simulator import load_real_sequences

    train_domains = load_domain_batches(TRAIN_DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=0)
    general_val = [mx.array([s[:64]]) for s in load_real_sequences("data/packed/repro_1024_val.jsonl", 10)]

    def init_layers() -> dict[str, mx.array]:
        flat: dict[str, mx.array] = {}
        for index in target_layers:
            block = model.blocks[index]
            flat[f"{index}.gate_w"] = block.gate.weight[:d_ff]
            flat[f"{index}.gate_b"] = block.gate.bias[:d_ff]
            flat[f"{index}.up_w"] = block.up.weight[:d_ff]
            flat[f"{index}.up_b"] = block.up.bias[:d_ff]
            flat[f"{index}.down_w"] = block.down.weight[:, :d_ff] * 5.0
            flat[f"{index}.down_b"] = block.down.bias
        return flat

    def loss_fn(params: dict[str, mx.array], tokens: mx.array) -> mx.array:
        x = model.embedding(tokens)
        for index, block in enumerate(model.blocks):
            if index not in target_layers:
                x, _ = block(x, None)
            else:
                mixed, _ = block.mixer(block.norm1(x), None)
                residual = x + mixed
                ffn_input = block.norm2(residual)
                mlp = (nn.silu(ffn_input @ params[f"{index}.gate_w"].T + params[f"{index}.gate_b"]) * (ffn_input @ params[f"{index}.up_w"].T + params[f"{index}.up_b"])) @ params[f"{index}.down_w"].T + params[f"{index}.down_b"]
                x = residual + mlp
        logits = mx.matmul(model.final_norm(x), model.embedding.weight.T)
        return cross_entropy_loss(logits, tokens)

    params = init_layers()
    stage1 = balanced_batches(train_domains, balanced_steps)
    stage2 = mixed_domain_batches(train_domains, mixed_steps, seed=seed)
    stage3 = imbalanced_batches(train_domains, imbalanced_steps)
    grad_fn = mx.value_and_grad(loss_fn, argnums=0)
    optimizer = optim.Adam(learning_rate=learning_rate)
    for tokens in stage1 + stage2 + stage3:
        _loss, grads = grad_fn(params, tokens)
        params = optimizer.apply_gradients(grads, params)
        mx.eval(params)

    held_out_domains = load_domain_batches(DOMAIN_DATA_PATHS, count=8, seq_len=64, offset=1)
    per_domain = {name: float(loss_fn(params, tb)) for name, tb in held_out_domains.items()}
    general_losses = [float(loss_fn(params, tb)) for tb in general_val]
    general = sum(general_losses) / len(general_losses)
    return per_domain, general


@dataclass(frozen=True)
class JointArmEvaluation:
    per_domain_moe: dict[str, float]
    per_domain_dense: dict[str, float]
    domain_win_count: int
    moe_general_loss: float
    dense_general_loss: float


def evaluate_joint_arm(model, trained_layers: dict[int, MoeLayerParams], dense_per_domain: dict[str, float], dense_general_loss: float, target_layers: tuple[int, ...] = TARGET_LAYERS) -> JointArmEvaluation:
    from scripts.hz0c_c3_trigger_simulator import load_real_sequences

    per_domain_moe = evaluate_joint_moe_per_domain(model, trained_layers, target_layers)
    wins = sum(1 for name in per_domain_moe if per_domain_moe[name] < dense_per_domain[name])

    general_val = [mx.array([s[:64]]) for s in load_real_sequences("data/packed/repro_1024_val.jsonl", 10)]
    moe_general_losses = [float(cross_entropy_loss(forward_e6(model, tb, moe_layers=trained_layers, enabled=True, target_layers=target_layers).logits, tb)) for tb in general_val]
    moe_general = sum(moe_general_losses) / len(moe_general_losses)

    return JointArmEvaluation(
        per_domain_moe=per_domain_moe, per_domain_dense=dense_per_domain, domain_win_count=wins,
        moe_general_loss=moe_general, dense_general_loss=dense_general_loss,
    )
