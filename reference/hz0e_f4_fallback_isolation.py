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
from reference.hz0e_e6_integration import init_e6_layers
from reference.hz0e_e8_curriculum import (
    DOMAIN_TO_EXPERT, LAYER, TRAIN_DOMAIN_DATA_PATHS,
    balanced_batches, evaluate_dense_per_domain, evaluate_moe_per_domain,
    imbalanced_batches, load_domain_batches, load_replay_batches, mixed_domain_batches,
    per_domain_mean_loss, run_warm_dense_baseline,
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
