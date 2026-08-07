"""Balanced E8 specialization probe for one frozen-backbone MoE layer."""
from __future__ import annotations

import argparse
import json

import mlx.core as mx
import mlx.utils

from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS
from reference.hz0e_e3_routing_objectives import (
    lm_forward_with_moe, params_to_dict, supervised_warm_start, train_moe_layer,
)
from reference.hz0e_e6_integration import init_e6_layers
from reference.hz0e_moe_contract import MoeConfig, MoeLayerParams
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c3_trigger_simulator import load_real_sequences


DOMAIN_TO_EXPERT = {"prose": 0, "code": 1, "math": 2, "json": 3, "tools": 0}


def _load_domain(path: str, count: int = 2, width: int = 256) -> mx.array:
    rows = load_real_sequences(path, count)
    return mx.array([row[:width] for row in rows], dtype=mx.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hard-domain-repeat", type=int, default=1, help="extra repeats for code/math/tools in the balanced schedule")
    args = parser.parse_args()
    model, _ = load_frozen_model()
    config = MoeConfig(dim=model.dim)
    domains = {name: _load_domain(path) for name, path in DOMAIN_DATA_PATHS.items()}
    start = init_e6_layers(model, seed=args.seed)[27]
    warm = supervised_warm_start(
        model, domains, DOMAIN_TO_EXPERT, config, layer_index=27,
        steps=40, learning_rate=1e-3, start_params=start,
    )
    repeats = max(1, args.steps // len(domains))
    hard_domains = {"code", "math", "tools"}
    schedule = [
        domains[name] for name in DOMAIN_DATA_PATHS
        for _ in range(repeats * (args.hard_domain_repeat if name in hard_domains else 1))
    ]
    train_batches = [schedule[index % len(schedule)] for index in range(args.steps)]
    trained, history = train_moe_layer(
        model, train_batches, config, layer_index=27, learning_rate=args.learning_rate,
        init_seed=args.seed, start_params=warm,
    )
    warm_dict = params_to_dict(warm)
    trained_dict = params_to_dict(trained)
    domain_losses = {}
    for name, tokens in domains.items():
        before = float(lm_forward_with_moe(warm_dict, model, tokens, config, 27)[0])
        after = float(lm_forward_with_moe(trained_dict, model, tokens, config, 27)[0])
        domain_losses[name] = {"before": before, "after": after, "improvement": before - after}
    report = {
        "steps": args.steps,
        "hard_domain_repeat": args.hard_domain_repeat,
        "router_warm_start_steps": 40,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "domains": list(domains),
        "domain_to_expert": DOMAIN_TO_EXPERT,
        "finite": all(bool(mx.all(mx.isfinite(value))) for _, value in mlx.utils.tree_flatten(params_to_dict(trained))),
        "history_steps": len(history),
        "domain_losses": domain_losses,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
