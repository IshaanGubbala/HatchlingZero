"""Short frozen-backbone calibration for the three integrated E6 MoE layers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from reference.hz0e_e6_integration import TARGET_LAYERS, cross_entropy_loss, forward_e6, init_e6_layers
from reference.hz0e_e3_routing_objectives import params_to_dict
from reference.hz0e_moe_contract import MoeLayerParams
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences


def _tokens(start: int, count: int, width: int = 256) -> mx.array:
    rows = load_real_sequences(GENERAL_DATA_PATH, start + count)[start:]
    return mx.array([row[:width] for row in rows], dtype=mx.int32)


def _loss(model, params, tokens):
    layers = {int(index): MoeLayerParams(**values) for index, values in params.items()}
    result = forward_e6(model, tokens, moe_layers=layers)
    return cross_entropy_loss(result.logits, tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None, help="Optional .npz path for the best external MoE parameters")
    args = parser.parse_args()
    model, _ = load_frozen_model()
    train = [_tokens(i * 4, 4) for i in range(4)]
    validation = _tokens(16, 4)
    initial = {str(index): params_to_dict(value) for index, value in init_e6_layers(model, seed=args.seed).items()}
    params = initial
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    value_and_grad = mx.value_and_grad(_loss, argnums=1)
    initial_loss = float(_loss(model, params, validation))
    best_loss = initial_loss
    best_step = 0
    best_params = {index: {name: mx.array(value) for name, value in values.items()} for index, values in params.items()}
    for step in range(args.steps):
        loss, grads = value_and_grad(model, params, train[step % len(train)])
        grads, _ = optim.clip_grad_norm(grads, args.max_grad_norm)
        params = optimizer.apply_gradients(grads, params)
        mx.eval(params)
        validation_loss = float(_loss(model, params, validation))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_step = step + 1
            best_params = {index: {name: mx.array(value) for name, value in values.items()} for index, values in params.items()}
    params = best_params
    final_loss = best_loss
    leaves = [value for values in params.values() for value in values.values()]
    mx.eval(*leaves)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"layer_{index}_{name}": value for index, values in params.items() for name, value in values.items()}
        mx.savez(str(args.output), **arrays)
    report = {
        "steps": args.steps, "best_step": best_step, "seed": args.seed, "learning_rate": args.learning_rate, "max_grad_norm": args.max_grad_norm, "output": str(args.output) if args.output else None,
        "validation_loss_before": initial_loss, "validation_loss_after": final_loss,
        "improvement": initial_loss - final_loss,
        "finite": all(bool(mx.all(mx.isfinite(value))) for value in leaves),
        "trainable_layers": list(TARGET_LAYERS),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
