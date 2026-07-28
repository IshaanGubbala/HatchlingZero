"""Run a bounded MLX AdamW smoke through the native-forward HZ-0A model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from reference.hz0a_mlx_model import HZ0AMlxModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    mx.random.seed(args.seed)
    model = HZ0AMlxModel(64, 32, 3, 4, 64, (), native_metal=True)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    tokens = mx.arange(16).reshape(1, 16) % 64
    validation_tokens = (tokens + 7) % 64

    def loss_fn(model):
        logits, _ = model(tokens)
        return mx.mean(nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:]))

    value_and_grad = nn.value_and_grad(model, loss_fn)
    before = [(key, mx.array(value)) for key, value in tree_flatten(model.parameters())]
    metrics = []
    for step in range(args.steps):
        loss, grads = value_and_grad(model)
        mx.eval(loss, grads)
        flat_grads = [value for _, value in tree_flatten(grads)]
        gradient_norm = float(mx.sqrt(sum(mx.sum(value * value) for value in flat_grads)))
        old_parameters = [(key, mx.array(value)) for key, value in tree_flatten(model.parameters())]
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)
        flat_old = [value for _, value in old_parameters]
        flat_new = [value for _, value in tree_flatten(model.parameters())]
        update_norm = float(mx.sqrt(sum(mx.sum((new - old) * (new - old)) for old, new in zip(flat_old, flat_new))))
        validation_loss = None
        if (step + 1) % args.validation_interval == 0 or step + 1 == args.steps:
            validation_logits, _ = model(validation_tokens)
            validation_loss = float(mx.mean(nn.losses.cross_entropy(validation_logits[:, :-1], validation_tokens[:, 1:])))
        metrics.append({"step": step + 1, "loss": float(loss), "validation_loss": validation_loss, "gradient_norm": gradient_norm, "update_norm": update_norm})
    flat_before = [value for _, value in before]
    flat_after = [value for _, value in tree_flatten(model.parameters())]
    delta = float(max(mx.max(mx.abs(new - old)) for old, new in zip(flat_before, flat_after)))
    if not all(metric["loss"] == metric["loss"] for metric in metrics) or delta <= 0:
        raise RuntimeError("non-finite loss or missing parameter update")
    print(json.dumps({"native_forward": True, "seed": args.seed, "steps": args.steps, "metrics": metrics, "max_parameter_delta": delta, "final_validation_loss": metrics[-1]["validation_loss"]}))


if __name__ == "__main__":
    main()
