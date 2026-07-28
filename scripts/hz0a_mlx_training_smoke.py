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

from reference.hz0a_mlx_model import HZ0AMlxModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    model = HZ0AMlxModel(64, 32, 3, 4, 64, (), native_metal=True)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
    tokens = mx.arange(16).reshape(1, 16) % 64

    def loss_fn(model):
        logits, _ = model(tokens)
        return mx.mean(nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:]))

    value_and_grad = nn.value_and_grad(model, loss_fn)
    before = model.embedding.weight
    metrics = []
    for step in range(args.steps):
        loss, grads = value_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)
        metrics.append({"step": step, "loss": float(loss)})
    mx.eval(before, model.embedding.weight)
    delta = float(mx.max(mx.abs(model.embedding.weight - before)))
    if not all(metric["loss"] == metric["loss"] for metric in metrics) or delta <= 0:
        raise RuntimeError("non-finite loss or missing parameter update")
    print(json.dumps({"native_forward": True, "steps": args.steps, "metrics": metrics, "max_parameter_delta": delta}))


if __name__ == "__main__":
    main()
