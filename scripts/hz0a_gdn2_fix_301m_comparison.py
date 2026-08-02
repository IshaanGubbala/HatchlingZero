"""Matched old-vs-corrected GDN-2 replay at the locked HZ-0A topology."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz0a_mlx_model import HZ0AMlxModel


def records(path: Path, steps: int):
    result = []
    with path.open() as handle:
        for line in handle:
            values = json.loads(line)
            if len(values) >= 129:
                result.append(mx.array(values[:129], dtype=mx.int32))
            if len(result) == steps:
                break
    if len(result) != steps:
        raise ValueError(f"requested {steps} records, found {len(result)}")
    return result


def run(mixer: str, batches, validation, vocab_size: int):
    mx.random.seed(444)
    model = HZ0AMlxModel(vocab_size, 768, 31, 12, 2304, (4, 9, 14, 19, 24, 29), native_metal=True, mixer=mixer)
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
    losses, gradient_norms, clipped_gradient_norms, update_norms = [], [], [], []
    started = time.perf_counter()
    for tokens in batches:
        before = [mx.array(value) for _, value in tree_flatten(model.parameters())]

        def loss_fn(m):
            logits, _ = m(tokens[None, :])
            return mx.mean(nn.losses.cross_entropy(logits[:, :-1], tokens[None, 1:]))

        loss, gradients = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, gradients)
        gradients, gradient_norm = optim.clip_grad_norm(gradients, 1.0)
        mx.eval(gradients, gradient_norm)
        gradient_norms.append(float(gradient_norm))
        clipped_gradient_norms.append(min(float(gradient_norm), 1.0))
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        after = [value for _, value in tree_flatten(model.parameters())]
        update_norms.append(float(mx.sqrt(sum(mx.sum((a - b) * (a - b)) for a, b in zip(after, before)))))
        losses.append(float(loss))
    elapsed = time.perf_counter() - started
    validation_losses = []
    for tokens in validation:
        logits, _ = model(tokens[None, :])
        value = mx.mean(nn.losses.cross_entropy(logits[:, :-1], tokens[None, 1:]))
        mx.eval(value)
        validation_losses.append(float(value))
    return {
        "mixer": mixer,
        "parameters": sum(value.size for _, value in tree_flatten(model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "validation_loss": sum(validation_losses) / len(validation_losses),
        "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms),
        "clipped_gradient_norm_mean": sum(clipped_gradient_norms) / len(clipped_gradient_norms),
        "update_norm_mean": sum(update_norms) / len(update_norms),
        "tokens_per_second": len(batches) * 128 / elapsed,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "finite": all(math.isfinite(value) for value in losses + gradient_norms + update_norms),
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/packed/stage1_10m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/repro_1024_val.jsonl"))
    parser.add_argument("--validation-steps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--vocab-size", type=int, default=24576)
    parser.add_argument("--mixer", choices=("gdn2", "gdn2_fix"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batches = records(args.data, args.steps)
    validation = records(args.validation_data, args.validation_steps)
    mixers = (args.mixer,) if args.mixer else ("gdn2", "gdn2_fix")
    report = {"steps": args.steps, "tokens": args.steps * 128, "validation_steps": args.validation_steps, "results": [run(mixer, batches, validation, args.vocab_size) for mixer in mixers]}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
