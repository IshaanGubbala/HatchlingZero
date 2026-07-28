"""Compare native-Metal GDN training with the MLX reference path."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from reference.hz0a_mlx_model import HZ0AMlxModel


def flat_bytes(model):
    values = [mx.array(value) for _, value in tree_flatten(model.parameters())]
    mx.eval(*values)
    return b"".join(np.asarray(value).tobytes() for value in values)


def flat_norm(values):
    return float(mx.sqrt(sum(mx.sum(value * value) for value in values)))


def run(steps: int) -> dict:
    mx.random.seed(41)
    native = HZ0AMlxModel(64, 32, 3, 4, 64, (), native_metal=True)
    reference = HZ0AMlxModel(64, 32, 3, 4, 64, (), native_metal=False)
    reference.update(native.parameters())
    native_optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
    reference_optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
    rng = mx.random.key(7)
    batches = [mx.random.randint(0, 64, (1, 16), key=mx.random.split(rng)[0]) for _ in range(steps)]
    metrics = []
    started = time.perf_counter()

    def loss_fn(model, tokens):
        logits, _ = model(tokens)
        return mx.mean(nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:]))

    native_value_grad = nn.value_and_grad(native, loss_fn)
    reference_value_grad = nn.value_and_grad(reference, loss_fn)
    for step, tokens in enumerate(batches, 1):
        native_loss, native_grads = native_value_grad(native, tokens)
        reference_loss, reference_grads = reference_value_grad(reference, tokens)
        mx.eval(native_loss, reference_loss, native_grads, reference_grads)
        native_gradient_values = [value for _, value in tree_flatten(native_grads)]
        reference_gradient_values = [value for _, value in tree_flatten(reference_grads)]
        gradient_error = max(float(mx.max(mx.abs(a - b))) for a, b in zip(native_gradient_values, reference_gradient_values))
        native_before = [mx.array(value) for _, value in tree_flatten(native.parameters())]
        reference_before = [mx.array(value) for _, value in tree_flatten(reference.parameters())]
        native_optimizer.update(native, native_grads)
        reference_optimizer.update(reference, reference_grads)
        mx.eval(native.parameters(), reference.parameters(), native_optimizer.state, reference_optimizer.state)
        native_after = [value for _, value in tree_flatten(native.parameters())]
        reference_after = [value for _, value in tree_flatten(reference.parameters())]
        update_error = max(float(mx.max(mx.abs((a - b) - (c - d)))) for a, b, c, d in zip(native_before, native_after, reference_before, reference_after))
        metrics.append({"step": step, "native_loss": float(native_loss), "reference_loss": float(reference_loss), "loss_difference": abs(float(native_loss - reference_loss)), "max_gradient_error": gradient_error, "max_update_error": update_error, "native_update_norm": flat_norm([a - b for a, b in zip(native_after, native_before)]), "finite": bool(mx.all(mx.isfinite(native_loss)) and all(bool(mx.all(mx.isfinite(value))) for value in native_after))})
    native_fingerprint = hashlib.sha256(flat_bytes(native)).hexdigest()
    reference_fingerprint = hashlib.sha256(flat_bytes(reference)).hexdigest()
    return {"steps": steps, "metrics": metrics, "native_fingerprint": native_fingerprint, "reference_fingerprint": reference_fingerprint, "fingerprints_match": native_fingerprint == reference_fingerprint, "max_loss_difference": max(item["loss_difference"] for item in metrics), "max_gradient_error": max(item["max_gradient_error"] for item in metrics), "max_update_error": max(item["max_update_error"] for item in metrics), "finite": all(item["finite"] for item in metrics), "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "execution_seconds": time.perf_counter() - started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args.steps), indent=2, sort_keys=True))
