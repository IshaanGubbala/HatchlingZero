"""Bounded exact-301M native GDN-2-fix chunk/record stability probe."""

from __future__ import annotations

import argparse
import gc
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


def run(records: int, chunks: int, seed: int) -> dict:
    mx.random.seed(seed)
    model = HZ0AMlxModel(24576, 768, 31, 12, 2304, (4, 9, 14, 19, 24, 29), native_metal=True, mixer="gdn2_fix")
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
    state = None
    telemetry = []
    losses = []
    started = time.perf_counter()

    for record in range(records):
        tokens = mx.random.randint(0, 24576, (1, chunks * 128))
        state = None
        for chunk in range(chunks):
            batch = tokens[:, chunk * 128:(chunk + 1) * 128]

            def loss_fn(m):
                logits, _ = m(batch, state)
                return mx.mean(nn.losses.cross_entropy(logits[:, :-1], batch[:, 1:]))

            loss, gradients = nn.value_and_grad(model, loss_fn)(model)
            mx.eval(loss, gradients)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(float(loss))
            telemetry.append({
                "record": record + 1,
                "chunk": chunk + 1,
                "loss": float(loss),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            })
            _, state = model(batch, state)
            mx.eval(state)
            gc.collect()

    return {
        "parameters": sum(value.size for _, value in tree_flatten(model.parameters())),
        "records": records,
        "chunks_per_record": chunks,
        "tokens": records * chunks * 128,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "finite": all(math.isfinite(value) for value in losses),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "execution_seconds": time.perf_counter() - started,
        "telemetry": telemetry,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=2)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=92)
    args = parser.parse_args()
    print(json.dumps(run(args.records, args.chunks, args.seed), indent=2, sort_keys=True))
