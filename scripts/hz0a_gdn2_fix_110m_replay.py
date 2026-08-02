"""100-step real-packed-corpus smoke for the corrected native 110M topology."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz0a_mlx_model import HZ0AMlxModel


def run(data: Path, steps: int) -> dict:
    records = []
    with data.open() as handle:
        for line in handle:
            tokens = json.loads(line)
            if len(tokens) >= 129:
                records.append(mx.array(tokens[:129], dtype=mx.int32))
            if len(records) == steps:
                break
    if len(records) != steps:
        raise ValueError(f"requested {steps} records but found {len(records)}")
    mx.random.seed(77)
    model = HZ0AMlxModel(8192, 576, 22, 18, 1728, (), native_metal=True, mixer="gdn2_fix")
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    def loss_fn(m, tokens):
        logits, _ = m(tokens[None, :])
        return mx.mean(nn.losses.cross_entropy(logits[:, :-1], tokens[None, 1:]))

    value_grad = nn.value_and_grad(model, loss_fn)
    losses = []
    started = time.perf_counter()
    for tokens in records:
        loss, gradients = value_grad(model, tokens)
        optimizer.update(model, gradients)
        mx.eval(loss, model.parameters(), optimizer.state)
        losses.append(float(loss))
    elapsed = time.perf_counter() - started
    return {
        "topology": {"vocab_size": 8192, "d_model": 576, "layers": 22, "heads": 18, "d_ff": 1728},
        "steps": steps,
        "tokens": steps * 128,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "tokens_per_second": steps * 128 / elapsed,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "finite": all(math.isfinite(loss) for loss in losses),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/packed/stage1_10m_train.jsonl"))
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.steps), indent=2, sort_keys=True))
