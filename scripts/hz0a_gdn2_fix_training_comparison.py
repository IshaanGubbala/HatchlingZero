"""Deterministic small-model before/after training comparison for GDN-2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz0a_torch_model import HZ0AConfig, HZ0AModel


def run(mixer: str, batches: torch.Tensor, targets: torch.Tensor, steps: int):
    torch.manual_seed(1234)
    config = HZ0AConfig(64, 32, 2, 2, 16, 16, 64, (1,), mixer=mixer)
    model = HZ0AModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    losses, grad_norms, update_norms = [], [], []
    started = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batches[step])
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets[step].reshape(-1))
        loss.backward()
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)))
        before = [p.detach().clone() for p in model.parameters()]
        optimizer.step()
        update_sq = sum((after.detach() - old).square().sum() for after, old in zip(model.parameters(), before))
        update_norms.append(float(update_sq.sqrt()))
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - started
    return {
        "mixer": mixer,
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "gradient_norm_mean": sum(grad_norms) / len(grad_norms),
        "update_norm_mean": sum(update_norms) / len(update_norms),
        "steps_per_second": steps / elapsed,
        "finite": all(torch.isfinite(torch.tensor(item)) for item in losses + grad_norms + update_norms),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    batches = torch.randint(0, 64, (args.steps, args.batch_size, args.sequence_length), generator=generator)
    targets = torch.roll(batches, shifts=-1, dims=-1)
    report = {
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "results": [run(mixer, batches, targets, args.steps) for mixer in ("gdn2", "gdn2_fix")],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
