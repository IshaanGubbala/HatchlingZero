"""Evaluate HZ-0A stage checkpoints on a shared streaming packed split."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restart.hz0a_dataset import StreamingResumablePackedDataset
from scripts.hz0a_tiny_training_comparison import TinyHybridLM, TinyTransformerLM, loss_for, parameter_bytes


def evaluate(checkpoint: Path, model, data: Path, batches: int, batch_size: int, vocab_size: int) -> dict:
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = StreamingResumablePackedDataset(data, shuffle_seed=0)
    total_loss = 0.0
    tokens = 0
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(batches):
            batch = torch.from_numpy(dataset.next_batch(batch_size)).remainder(vocab_size)
            loss = loss_for(model, batch)
            count = batch.shape[0] * (batch.shape[1] - 1)
            total_loss += float(loss) * count
            tokens += count
    elapsed = time.perf_counter() - start
    mean_loss = total_loss / tokens
    return {"checkpoint": str(checkpoint), "tokens": tokens, "loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0)), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "parameter_bytes": parameter_bytes(model), "evaluation_seconds": elapsed, "tokens_per_second": tokens / elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate shared HZ-0A checkpoints.")
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"data": str(args.data), "batches": args.batches, "batch_size": args.batch_size, "models": {"hybrid": evaluate(args.hybrid_checkpoint, TinyHybridLM(vocab_size=args.vocab_size), args.data, args.batches, args.batch_size, args.vocab_size), "transformer": evaluate(args.transformer_checkpoint, TinyTransformerLM(vocab_size=args.vocab_size), args.data, args.batches, args.batch_size, args.vocab_size)}}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
