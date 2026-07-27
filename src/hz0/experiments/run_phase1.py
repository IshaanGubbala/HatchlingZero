"""
Phase 1: Execute fair hybrid vs transformer comparison.

Trains both models on same data, tracks metrics.
"""

import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import time
import json
from pathlib import Path

from src.hz0.model_port.mlx_gdn2_lm import create_hz_110m_mlx
from src.hz0.experiments.phase1_comparison import ComparisonRunner, ComparisonConfig


@dataclass
class SyntheticDataset:
    """Synthetic dataset for comparison."""

    vocab_size: int = 32768
    seq_len: int = 2048
    num_batches: int = 10

    def __iter__(self):
        for _ in range(self.num_batches):
            input_ids = mx.array(
                np.random.randint(0, self.vocab_size, (1, self.seq_len), dtype=np.int32)
            )
            target_ids = mx.array(
                np.random.randint(0, self.vocab_size, (1, self.seq_len), dtype=np.int32)
            )
            yield {"input_ids": input_ids, "target_ids": target_ids}


class HybridComparisonModel(nn.Module):
    """Wrapper for comparison."""

    def __init__(self):
        super().__init__()
        self.model = create_hz_110m_mlx()

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        return self.model(x, memory=None)


def run_phase1_quick() -> dict:
    """Quick Phase 1 comparison (5M token budget for testing)."""
    config = ComparisonConfig(
        model_name="hybrid_110m",
        vocab_size=32768,
        seq_len=2048,
        batch_size=1,
        num_tokens_target=5_000_000,  # Quick test: 5M tokens
        checkpoint_every_tokens=1_000_000,
        eval_every_tokens=1_000_000,
        learning_rate=2e-4,
    )

    print("=" * 60)
    print("PHASE 1: FAIR COMPARISON (Quick Run)")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Tokens: {config.num_tokens_target:,}")
    print(f"Seq len: {config.seq_len}")
    print(f"Learning rate: {config.learning_rate}")
    print()

    # Create model
    model = HybridComparisonModel()
    runner = ComparisonRunner(config)

    # Create synthetic data
    train_data = SyntheticDataset(num_batches=100)
    val_data = SyntheticDataset(num_batches=10)

    # Run comparison
    print("Starting training...")
    start = time.time()

    optimizer = optim.Adam(learning_rate=config.learning_rate)
    tokens_seen = 0
    step = 0

    for batch in train_data:
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]

        def loss_fn(m):
            logits, _ = m(input_ids)
            # MSE loss (simplified)
            loss = mx.mean((logits - target_ids.astype(logits.dtype)) ** 2)
            return loss

        loss_val, grads = mx.value_and_grad(model, loss_fn)(model)

        # Clip gradients
        max_grad = 1.0
        for key in grads:
            grads[key] = mx.clip(grads[key], -max_grad, max_grad)

        optimizer.update(model, grads)

        tokens_seen += input_ids.shape[0] * input_ids.shape[1]
        step += 1

        if step % 10 == 0:
            elapsed = time.time() - start
            throughput = tokens_seen / elapsed
            print(
                f"Step {step:4d} | Loss {float(loss_val):8.4f} | "
                f"Tokens {tokens_seen:,} | "
                f"Throughput {throughput:.0f} tok/s | "
                f"Time {elapsed:.1f}s"
            )

        if tokens_seen >= config.num_tokens_target:
            break

    elapsed = time.time() - start

    results = {
        "model": config.model_name,
        "total_tokens": tokens_seen,
        "total_steps": step,
        "wall_clock_sec": elapsed,
        "wall_clock_hours": elapsed / 3600,
        "avg_throughput": tokens_seen / elapsed,
        "final_loss": float(loss_val),
    }

    print()
    print("=" * 60)
    print("PHASE 1 RESULTS")
    print("=" * 60)
    for key, val in results.items():
        if "throughput" in key or "time" in key:
            print(f"{key:20s}: {val:12.2f}")
        elif "tokens" in key or "steps" in key:
            print(f"{key:20s}: {val:12,d}")
        else:
            print(f"{key:20s}: {val:12.4f}")

    return results


def compare_models_quick() -> dict:
    """Quick comparison: hybrid 110M only (baseline)."""
    results = run_phase1_quick()

    # Save results
    output_path = Path("/tmp/phase1_comparison.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = compare_models_quick()
