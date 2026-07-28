"""
Phase 1: Fair comparison baseline.
"""

import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
from mlx.nn import losses
import numpy as np
import time
import json
from pathlib import Path

from hz0.model_port.mlx_gdn2_lm import create_hz_36m_mlx


def run_phase1_baseline():
    """Run Phase 1 fair comparison baseline."""
    print("=" * 60)
    print("PHASE 1: FAIR COMPARISON BASELINE")
    print("=" * 60)

    # Config
    model_name = "hybrid_36m"
    vocab_size = 32768
    seq_len = 256
    batch_size = 1
    num_tokens_target = 10_000
    learning_rate = 2e-4

    print(f"Model: {model_name}")
    print(f"Tokens: {num_tokens_target:,}")
    print(f"Seq len: {seq_len}")
    print(f"Learning rate: {learning_rate}")
    print()

    # Create model
    model = create_hz_36m_mlx()
    optimizer = optim.Adam(learning_rate=learning_rate)

    print("Starting training...")
    start = time.time()
    tokens_seen = 0
    step = 0

    while tokens_seen < num_tokens_target:
        # Generate batch
        input_ids = mx.array(
            np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
        )
        target_ids = mx.array(
            np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
        )

        def loss_fn(m):
            logits, _ = m(input_ids)
            loss = mx.mean(losses.cross_entropy(logits, target_ids))
            return loss

        # Train step
        loss_val, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(loss_val)

        tokens_seen += batch_size * seq_len
        step += 1

        if step % 5 == 0:
            elapsed = time.time() - start
            throughput = tokens_seen / elapsed
            print(
                f"Step {step:4d} | Loss {float(loss_val):8.4f} | "
                f"Tokens {tokens_seen:,} | "
                f"Throughput {throughput:.0f} tok/s"
            )

    elapsed = time.time() - start

    results = {
        "model": model_name,
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
        if isinstance(val, str):
            print(f"{key:20s}: {val}")
        elif "throughput" in key or "time" in key:
            print(f"{key:20s}: {val:12.2f}")
        elif "tokens" in key or "steps" in key:
            print(f"{key:20s}: {val:12,d}")
        else:
            print(f"{key:20s}: {val:12.4f}")

    # Save
    output_path = Path("/tmp/phase1_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = run_phase1_baseline()
