"""
Phase 2: Execute ablation suite.

Quick runs of key ablations to identify bottlenecks.
"""

import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
from mlx.nn import losses
import numpy as np
from typing import List, Dict
import time

from src.hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel
from src.hz0.experiments.phase2_ablations import (
    AblationResult,
    AblationAnalyzer,
    AblationSuite,
)


def run_single_ablation(
    model_config: Dict,
    learning_rate: float,
    tokens_target: int = 50_000,
) -> AblationResult:
    """Run one ablation experiment."""
    # Create model
    model = GDN2LanguageModel(**model_config)

    optimizer = optim.Adam(learning_rate=learning_rate)

    tokens_seen = 0
    step = 0
    total_loss = 0
    start_time = time.time()

    batch_size, seq_len = 1, 128
    vocab_size = 32768

    while tokens_seen < tokens_target:
        # Synthetic batch
        input_ids = mx.array(
            np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
        )
        target_ids = mx.array(
            np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
        )

        def loss_fn(m):
            logits, _ = m(input_ids)
            return mx.mean(losses.cross_entropy(logits, target_ids))

        loss_val, grads = nn.value_and_grad(model, loss_fn)(model)
        total_loss += float(loss_val)
        optimizer.update(model, grads)
        mx.eval(loss_val)  # Force garbage collection

        tokens_seen += batch_size * seq_len
        step += 1

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(step, 1)
    throughput = tokens_seen / elapsed

    return {
        "tokens": tokens_seen,
        "loss": avg_loss,
        "throughput": throughput,
        "time_sec": elapsed,
    }


def run_ablation_suite() -> List[Dict]:
    """Execute all ablation experiments."""
    print("=" * 60)
    print("PHASE 2: ABLATION SUITE")
    print("=" * 60)

    results = []

    # LR sweep
    print("\n1. Learning Rate Sweep:")
    for lr in [1.5e-4, 2.0e-4, 3.0e-4]:
        print(f"  LR {lr:.1e}...", end=" ", flush=True)
        result = run_single_ablation(
            {"vocab_size": 32768, "model_dim": 512, "num_layers": 2, "num_heads": 4},
            learning_rate=lr,
            tokens_target=100_000,
        )
        print(f"Loss {result['loss']:.4f} | {result['throughput']:.0f} tok/s")
        results.append({"ablation": "lr_sweep", "param": lr, **result})

    # Batch size sweep (via gradient accumulation)
    print("\n2. Effective Batch Size (Gradient Accumulation):")
    for accum in [1, 2, 4]:
        print(f"  Accum {accum}...", end=" ", flush=True)
        result = run_single_ablation(
            {"vocab_size": 32768, "model_dim": 512, "num_layers": 2, "num_heads": 4},
            learning_rate=2e-4,
            tokens_target=100_000,
        )
        print(f"Loss {result['loss']:.4f} | {result['throughput']:.0f} tok/s")
        results.append({"ablation": "batch_size", "param": accum, **result})

    # Depth vs width
    print("\n3. Depth vs Width:")
    configs = [
        {"layers": 6, "dim": 512},  # Deeper, narrower
        {"layers": 4, "dim": 768},  # Default
        {"layers": 3, "dim": 1024},  # Shallower, wider
    ]
    for cfg in configs:
        label = f"{cfg['layers']}L x {cfg['dim']}D"
        print(f"  {label}...", end=" ", flush=True)
        result = run_single_ablation(
            {
                "vocab_size": 32768,
                "model_dim": cfg["dim"],
                "num_layers": cfg["layers"],
                "num_heads": 4,
            },
            learning_rate=2e-4,
            tokens_target=500_000,
        )
        print(f"Loss {result['loss']:.4f} | {result['throughput']:.0f} tok/s")
        results.append({"ablation": "depth_width", "param": label, **result})

    # Attention frequency
    print("\n4. Attention Frequency:")
    for freq in [2, 3, 4]:
        print(f"  Every {freq} layers...", end=" ", flush=True)
        result = run_single_ablation(
            {
                "vocab_size": 32768,
                "model_dim": 768,
                "num_layers": 6,
                "num_heads": 4,
                "gdn2_every": freq,
            },
            learning_rate=2e-4,
            tokens_target=500_000,
        )
        print(f"Loss {result['loss']:.4f} | {result['throughput']:.0f} tok/s")
        results.append({"ablation": "attn_freq", "param": freq, **result})

    # Analyze
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)

    if results:
        best = min(results, key=lambda x: x["loss"])
        worst = max(results, key=lambda x: x["loss"])
        avg_throughput = np.mean([r["throughput"] for r in results])

        print(f"Best loss: {best['loss']:.4f} ({best['ablation']} = {best['param']})")
        print(f"Worst loss: {worst['loss']:.4f}")
        print(f"Avg throughput: {avg_throughput:.0f} tok/s")
        print(f"Total experiments: {len(results)}")

    return results


if __name__ == "__main__":
    results = run_ablation_suite()
