"""Phase 6: Optimization sweep (learning rate, batch size).

Quick experiments to find best hyperparameters for 110M model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import json
from typing import Tuple, List, Dict, Any

from hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel


def load_synthetic_batches(num_batches: int = 50) -> Tuple[List, List]:
    """Create synthetic data for quick sweep."""
    train_batches = []
    val_batches = []

    for _ in range(num_batches):
        tokens = mx.random.randint(0, 256, (2, 256))
        targets = mx.random.randint(0, 256, (2, 256))
        train_batches.append((tokens, targets))

    for _ in range(10):
        tokens = mx.random.randint(0, 256, (2, 256))
        targets = mx.random.randint(0, 256, (2, 256))
        val_batches.append((tokens, targets))

    return train_batches, val_batches


def compute_loss(logits: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    logits_flat = mx.clip(logits_flat, -100, 100)
    max_logits = mx.max(logits_flat, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits_flat - max_logits)
    sum_exp = mx.sum(exp_logits, axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - mx.log(sum_exp + 1e-9)

    correct_log_probs = log_softmax[mx.arange(len(targets_flat)), targets_flat]
    loss = -mx.mean(correct_log_probs)
    return loss


def run_sweep():
    """Run Phase 6 optimization sweep."""
    print("="*70)
    print("Phase 6: Optimization Sweep (Learning Rate + Batch Size)")
    print("="*70)

    # Create model (110M config)
    model = GDN2LanguageModel(vocab_size=256, model_dim=768, num_layers=24, num_heads=12)

    # Load data
    print("\n[1/3] Loading synthetic batches...")
    train_batches, val_batches = load_synthetic_batches(50)
    print(f"✓ Train: {len(train_batches)} batches, Val: {len(val_batches)} batches")

    # Sweep configs
    lrs = [1.0e-4, 1.5e-4, 2.0e-4, 3.0e-4]
    batch_sizes = [1, 2, 4]

    results = []

    print("\n[2/3] Running sweeps...")
    print("-" * 70)
    print(f"{'LR':<12} {'Accum':<8} {'Loss':<10} {'Time':<10}")
    print("-" * 70)

    for lr in lrs:
        for accum in batch_sizes:
            start = time.time()

            # Create fresh model for each sweep
            model = GDN2LanguageModel(vocab_size=256, model_dim=768, num_layers=24, num_heads=12)
            optimizer = optim.Adam(learning_rate=lr)

            total_loss = 0.0

            # Quick 10 steps
            for step in range(10):
                for batch_idx, (tokens, targets) in enumerate(train_batches[:5]):
                    def loss_fn(m):
                        logits, _ = m(tokens)
                        return compute_loss(logits, targets)

                    loss, grads = nn.value_and_grad(model, loss_fn)(model)
                    optimizer.update(model, grads)
                    total_loss += float(loss)

            elapsed = time.time() - start
            avg_loss = total_loss / (10 * 5)

            result = {
                "lr": lr,
                "gradient_accumulation": accum,
                "loss": avg_loss,
                "time": elapsed,
            }
            results.append(result)

            print(f"{lr:<12.1e} {accum:<8d} {avg_loss:<10.4f} {elapsed:<10.1f}s")

    # Save results
    print("\n[3/3] Saving results...")
    output_path = Path("outputs/phase6_sweep.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved: {output_path}")

    # Find best
    best = min(results, key=lambda x: x["loss"])
    print(f"\n✓ Best config: LR={best['lr']:.1e}, Loss={best['loss']:.4f}")

    print("="*70)


if __name__ == "__main__":
    run_sweep()
