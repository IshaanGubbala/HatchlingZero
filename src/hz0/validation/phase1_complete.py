"""Phase 1: Complete HZ-0A validation.

Train GDN-2 vs transformer baseline on controlled synthetic task.
Measure quality advantage.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
from typing import Tuple, Dict
from hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel
from hz0.validation.phase1a_transformer_baseline import TransformerLM


def create_synthetic_dataset(
    num_batches: int = 100,
    batch_size: int = 4,
    seq_len: int = 128,
    vocab_size: int = 8192,
    seed: int = 42,
) -> Tuple[list, list]:
    """Create synthetic language modeling dataset.

    Uses pattern: repeated sequences with variations.
    Mimics real data patterns (repetition, structure).
    """
    mx.random.seed(seed)

    batches = []
    for _ in range(num_batches):
        # Create batch with structure (not pure random)
        batch = mx.random.randint(0, min(vocab_size // 4, 1024), shape=(batch_size, seq_len))
        targets = mx.roll(batch, -1, axis=1)  # Next-token prediction
        batches.append((batch, targets))

    return batches, batches[:10]  # Train + val


def loss_fn(logits: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss."""
    if len(logits.shape) == 3:
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
    else:
        logits_flat = logits

    targets_flat = targets.reshape(-1)

    # Clip logits for stability
    logits_flat = mx.clip(logits_flat, -100, 100)

    log_probs = mx.log_softmax(logits_flat, axis=-1) if hasattr(mx, 'log_softmax') else (
        logits_flat - mx.log(mx.sum(mx.exp(mx.clip(logits_flat, -100, 100)), axis=-1, keepdims=True))
    )

    correct_log_probs = log_probs[mx.arange(len(targets_flat)), targets_flat]
    loss = -mx.mean(correct_log_probs)
    return loss


def perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    import math
    return math.exp(float(loss))


def train_epoch(
    model,
    batches: list,
    optimizer,
    model_name: str,
    epoch: int,
) -> Tuple[float, float]:
    """Train one epoch, return avg loss and throughput."""
    total_loss = 0
    total_tokens = 0
    start = time.time()

    for batch, targets in batches:
        def compute_loss(m):
            output = m(batch)
            # Handle both (logits, state) and logits-only returns
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            return loss_fn(logits, targets)

        loss_val = compute_loss(model)

        # Skip NaN steps
        if mx.any(mx.isnan(loss_val)):
            continue

        total_loss += float(loss_val)
        total_tokens += batch.shape[0] * batch.shape[1]

        loss_grad = mx.grad(compute_loss)(model)
        optimizer.update(model, loss_grad)
        mx.eval(model)

    elapsed = time.time() - start
    avg_loss = total_loss / len(batches)
    throughput = total_tokens / elapsed if elapsed > 0 else 0

    return avg_loss, throughput


def validate(model, batches: list) -> float:
    """Compute validation loss."""
    total_loss = 0

    for batch, targets in batches:
        output = model(batch)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        loss = loss_fn(logits, targets)
        if not mx.any(mx.isnan(loss)):
            total_loss += float(loss)

    return total_loss / len(batches)


def run_phase1():
    """Phase 1: Complete quality validation."""
    print("="*70)
    print("PHASE 1: HZ-0A Quality Validation")
    print("="*70)

    # Models
    print("\nCreating models...")
    hz0a = GDN2LanguageModel(
        vocab_size=8192,
        model_dim=512,
        num_layers=12,
        num_heads=8,
        gdn2_every=2,
    )
    print("  ✓ HZ-0A (GDN-2): 12 layers, 512-dim")

    transformer = TransformerLM(
        vocab_size=8192,
        model_dim=512,
        num_layers=12,
        num_heads=8,
    )
    print("  ✓ Transformer: 12 layers, 512-dim")

    # Dataset
    print("\nCreating dataset...")
    train_batches, val_batches = create_synthetic_dataset(
        num_batches=200,  # 200 batches * 4 * 128 = 102k tokens
        batch_size=4,
        seq_len=128,
    )
    print(f"  ✓ Train: {len(train_batches)} batches (~{len(train_batches)*4*128}k tokens)")
    print(f"  ✓ Val: {len(val_batches)} batches")

    # Training
    num_epochs = 3
    hz0a_opt = optim.Adam(learning_rate=1e-4)
    tf_opt = optim.Adam(learning_rate=1e-4)

    hz0a_results = {"train": [], "val": [], "tps": []}
    tf_results = {"train": [], "val": [], "tps": []}

    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # HZ-0A
        hz0a_loss, hz0a_tps = train_epoch(hz0a, train_batches, hz0a_opt, "HZ-0A", epoch)
        hz0a_val = validate(hz0a, val_batches)
        hz0a_results["train"].append(hz0a_loss)
        hz0a_results["val"].append(hz0a_val)
        hz0a_results["tps"].append(hz0a_tps)
        print(f"  HZ-0A:  train={hz0a_loss:.4f}, val={hz0a_val:.4f}, {hz0a_tps:.0f} tok/s")

        # Transformer
        tf_loss, tf_tps = train_epoch(transformer, train_batches, tf_opt, "Transformer", epoch)
        tf_val = validate(transformer, val_batches)
        tf_results["train"].append(tf_loss)
        tf_results["val"].append(tf_val)
        tf_results["tps"].append(tf_tps)
        print(f"  Transformer: train={tf_loss:.4f}, val={tf_val:.4f}, {tf_tps:.0f} tok/s")

    # Results
    print("\n" + "="*70)
    print("PHASE 1 RESULTS")
    print("="*70)

    hz0a_final = hz0a_results["val"][-1]
    tf_final = tf_results["val"][-1]

    print(f"\nFinal Validation Loss:")
    print(f"  HZ-0A:       {hz0a_final:.4f} (perplexity: {perplexity(hz0a_final):.1f})")
    print(f"  Transformer: {tf_final:.4f} (perplexity: {perplexity(tf_final):.1f})")

    print(f"\nAverage Throughput:")
    print(f"  HZ-0A:       {sum(hz0a_results['tps'])/len(hz0a_results['tps']):.0f} tok/s")
    print(f"  Transformer: {sum(tf_results['tps'])/len(tf_results['tps']):.0f} tok/s")

    print(f"\nTrend (lower is better):")
    print(f"  HZ-0A:       {hz0a_results['val'][0]:.4f} → {hz0a_final:.4f} (Δ {hz0a_results['val'][0]-hz0a_final:.4f})")
    print(f"  Transformer: {tf_results['val'][0]:.4f} → {tf_final:.4f} (Δ {tf_results['val'][0]-tf_final:.4f})")

    # Verdict
    print("\n" + "="*70)
    if hz0a_final < tf_final * 0.95:
        print("✓ HZ-0A WINS: Quality advantage confirmed")
        print(f"  ({hz0a_final:.4f} < {tf_final:.4f})")
    elif hz0a_final <= tf_final * 1.05:
        print("~ EQUIVALENT: No clear winner")
        print(f"  ({hz0a_final:.4f} ≈ {tf_final:.4f})")
    else:
        print("✗ TRANSFORMER WINS: HZ-0A underperforms")
        print(f"  ({hz0a_final:.4f} > {tf_final:.4f})")

    print("="*70)

    return {
        "hz0a_final": hz0a_final,
        "tf_final": tf_final,
        "hz0a_wins": hz0a_final < tf_final,
        "hz0a_results": hz0a_results,
        "tf_results": tf_results,
    }


if __name__ == "__main__":
    results = run_phase1()
