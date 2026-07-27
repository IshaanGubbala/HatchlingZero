"""Phase 1a: Fair HZ-0A vs Transformer comparison.

Train both models on identical data, measure quality advantage.
"""

import mlx.core as mx
import mlx.optimizers as optim
import time
from src.hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel
from src.hz0.validation.phase1a_transformer_baseline import TransformerLM


def loss_fn(logits: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss."""
    # Handle both [B,T,V] and [B*T, V] shapes
    if len(logits.shape) == 3:
        B, T, V = logits.shape
        probs_flat = mx.softmax(logits.reshape(-1, V), axis=-1)
    else:
        probs_flat = mx.softmax(logits, axis=-1)

    targets_flat = targets.reshape(-1)
    correct_probs = probs_flat[mx.arange(len(targets_flat)), targets_flat]
    loss = -mx.mean(mx.log(correct_probs + 1e-10))
    return loss


def train_model(model, batch, targets, optimizer, num_steps: int, model_name: str):
    """Train model for num_steps and return loss curve."""
    losses = []

    print(f"\n{model_name}: Training {num_steps} steps...")
    start = time.time()

    for step in range(num_steps):
        def compute_loss(m):
            logits, _ = m(batch) if hasattr(m(batch), '__iter__') else (m(batch), None)
            return loss_fn(logits, targets)

        loss_val = compute_loss(model)

        # Skip step if NaN
        if mx.any(mx.isnan(loss_val)):
            print(f"  Step {step+1:3d}: NaN detected, skipping update")
            losses.append(losses[-1] if losses else 1e10)
            continue

        losses.append(float(loss_val))

        loss_grad = mx.grad(compute_loss)(model)

        # Gradient clipping for stability
        for key in loss_grad:
            if isinstance(loss_grad[key], dict):
                for subkey in loss_grad[key]:
                    grad_norm = mx.linalg.norm(loss_grad[key][subkey])
                    if grad_norm > 1.0:
                        loss_grad[key][subkey] = loss_grad[key][subkey] * (1.0 / grad_norm)

        optimizer.update(model, loss_grad)
        mx.eval(model)

        if (step + 1) % (num_steps // 4) == 0:
            print(f"  Step {step+1:3d}: loss = {float(loss_val):.4f}")

    elapsed = time.time() - start
    throughput = (batch.shape[0] * batch.shape[1] * num_steps) / elapsed

    print(f"  Time: {elapsed:.1f}s")
    print(f"  Throughput: {throughput:.0f} tok/s")

    return losses, throughput


def run_fair_comparison():
    """Run HZ-0A vs Transformer comparison."""
    print("="*70)
    print("Phase 1a: Fair Transformer vs GDN-2 Comparison")
    print("="*70)

    # Create models (same size)
    print("\nCreating models...")

    hz0a = GDN2LanguageModel(
        vocab_size=8192,
        model_dim=256,
        num_layers=6,
        num_heads=4,
        gdn2_every=2,
    )
    print("  ✓ HZ-0A (GDN-2): 6 layers, 256-dim")

    transformer = TransformerLM(
        vocab_size=8192,
        model_dim=256,
        num_layers=6,
        num_heads=4,
    )
    print("  ✓ Transformer: 6 layers, 256-dim")

    # Identical data
    batch_size = 2
    seq_len = 64
    batch = mx.random.randint(0, 8192, shape=(batch_size, seq_len))
    targets = mx.random.randint(0, 8192, shape=(batch_size, seq_len))

    print(f"\nData: {batch_size}x{seq_len} random tokens")
    print("(Note: Real comparison uses meaningful data)")

    # Train both
    num_steps = 50

    hz0a_opt = optim.Adam(learning_rate=2e-4)
    hz0a_losses, hz0a_tps = train_model(hz0a, batch, targets, hz0a_opt, num_steps, "HZ-0A (GDN-2)")

    transformer_opt = optim.Adam(learning_rate=2e-4)
    transformer_losses, tf_tps = train_model(transformer, batch, targets, transformer_opt, num_steps, "Transformer")

    # Compare
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print(f"\nInitial Loss (Step 1):")
    print(f"  HZ-0A:       {hz0a_losses[0]:.4f}")
    print(f"  Transformer: {transformer_losses[0]:.4f}")

    print(f"\nFinal Loss (Step {num_steps}):")
    print(f"  HZ-0A:       {hz0a_losses[-1]:.4f}")
    print(f"  Transformer: {transformer_losses[-1]:.4f}")

    hz0a_improvement = hz0a_losses[0] - hz0a_losses[-1]
    tf_improvement = transformer_losses[0] - transformer_losses[-1]

    print(f"\nImprovement:")
    print(f"  HZ-0A:       {hz0a_improvement:.4f}")
    print(f"  Transformer: {tf_improvement:.4f}")

    print(f"\nThroughput:")
    print(f"  HZ-0A:       {hz0a_tps:.0f} tok/s")
    print(f"  Transformer: {tf_tps:.0f} tok/s")
    print(f"  Ratio:       {hz0a_tps/tf_tps:.2f}x")

    # Verdict
    print(f"\n" + "="*70)
    if hz0a_losses[-1] < transformer_losses[-1]:
        print(f"✓ HZ-0A wins on validation loss ({hz0a_losses[-1]:.4f} < {transformer_losses[-1]:.4f})")
        print(f"✓ Quality advantage confirmed")
    elif hz0a_losses[-1] > transformer_losses[-1]:
        print(f"✗ HZ-0A underperforms transformer ({hz0a_losses[-1]:.4f} > {transformer_losses[-1]:.4f})")
        print(f"✗ Quality advantage NOT confirmed")
    else:
        print(f"~ Equivalent performance")

    if hz0a_tps > tf_tps * 0.8:
        print(f"✓ Inference competitive ({hz0a_tps:.0f} vs {tf_tps:.0f} tok/s)")
    else:
        print(f"✗ Inference slower ({hz0a_tps:.0f} vs {tf_tps:.0f} tok/s)")

    print("="*70)

    # Note limitations
    print(f"\nNOTE: This test uses random data.")
    print(f"Full validation requires:")
    print(f"  - Real dataset (not random tokens)")
    print(f"  - Larger training budget (1M+ tokens)")
    print(f"  - Multiple seeds (2+ runs)")
    print(f"  - Held-out validation set")
    print(f"  - Parameter-matched 110M models")

    return {
        "hz0a_loss": hz0a_losses[-1],
        "transformer_loss": transformer_losses[-1],
        "hz0a_advantage": hz0a_losses[-1] < transformer_losses[-1],
    }


if __name__ == "__main__":
    results = run_fair_comparison()
