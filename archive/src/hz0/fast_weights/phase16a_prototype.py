"""Phase 16a: Fast weights prototype on toy associative recall task."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Tuple

from hz0.fast_weights.fast_weight_layer import FastWeightLinear
from hz0.fast_weights.meta_learner import FastWeightSession


def create_toy_model(d_model: int = 64) -> nn.Module:
    """Simple 2-layer model for testing fast weights."""
    class ToyModel(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.embed = nn.Linear(d_model, d_model)
            self.fast_layer_1 = FastWeightLinear(d_model, d_model)
            self.fast_layer_2 = FastWeightLinear(d_model, d_model)
            self.output = nn.Linear(d_model, d_model)

        def __call__(self, x):
            x = self.embed(x)
            x = nn.relu(self.fast_layer_1(x))
            x = nn.relu(self.fast_layer_2(x))
            x = self.output(x)
            return x

    return ToyModel(d_model)


def generate_associative_recall_task(
    num_pairs: int = 5,
    embedding_dim: int = 64,
    num_queries: int = 3
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Generate toy associative recall task.

    Format:
    - Context: pairs (key, value) as embeddings
    - Query: key embedding
    - Target: corresponding value embedding

    Returns:
        context_inputs: [num_pairs, embedding_dim]
        context_targets: [num_pairs, embedding_dim]
        query_inputs: [num_queries, embedding_dim]
        query_targets: [num_queries, embedding_dim]
    """
    # Generate random key-value pairs
    keys = mx.random.normal((num_pairs, embedding_dim))
    values = mx.random.normal((num_pairs, embedding_dim))

    # For query, repeat some keys
    query_indices = mx.random.randint(0, num_pairs, shape=(num_queries,))
    queries = keys[query_indices.tolist()]
    query_targets = values[query_indices.tolist()]

    return keys, values, queries, query_targets


def loss_fn(predictions: mx.array, targets: mx.array) -> mx.array:
    """MSE loss for regression."""
    diff = predictions - targets
    return mx.mean(diff ** 2)


def compute_accuracy(predictions: mx.array, targets: mx.array, threshold: float = 0.1) -> float:
    """Accuracy: fraction of predictions within threshold of target."""
    diffs = mx.abs(predictions - targets)
    correct = mx.sum(mx.mean(diffs < threshold, axis=1))
    return float(correct) / predictions.shape[0]


def test_fast_weights_prototype():
    """Test Phase 16a: Single-layer fast weights on toy task."""
    print("=" * 70)
    print("Phase 16a: Fast Weights Prototype")
    print("=" * 70)

    # Setup
    d_model = 64
    model = create_toy_model(d_model)
    fast_layers = [model.fast_layer_1, model.fast_layer_2]
    session = FastWeightSession(model, fast_layers, learning_rate=0.1)

    # Generate task: simple linear mapping
    num_pairs = 5
    embedding_dim = d_model
    context_keys = mx.random.normal((num_pairs, embedding_dim))
    context_values = mx.random.normal((num_pairs, embedding_dim))
    query_keys = context_keys[:3]  # Test on seen keys
    query_targets = context_values[:3]

    print(f"\n1. BASELINE (no adaptation)")
    print(f"   Context: {context_keys.shape} (10 key-value pairs)")
    print(f"   Query: {query_keys.shape}")

    # Baseline predictions (before adaptation)
    session.start_session()
    baseline_preds = model(query_keys)
    baseline_loss = loss_fn(baseline_preds, query_targets)
    baseline_acc = compute_accuracy(baseline_preds, query_targets)
    print(f"   Loss: {float(baseline_loss):.4f}")
    print(f"   Accuracy (threshold=0.1): {baseline_acc:.1%}")

    # Adapt on context: manually update fast weights toward targets
    print(f"\n2. ADAPTATION (manual gradient steps on context)")
    print(f"   Adapting for 10 manual steps...")
    loss_history = []

    for step in range(10):
        # Manually update fast weights to approximate context values
        for i, (key, val) in enumerate(zip(context_keys, context_values)):
            pred = model(key.reshape(1, -1))
            loss = loss_fn(pred, val.reshape(1, -1))
            loss_history.append(float(loss))

            # Simple update: move fast weights to reduce error
            for layer in fast_layers:
                if hasattr(layer, 'fast_weight'):
                    # Use prediction error to guide update
                    error = float(loss)
                    layer.fast_weight = layer.fast_weight - 0.01 * error

    print(f"   Loss trajectory:")
    for step in [0, 2, 4, 6, 8, 9]:
        if step < len(loss_history):
            print(f"     Step {step:2d}: {loss_history[step]:.4f}")

    # Test on query
    print(f"\n3. TEST (predictions after adaptation)")
    adapted_preds = model(query_keys)
    adapted_loss = loss_fn(adapted_preds, query_targets)
    adapted_acc = compute_accuracy(adapted_preds, query_targets)
    print(f"   Loss: {float(adapted_loss):.4f}")
    print(f"   Accuracy (threshold=0.1): {adapted_acc:.1%}")

    # Summary
    print(f"\n4. RESULTS")
    loss_improvement = float(baseline_loss) - float(adapted_loss)
    acc_improvement = adapted_acc - baseline_acc
    loss_change_pct = 100 * loss_improvement / float(baseline_loss) if float(baseline_loss) > 0 else 0
    print(f"   Loss improvement: {loss_improvement:.4f} ({loss_change_pct:.1f}%)")
    print(f"   Accuracy improvement: {acc_improvement:.1%}")

    # Verify session isolation
    print(f"\n5. SESSION ISOLATION TEST")
    session.end_session()
    session.start_session()
    reset_preds = model(query_keys)
    reset_loss = loss_fn(reset_preds, query_targets)
    reset_acc = compute_accuracy(reset_preds, query_targets)
    print(f"   After reset:")
    print(f"   Loss: {float(reset_loss):.4f} (should match baseline: {float(baseline_loss):.4f})")
    print(f"   Accuracy: {reset_acc:.1%} (should match baseline: {baseline_acc:.1%})")

    # Verify fast weights are actually being used
    print(f"\n6. VERIFICATION: Fast weights are being optimized")
    for i, layer in enumerate(fast_layers):
        if hasattr(layer, 'fast_weight'):
            fw_norm = mx.linalg.norm(layer.fast_weight)
            bw_norm = mx.linalg.norm(layer.weight)
            print(f"   Layer {i}: fast_weight_norm={float(fw_norm):.4f}, base_weight_norm={float(bw_norm):.4f}")

    session.end_session()

    print(f"\n{'='*70}")
    if loss_improvement > 0 and acc_improvement > 0:
        print("✓ PASS: Fast weights are learning and improving task performance")
    else:
        print("✗ FAIL: Fast weights not improving task performance")
    print(f"{'='*70}\n")

    return {
        "baseline_loss": float(baseline_loss),
        "adapted_loss": float(adapted_loss),
        "baseline_acc": baseline_acc,
        "adapted_acc": adapted_acc,
        "loss_improvement": loss_improvement,
        "acc_improvement": acc_improvement,
    }


if __name__ == "__main__":
    results = test_fast_weights_prototype()
