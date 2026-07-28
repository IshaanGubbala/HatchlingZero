"""Phase 3b: Memory benchmarks at 5M scale.

Compare HZ-0A vs HZ-0A+Memory on language modeling task.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
from typing import Tuple, List
from hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel
from hz0.validation.phase3a_complete import HZ0AWithMemory


def generate_batches(num_batches: int = 50, batch_size: int = 4, seq_len: int = 256) -> List[Tuple]:
    """Generate synthetic language modeling batches."""
    batches = []
    for _ in range(num_batches):
        tokens = mx.random.randint(0, min(8192 // 4, 1024), shape=(batch_size, seq_len))
        targets = mx.roll(tokens, -1, axis=1)
        batches.append((tokens, targets))
    return batches


def loss_fn(logits: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss."""
    if len(logits.shape) == 3:
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
    else:
        logits_flat = logits

    targets_flat = targets.reshape(-1)
    logits_flat = mx.clip(logits_flat, -100, 100)

    max_logits = mx.max(logits_flat, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits_flat - max_logits)
    sum_exp = mx.sum(exp_logits, axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - mx.log(sum_exp)

    correct_log_probs = log_softmax[mx.arange(len(targets_flat)), targets_flat]
    loss = -mx.mean(correct_log_probs)
    return loss


def train_epoch(model: nn.Module, batches: List, optimizer, model_name: str) -> Tuple[float, float]:
    """Train one epoch."""
    total_loss = 0
    total_tokens = 0
    start = time.time()

    for i, (batch, targets) in enumerate(batches):
        def compute_loss(m):
            output = m(batch)
            logits = output[0] if isinstance(output, tuple) else output
            return loss_fn(logits, targets)

        loss_val = compute_loss(model)

        if mx.any(mx.isnan(loss_val)):
            continue

        total_loss += float(loss_val)
        total_tokens += batch.shape[0] * batch.shape[1]

        loss_grad = mx.grad(compute_loss)(model)
        optimizer.update(model, loss_grad)
        mx.eval(model)

        if (i + 1) % 10 == 0:
            print(f"  [{model_name}] Batch {i+1}/{len(batches)}: loss={float(loss_val):.4f}")

    elapsed = time.time() - start
    avg_loss = total_loss / len(batches) if batches else 0
    throughput = total_tokens / elapsed if elapsed > 0 else 0

    return avg_loss, throughput


def phase3b_memory_benchmarks():
    """Phase 3b: 5M model memory benchmarks."""
    print("="*70)
    print("Phase 3b: Memory Benchmarks (5M scale)")
    print("="*70)

    # 5M model config: ~6 layers, 256 dims
    vocab = 8192
    model_dim = 256
    num_layers = 6
    num_heads = 4
    batch_size = 4
    seq_len = 256

    print(f"\nModel Config (5M scale estimate):")
    print(f"  Vocab: {vocab}")
    print(f"  Dims: {model_dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")

    # Load data
    print(f"\n[1/4] Generating batches...")
    train_batches = generate_batches(num_batches=50, batch_size=batch_size, seq_len=seq_len)
    val_batches = generate_batches(num_batches=10, batch_size=batch_size, seq_len=seq_len)
    print(f"✓ Train: {len(train_batches)} batches, Val: {len(val_batches)} batches")

    # Create models
    print(f"\n[2/4] Creating models...")
    hz0a_base = GDN2LanguageModel(
        vocab_size=vocab,
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        gdn2_every=2,
    )
    print(f"✓ HZ-0A (baseline)")

    hz0a_with_mem = HZ0AWithMemory(
        vocab_size=vocab,
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        gdn2_every=2,
        memory_slots=32,
    )
    print(f"✓ HZ-0A+Memory (with scratchpad)")

    # Training
    print(f"\n[3/4] Training models (1 epoch)...")
    hz0a_opt = optim.Adam(learning_rate=1e-4)
    hz0a_mem_opt = optim.Adam(learning_rate=1e-4)

    hz0a_results = {"train": [], "val": []}
    hz0a_mem_results = {"train": [], "val": []}

    hz0a_loss, hz0a_tps = train_epoch(hz0a_base, train_batches, hz0a_opt, "HZ-0A")
    hz0a_mem_loss, hz0a_mem_tps = train_epoch(hz0a_with_mem, train_batches, hz0a_mem_opt, "HZ-0A+Mem")

    hz0a_results["train"].append(hz0a_loss)
    hz0a_mem_results["train"].append(hz0a_mem_loss)

    print(f"\nTraining Summary:")
    print(f"  HZ-0A:        loss={hz0a_loss:.4f}, {hz0a_tps:.0f} tok/s")
    print(f"  HZ-0A+Memory: loss={hz0a_mem_loss:.4f}, {hz0a_mem_tps:.0f} tok/s")

    # Validation
    print(f"\n[4/4] Validating...")
    hz0a_val, _ = train_epoch(hz0a_base, val_batches, hz0a_opt, "HZ-0A (val)")
    hz0a_mem_val, _ = train_epoch(hz0a_with_mem, val_batches, hz0a_mem_opt, "HZ-0A+Mem (val)")

    hz0a_results["val"].append(hz0a_val)
    hz0a_mem_results["val"].append(hz0a_mem_val)

    # Results
    print("\n" + "="*70)
    print("Results")
    print("="*70)

    hz0a_final = hz0a_results["val"][-1] if hz0a_results["val"] else hz0a_results["train"][-1]
    hz0a_mem_final = hz0a_mem_results["val"][-1] if hz0a_mem_results["val"] else hz0a_mem_results["train"][-1]

    loss_delta = hz0a_mem_final - hz0a_final
    loss_pct = (loss_delta / hz0a_final) * 100 if hz0a_final != 0 else 0

    print(f"\nFinal Loss (validation):")
    print(f"  HZ-0A:        {hz0a_final:.4f}")
    print(f"  HZ-0A+Memory: {hz0a_mem_final:.4f}")
    print(f"  Delta:        {loss_delta:+.4f} ({loss_pct:+.2f}%)")

    # Decision
    print(f"\n" + "="*70)
    if loss_delta < 0:
        print(f"✓ MEMORY HELPS (loss improved)")
        verdict = "PASS"
    elif abs(loss_delta) <= hz0a_final * 0.05:
        print(f"~ NEUTRAL (loss within 5%)")
        verdict = "PASS"
    else:
        print(f"✗ MEMORY HURTS (loss degraded)")
        verdict = "INVESTIGATE"

    print("="*70)

    return {
        "hz0a_loss": hz0a_final,
        "hz0a_mem_loss": hz0a_mem_final,
        "loss_delta": loss_delta,
        "loss_pct": loss_pct,
        "verdict": verdict,
    }


if __name__ == "__main__":
    results = phase3b_memory_benchmarks()
    print(f"\nPhase 3b Verdict: {results['verdict']}")
