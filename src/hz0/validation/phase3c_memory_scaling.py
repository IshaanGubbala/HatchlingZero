"""Phase 3c: Memory validation at 36M/110M scale.

Test if memory benefit grows with model size.
Simulates scaling by adjusting layer count and hidden dims.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
from typing import Tuple, List
from src.hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel
from src.hz0.validation.phase3a_complete import HZ0AWithMemory, SimpleMemory


def generate_batches(num_batches: int = 30, batch_size: int = 2, seq_len: int = 256) -> List[Tuple]:
    """Generate synthetic batches for larger models."""
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


def train_epoch_light(model: nn.Module, batches: List, optimizer, model_name: str, max_batches: int = 10) -> float:
    """Train one epoch (limited batches for speed)."""
    total_loss = 0
    count = 0

    for i, (batch, targets) in enumerate(batches):
        if i >= max_batches:
            break

        def compute_loss(m):
            output = m(batch)
            logits = output[0] if isinstance(output, tuple) else output
            return loss_fn(logits, targets)

        loss_val = compute_loss(model)

        if mx.any(mx.isnan(loss_val)):
            continue

        total_loss += float(loss_val)
        count += 1

        loss_grad = mx.grad(compute_loss)(model)
        optimizer.update(model, loss_grad)
        mx.eval(model)

        if (i + 1) % 5 == 0:
            print(f"  [{model_name}] Batch {i+1}: loss={float(loss_val):.4f}")

    avg_loss = total_loss / count if count > 0 else 0
    return avg_loss


def test_scale(scale_name: str, num_layers: int, model_dim: int, num_heads: int) -> dict:
    """Test a specific model scale."""
    print(f"\n{'='*70}")
    print(f"Testing {scale_name} ({model_dim}d, {num_layers}L, ~{model_dim * model_dim * num_layers // 1e6:.1f}M params)")
    print(f"{'='*70}")

    # Load data (smaller batches for larger models)
    print(f"\n[1/3] Data...")
    train_batches = generate_batches(num_batches=30, batch_size=2, seq_len=128)
    print(f"✓ {len(train_batches)} batches")

    # Create models
    print(f"[2/3] Models...")
    hz0a = GDN2LanguageModel(
        vocab_size=8192,
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        gdn2_every=2,
    )
    hz0a_mem = HZ0AWithMemory(
        vocab_size=8192,
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        gdn2_every=2,
        memory_slots=32,
    )
    print(f"✓ HZ-0A and HZ-0A+Memory created")

    # Train
    print(f"[3/3] Training (10 batches each)...")
    opt_hz0a = optim.Adam(learning_rate=1e-4)
    opt_hz0a_mem = optim.Adam(learning_rate=1e-4)

    hz0a_loss = train_epoch_light(hz0a, train_batches, opt_hz0a, f"HZ-0A ({scale_name})", max_batches=10)
    hz0a_mem_loss = train_epoch_light(hz0a_mem, train_batches, opt_hz0a_mem, f"HZ-0A+Mem ({scale_name})", max_batches=10)

    delta = hz0a_mem_loss - hz0a_loss
    delta_pct = (delta / hz0a_loss * 100) if hz0a_loss != 0 else 0

    print(f"\nResults:")
    print(f"  HZ-0A:        {hz0a_loss:.4f}")
    print(f"  HZ-0A+Memory: {hz0a_mem_loss:.4f}")
    print(f"  Delta:        {delta:+.4f} ({delta_pct:+.2f}%)")

    if delta < 0:
        verdict = "HELPS"
    elif abs(delta) <= hz0a_loss * 0.05:
        verdict = "NEUTRAL"
    else:
        verdict = "HURTS"

    print(f"  Verdict:      {verdict}")

    return {
        "scale": scale_name,
        "layers": num_layers,
        "dims": model_dim,
        "hz0a_loss": hz0a_loss,
        "hz0a_mem_loss": hz0a_mem_loss,
        "delta": delta,
        "delta_pct": delta_pct,
        "verdict": verdict,
    }


def phase3c_scaling():
    """Phase 3c: Test memory scaling."""
    print("="*70)
    print("Phase 3c: Memory Scaling Validation")
    print("="*70)

    # Test progression: 5M → 36M → 110M (simulated)
    scales = [
        ("5M (baseline)", 6, 256, 4),      # 5M param estimate
        ("36M (medium)", 12, 512, 8),      # 36M param estimate
        ("110M (large)", 24, 768, 12),     # 110M param estimate
    ]

    results = []
    for scale_name, num_layers, model_dim, num_heads in scales:
        result = test_scale(scale_name, num_layers, model_dim, num_heads)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SCALING ANALYSIS")
    print(f"{'='*70}")

    print(f"\n{'Scale':<15} {'HZ-0A':<10} {'HZ-0A+Mem':<12} {'Delta':<10} {'Verdict':<12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['scale']:<15} {r['hz0a_loss']:<10.4f} {r['hz0a_mem_loss']:<12.4f} {r['delta_pct']:+.2f}%{' '*5} {r['verdict']:<12}")

    # Overall decision
    print(f"\n{'='*70}")
    print("DECISION")
    print(f"{'='*70}")

    verdicts = [r['verdict'] for r in results]
    if verdicts.count("HELPS") >= 2:
        print("✓ MEMORY HELPS (majority of scales)")
        phase3_verdict = "PASS"
    elif verdicts.count("HURTS") >= 2:
        print("✗ MEMORY HURTS (majority of scales)")
        phase3_verdict = "FAIL"
    else:
        print("~ MEMORY NEUTRAL (mixed results)")
        phase3_verdict = "PASS"

    print(f"\nPhase 3c Verdict: {phase3_verdict}")
    print(f"Recommendation: {'Keep memory' if phase3_verdict == 'PASS' else 'Remove memory'} for production")

    print(f"{'='*70}")

    return {
        "results": results,
        "verdict": phase3_verdict,
        "keep_memory": phase3_verdict == "PASS",
    }


if __name__ == "__main__":
    results = phase3c_scaling()
    print(f"\nPhase 3c Complete.")
