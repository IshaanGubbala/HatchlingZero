"""Phase 7: Memory task training (recall, overwrite, protect).

Train memory on explicit associative tasks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import json
from typing import Dict, Any

from hz0.validation.phase3_memory_redesign import ImprovedScratchpadMemory


def train_recall_task():
    """Train memory on associative recall."""
    print("\n" + "="*70)
    print("Memory Task 1: Associative Recall Training")
    print("="*70)

    memory = ImprovedScratchpadMemory(model_dim=64, num_slots=8, slot_dim=64)
    optimizer = optim.Adam(learning_rate=1e-3)

    B, D = 2, 64
    results = []

    print("\nTraining (20 steps)...")

    for step in range(20):
        # Create key-value pairs
        keys = mx.random.normal((B, D)) * 0.1
        values = mx.random.normal((B, D)) * 0.1

        # Initialize memory
        memory_state = memory.initialize_state(B)

        # Write phase (5 steps)
        for _ in range(5):
            memory_state = memory.write(keys, values, memory_state)

        # Read phase
        def loss_fn(mem):
            memory_read, _ = mem.read(keys, memory_state)
            # Loss: distance to target value
            loss = mx.mean(mx.abs(memory_read - values))
            return loss

        loss, grads = nn.value_and_grad(memory, loss_fn)(memory)
        optimizer.update(memory, grads)

        if (step + 1) % 5 == 0:
            memory_read, _ = memory.read(keys, memory_state)
            dist = float(mx.mean(mx.abs(memory_read - values)))
            print(f"Step {step+1:2d}: loss={float(loss):.4f}, recall_dist={dist:.4f}")
            results.append({"step": step + 1, "loss": float(loss), "recall_dist": dist})

    print("✓ Recall task training complete")
    return results


def train_overwrite_task():
    """Train memory on overwrite."""
    print("\n" + "="*70)
    print("Memory Task 2: Overwrite Training")
    print("="*70)

    memory = ImprovedScratchpadMemory(model_dim=64, num_slots=8, slot_dim=64)
    optimizer = optim.Adam(learning_rate=1e-3)

    B, D = 2, 64
    results = []

    print("\nTraining (20 steps)...")

    for step in range(20):
        # Key + two values (v1 then v2)
        key = mx.ones((B, D)) * 0.1
        val1 = mx.ones((B, D)) * 1.0
        val2 = mx.ones((B, D)) * 2.0

        memory_state = memory.initialize_state(B)

        # Write v1
        for _ in range(3):
            memory_state = memory.write(key, val1, memory_state)

        # Overwrite to v2
        for _ in range(3):
            memory_state = memory.write(key, val2, memory_state)

        # Loss: distance to v2
        def loss_fn(mem):
            memory_read, _ = mem.read(key, memory_state)
            loss = mx.mean(mx.abs(memory_read - val2))
            return loss

        loss, grads = nn.value_and_grad(memory, loss_fn)(memory)
        optimizer.update(memory, grads)

        if (step + 1) % 5 == 0:
            memory_read, _ = memory.read(key, memory_state)
            dist = float(mx.mean(mx.abs(memory_read - val2)))
            print(f"Step {step+1:2d}: loss={float(loss):.4f}, overwrite_dist={dist:.4f}")
            results.append({"step": step + 1, "loss": float(loss), "overwrite_dist": dist})

    print("✓ Overwrite task training complete")
    return results


def train_protect_task():
    """Train memory on protected retention."""
    print("\n" + "="*70)
    print("Memory Task 3: Protected Retention Training")
    print("="*70)

    memory = ImprovedScratchpadMemory(model_dim=64, num_slots=8, slot_dim=64)
    optimizer = optim.Adam(learning_rate=1e-3)

    B, D = 2, 64
    results = []

    print("\nTraining (20 steps)...")

    for step in range(20):
        # Important memory
        key_important = mx.ones((B, D)) * 0.1
        val_important = mx.ones((B, D)) * 5.0

        memory_state = memory.initialize_state(B)

        # Store important
        for _ in range(3):
            memory_state = memory.write(key_important, val_important, memory_state)

        # Interfering writes
        for _ in range(5):
            key_noise = mx.random.normal((B, D))
            val_noise = mx.random.normal((B, D))
            memory_state = memory.write(key_noise, val_noise, memory_state)

        # Loss: distance to important
        def loss_fn(mem):
            memory_read, _ = mem.read(key_important, memory_state)
            loss = mx.mean(mx.abs(memory_read - val_important))
            return loss

        loss, grads = nn.value_and_grad(memory, loss_fn)(memory)
        optimizer.update(memory, grads)

        if (step + 1) % 5 == 0:
            memory_read, _ = memory.read(key_important, memory_state)
            dist = float(mx.mean(mx.abs(memory_read - val_important)))
            print(f"Step {step+1:2d}: loss={float(loss):.4f}, protect_dist={dist:.4f}")
            results.append({"step": step + 1, "loss": float(loss), "protect_dist": dist})

    print("✓ Protect task training complete")
    return results


def main():
    """Run Phase 7 memory training."""
    print("="*70)
    print("Phase 7: Memory Task Training")
    print("="*70)

    all_results = {}

    start = time.time()

    all_results["recall"] = train_recall_task()
    all_results["overwrite"] = train_overwrite_task()
    all_results["protect"] = train_protect_task()

    elapsed = time.time() - start

    # Save results
    output_path = Path("outputs/phase7_memory_training.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("Phase 7 Complete")
    print("="*70)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Results: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
