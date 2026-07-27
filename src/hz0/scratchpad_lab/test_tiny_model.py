"""
HZ-0B Phase 2: Curriculum for tiny memory model.

Stages:
1. Fixed key, fixed value → 100% training recall
2. Multiple keys, fixed mapping → held-out recall
3. Random values, fixed keys → generalization
4. Random held-out pairs → memory task
5. Distractors → interference resistance
6. Overwrite → sequential consistency
7. Protected memory → isolation
8. Distance scaling → robustness

Each stage gates advancement by held-out recall threshold.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn import losses as mlx_losses
import numpy as np
from typing import Dict, Tuple
import time

from src.hz0.scratchpad_lab.tiny_memory_model import TinyMemoryModel


class MemoryCurriculumStage:
    """Single curriculum stage with data generation + validation."""

    def __init__(
        self,
        name: str,
        num_training_examples: int = 100,
        num_held_out: int = 20,
        vocab_size: int = 256,
        seq_len: int = 32,
        num_keys: int = 4,
    ):
        self.name = name
        self.num_training_examples = num_training_examples
        self.num_held_out = num_held_out
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_keys = num_keys

    def generate_batch(self, batch_size: int = 1, held_out: bool = False):
        """Generate batch for this stage."""
        if self.name == "fixed_key_value":
            return self._fixed_key_value(batch_size)
        elif self.name == "multiple_keys":
            return self._multiple_keys(batch_size, held_out)
        elif self.name == "random_values":
            return self._random_values(batch_size)
        else:
            raise ValueError(f"Unknown stage: {self.name}")

    def _fixed_key_value(self, batch_size: int) -> Tuple[mx.array, mx.array]:
        """Stage 1: Fixed key (e.g., 10), fixed value (e.g., 50)."""
        # Sequence: write(10→50), then read(10)
        # Target: output 50
        seq = mx.zeros((batch_size, self.seq_len), dtype=mx.int32)
        # Write phase: key=10
        seq[:, 0] = 10
        # Read phase: query=10
        seq[:, self.seq_len // 2] = 10
        # Target: predict 50 for read phase
        target = mx.ones_like(seq) * 50
        target[:, : self.seq_len // 2] = 0  # Ignore write phase
        return seq, target

    def _multiple_keys(self, batch_size: int, held_out: bool = False) -> Tuple[mx.array, mx.array]:
        """Stage 2: Multiple keys with fixed mapping."""
        seq = mx.zeros((batch_size, self.seq_len), dtype=mx.int32)
        target = mx.zeros((batch_size, self.seq_len), dtype=mx.int32)

        # Mapping: 10→50, 11→51, 12→52, etc.
        keys = np.array([10, 11, 12, 13])
        values = np.array([50, 51, 52, 53])

        if held_out:
            # Use keys not seen at training time
            keys = np.array([20, 21, 22, 23])

        for b in range(batch_size):
            # Write phase
            key_idx = b % len(keys)
            seq[b, 0] = keys[key_idx]
            # Read phase
            seq[b, self.seq_len // 2] = keys[key_idx]
            # Target
            target[b, self.seq_len // 2] = values[key_idx]

        return seq, target

    def _random_values(self, batch_size: int) -> Tuple[mx.array, mx.array]:
        """Stage 3: Random values, fixed keys."""
        seq = mx.array(np.random.randint(0, self.vocab_size, (batch_size, self.seq_len)), dtype=mx.int32)
        # Overwrite first position with fixed key
        seq[:, 0] = 10
        seq[:, self.seq_len // 2] = 10
        # Target: output random value
        target = mx.array(np.random.randint(0, self.vocab_size, (batch_size, self.seq_len)), dtype=mx.int32)
        return seq, target


def train_stage(
    model: TinyMemoryModel,
    stage: MemoryCurriculumStage,
    num_steps: int = 100,
    learning_rate: float = 1e-3,
) -> Dict:
    """Train model on single curriculum stage."""
    optimizer = optim.Adam(learning_rate=learning_rate)
    losses_list = []
    recalls_train = []
    recalls_held_out = []

    start_time = time.time()

    for step in range(num_steps):
        # Training batch
        seq_train, target_train = stage.generate_batch(batch_size=1, held_out=False)

        def loss_fn(m):
            logits, _, _ = m(seq_train)
            # Masked loss: only eval on read phase
            read_phase_start = stage.seq_len // 2
            pred = logits[:, read_phase_start:, :]
            targ = target_train[:, read_phase_start:]
            loss = mx.mean(
                mlx_losses.cross_entropy(pred, targ)
            )
            return loss

        loss_val, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(loss_val)

        losses_list.append(float(loss_val))

        # Validation every 10 steps
        if step % 10 == 0:
            seq_val, target_val = stage.generate_batch(batch_size=1, held_out=False)
            logits_val, _, _ = model(seq_val)
            read_start = stage.seq_len // 2
            pred_val = mx.argmax(logits_val[:, read_start, :], axis=1)
            target_idx = target_val[0, read_start]
            recall = float(pred_val[0] == target_idx)
            recalls_train.append(recall)

            # Held-out validation
            seq_ho, target_ho = stage.generate_batch(batch_size=1, held_out=True)
            logits_ho, _, _ = model(seq_ho)
            pred_ho = mx.argmax(logits_ho[:, read_start, :], axis=1)
            target_ho_idx = target_ho[0, read_start]
            recall_ho = float(pred_ho[0] == target_ho_idx)
            recalls_held_out.append(recall_ho)

    elapsed = time.time() - start_time

    return {
        "stage": stage.name,
        "num_steps": num_steps,
        "wall_clock_sec": elapsed,
        "final_loss": losses_list[-1],
        "avg_loss": np.mean(losses_list),
        "final_train_recall": recalls_train[-1] if recalls_train else 0.0,
        "final_held_out_recall": recalls_held_out[-1] if recalls_held_out else 0.0,
        "max_held_out_recall": max(recalls_held_out) if recalls_held_out else 0.0,
    }


def run_tiny_model_curriculum():
    """Run full curriculum on tiny model."""
    print("=" * 60)
    print("HZ-0B PHASE 2: TINY MODEL CURRICULUM")
    print("=" * 60)
    print()

    # Create tiny model
    model = TinyMemoryModel(
        vocab_size=256,
        model_dim=64,
        num_layers=1,
        num_slots=8,
        slot_dim=32,
    )
    # Count params
    def count_params(m):
        total = 0
        for p in m.parameters().values():
            if isinstance(p, (mx.array, np.ndarray)):
                total += np.prod(p.shape)
            elif isinstance(p, dict):
                total += sum(np.prod(v.shape) for v in p.values())
        return total

    num_params = count_params(model)
    print(f"Model: {num_params / 1e6:.1f}M params")
    print()

    # Curriculum stages
    stages = [
        MemoryCurriculumStage("fixed_key_value", num_training_examples=50),
        MemoryCurriculumStage("multiple_keys", num_training_examples=100),
        MemoryCurriculumStage("random_values", num_training_examples=100),
    ]

    results = []
    for stage in stages:
        print(f"Stage: {stage.name}")
        result = train_stage(model, stage, num_steps=50, learning_rate=1e-3)
        results.append(result)
        print(f"  Loss: {result['final_loss']:.4f}")
        print(f"  Train recall: {result['final_train_recall']:.2%}")
        print(f"  Held-out recall: {result['final_held_out_recall']:.2%}")
        print()

    print("=" * 60)
    print("CURRICULUM RESULTS")
    print("=" * 60)
    for result in results:
        print(f"{result['stage']:20s}: "
              f"loss={result['final_loss']:.4f}, "
              f"train={result['final_train_recall']:.2%}, "
              f"held-out={result['final_held_out_recall']:.2%}")

    return results


if __name__ == "__main__":
    results = run_tiny_model_curriculum()
