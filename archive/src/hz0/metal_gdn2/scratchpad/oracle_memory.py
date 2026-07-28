"""
Oracle memory tests for GDN-2 scratchpad.

Validates:
- Associative recall
- Overwrite and protected memory
- Multi-key interference
- Recall vs distance
- State reset
"""

import mlx.core as mx
import numpy as np
from typing import Tuple, List, Dict


class OracleMemoryTest:
    """
    Deterministic routing to measure storage/retrieval.
    Uses hash(key) % num_slots to avoid learning routing.
    """

    def __init__(self, num_slots: int, slot_dim: int, vocab_size: int = 256):
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.vocab_size = vocab_size
        self.memory = mx.zeros((num_slots, slot_dim))

    def _route(self, key: int) -> int:
        """Deterministic routing: hash to slot."""
        return hash(key) % self.num_slots

    def write(self, key: int, value: mx.array) -> None:
        """Store value at hashed slot."""
        slot = self._route(key)
        self.memory[slot] = value

    def read(self, key: int) -> mx.array:
        """Retrieve from hashed slot."""
        slot = self._route(key)
        return self.memory[slot]

    def reset(self) -> None:
        """Clear all memory."""
        self.memory = mx.zeros_like(self.memory)


def test_associative_recall(
    num_pairs: int = 4,
    slot_dim: int = 8,
) -> Dict[str, float]:
    """
    Write key-value pairs, then query keys.
    Return accuracy.
    """
    test = OracleMemoryTest(num_slots=16, slot_dim=slot_dim)

    # Write phase
    np.random.seed(42)
    keys = np.arange(num_pairs)
    values = {}

    for k in keys:
        value = mx.array(np.random.randn(slot_dim).astype(np.float32))
        test.write(int(k), value)
        values[k] = value

    # Query phase: measure reconstruction error
    errors = []
    for k in keys:
        retrieved = test.read(int(k))
        original = values[k]
        error = float(mx.mean((retrieved - original) ** 2))
        errors.append(error)

    # Perfect oracle: error should be zero (exact match)
    mean_error = np.mean(errors)

    return {
        "mean_reconstruction_error": mean_error,
        "max_reconstruction_error": np.max(errors),
        "perfect_match_count": sum(1 for e in errors if e < 1e-5),
        "perfect_match_rate": sum(1 for e in errors if e < 1e-5) / len(errors),
    }


def test_overwrite(
    slot_dim: int = 8,
) -> Dict[str, float]:
    """
    Write A→red, then overwrite A→green.
    Query A, should retrieve green.
    """
    test = OracleMemoryTest(num_slots=16, slot_dim=slot_dim)

    np.random.seed(42)

    # Initial write
    key = 1
    value_red = mx.array(np.array([1.0] + [0.0] * (slot_dim - 1), dtype=np.float32))
    test.write(key, value_red)

    # Overwrite
    value_green = mx.array(np.array([0.0, 1.0] + [0.0] * (slot_dim - 2), dtype=np.float32))
    test.write(key, value_green)

    # Query
    retrieved = test.read(key)

    # Measure: should match green, not red
    green_error = float(mx.mean((retrieved - value_green) ** 2))
    red_error = float(mx.mean((retrieved - value_red) ** 2))

    return {
        "overwrite_success": 1.0 if green_error < red_error else 0.0,
        "green_reconstruction_error": green_error,
        "red_reconstruction_error": red_error,
    }


def test_protected_unrelated_memory(
    slot_dim: int = 8,
) -> Dict[str, float]:
    """
    Write A→red, B→blue.
    Overwrite A→green.
    Query B, should still be blue.
    """
    test = OracleMemoryTest(num_slots=16, slot_dim=slot_dim)

    np.random.seed(42)

    # Write A and B
    value_red = mx.array(np.array([1.0] + [0.0] * (slot_dim - 1), dtype=np.float32))
    value_blue = mx.array(np.array([0.0, 1.0] + [0.0] * (slot_dim - 2), dtype=np.float32))

    test.write(1, value_red)
    test.write(2, value_blue)

    # Overwrite A
    value_green = mx.array(np.array([0.0, 0.0, 1.0] + [0.0] * (slot_dim - 3), dtype=np.float32))
    test.write(1, value_green)

    # Query B
    retrieved = test.read(2)

    # Should match blue, not green
    blue_error = float(mx.mean((retrieved - value_blue) ** 2))
    green_error = float(mx.mean((retrieved - value_green) ** 2))

    return {
        "protected_success": 1.0 if blue_error < green_error else 0.0,
        "blue_reconstruction_error": blue_error,
        "green_interference_error": green_error,
    }


def test_recall_vs_distance(
    num_keys: int = 8,
    slot_dim: int = 8,
) -> Dict[int, float]:
    """
    Write keys, then add distractors at increasing distances.
    Track whether target is recalled.
    """
    recall_by_distance = {}
    distances = [1, 2, 4, 8, 16]

    for dist in distances:
        test = OracleMemoryTest(num_slots=32, slot_dim=slot_dim)

        np.random.seed(42)

        # Write target key
        target_key = 0
        target_value = mx.array(np.random.randn(slot_dim).astype(np.float32))
        test.write(target_key, target_value)

        # Add distractors at this distance
        for i in range(1, dist + 1):
            distractor = mx.array(np.random.randn(slot_dim).astype(np.float32))
            test.write(i, distractor)

        # Query target
        retrieved = test.read(target_key)
        error = float(mx.mean((retrieved - target_value) ** 2))
        recalled = 1.0 if error < 1e-5 else 0.0

        recall_by_distance[dist] = recalled

    return recall_by_distance


def test_multi_key_interference(
    num_distractors: int = 4,
    slot_dim: int = 8,
) -> Dict[str, float]:
    """
    Write target key, then write many distractors.
    Query target, measure whether it's recoverable.
    """
    test = OracleMemoryTest(num_slots=16, slot_dim=slot_dim)

    np.random.seed(42)

    # Write target
    target_key = 0
    target_value = mx.array(np.random.randn(slot_dim).astype(np.float32))
    test.write(target_key, target_value)

    # Write distractors
    for i in range(1, num_distractors + 1):
        distractor = mx.array(np.random.randn(slot_dim).astype(np.float32))
        test.write(i, distractor)

    # Query target
    retrieved = test.read(target_key)

    error = float(mx.mean((retrieved - target_value) ** 2))
    recalled = 1.0 if error < 1e-5 else 0.0

    return {
        "interference_recall_accuracy": recalled,
        "target_reconstruction_error": error,
        "num_distractors": num_distractors,
    }


def test_state_reset_isolation(
    slot_dim: int = 8,
) -> Dict[str, float]:
    """
    Session 1: write A→red
    Reset
    Session 2: query A, should get zeros (not red)
    """
    test = OracleMemoryTest(num_slots=16, slot_dim=slot_dim)

    np.random.seed(42)

    # Session 1
    value_red = mx.array(np.array([1.0] + [0.0] * (slot_dim - 1), dtype=np.float32))
    test.write(1, value_red)

    # Reset
    test.reset()

    # Session 2: query should get zero
    retrieved = test.read(1)
    zeros = mx.zeros(slot_dim)

    error_to_zero = float(mx.mean((retrieved - zeros) ** 2))
    error_to_red = float(mx.mean((retrieved - value_red) ** 2))

    return {
        "state_isolation_success": 1.0 if error_to_zero < error_to_red else 0.0,
        "zero_reconstruction_error": error_to_zero,
        "red_contamination_error": error_to_red,
    }


def run_all_diagnostics(slot_dim: int = 8) -> Dict[str, Dict]:
    """Run complete memory diagnostic suite."""
    results = {
        "associative_recall": test_associative_recall(slot_dim=slot_dim),
        "overwrite": test_overwrite(slot_dim=slot_dim),
        "protected_memory": test_protected_unrelated_memory(slot_dim=slot_dim),
        "recall_vs_distance": test_recall_vs_distance(slot_dim=slot_dim),
        "interference": test_multi_key_interference(slot_dim=slot_dim),
        "state_reset": test_state_reset_isolation(slot_dim=slot_dim),
    }
    return results
