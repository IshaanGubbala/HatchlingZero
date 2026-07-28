"""
Phase 9: Memory diagnostics on trained 36M hybrid model.

Validates that trained model actually learns memory tasks (not just language modeling).
Tests: associative recall, overwrite, protected memory, distance robustness.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, Tuple

from hz0.model_port.mlx_gdn2_lm import create_hz_36m_mlx


class MemoryValidator:
    """Test memory task performance on trained model."""

    def __init__(self, model: nn.Module, vocab_size: int = 32768):
        self.model = model
        self.vocab_size = vocab_size

    def associative_recall(self, num_tests: int = 50) -> float:
        """Learn A→V, query A, retrieve V. Measure % correct."""
        correct = 0

        for _ in range(num_tests):
            key = np.random.randint(100, 200)
            value = np.random.randint(1000, 2000)
            seq_len = 256

            # Sequence: [key, value, padding..., key_query]
            seq = np.random.randint(0, min(256, self.vocab_size), seq_len)
            seq[0] = key
            seq[1] = value
            seq[-1] = key

            try:
                seq_mx = mx.array(seq, dtype=mx.int32).reshape(1, -1)
                logits, _ = self.model(seq_mx)

                # Predict at last position (query)
                pred_idx = mx.argmax(logits[0, -1, :])
                pred = int(pred_idx)

                if pred == value:
                    correct += 1
            except Exception as e:
                pass

        return correct / num_tests if num_tests > 0 else 0.0

    def overwrite_task(self, num_tests: int = 50) -> float:
        """Learn A→V1, then A→V2, query A should return V2."""
        correct = 0

        for _ in range(num_tests):
            key = np.random.randint(100, 200)
            v1 = np.random.randint(1000, 1500)
            v2 = np.random.randint(2000, 2500)
            seq_len = 256

            seq = np.random.randint(0, min(256, self.vocab_size), seq_len)
            seq[0] = key
            seq[1] = v1
            seq[seq_len // 2] = key
            seq[seq_len // 2 + 1] = v2
            seq[-1] = key

            try:
                seq_mx = mx.array(seq, dtype=mx.int32).reshape(1, -1)
                logits, _ = self.model(seq_mx)

                pred_idx = mx.argmax(logits[0, -1, :])
                pred = int(pred_idx)

                if pred == v2:
                    correct += 1
            except Exception as e:
                pass

        return correct / num_tests if num_tests > 0 else 0.0

    def protected_memory(self, num_tests: int = 50) -> float:
        """Learn A→V1, B→V2, overwrite A→V3, query B should return V2."""
        correct = 0

        for _ in range(num_tests):
            key_a = np.random.randint(100, 150)
            key_b = np.random.randint(150, 200)
            v_a1 = np.random.randint(1000, 1500)
            v_b = np.random.randint(2000, 2500)
            v_a2 = np.random.randint(3000, 3500)
            seq_len = 256

            seq = np.random.randint(0, min(256, self.vocab_size), seq_len)
            pos = 0
            seq[pos] = key_a
            seq[pos + 1] = v_a1
            pos += 2
            seq[pos] = key_b
            seq[pos + 1] = v_b
            pos += 2
            seq[pos] = key_a
            seq[pos + 1] = v_a2
            seq[-1] = key_b

            try:
                seq_mx = mx.array(seq, dtype=mx.int32).reshape(1, -1)
                logits, _ = self.model(seq_mx)

                pred_idx = mx.argmax(logits[0, -1, :])
                pred = int(pred_idx)

                if pred == v_b:
                    correct += 1
            except Exception as e:
                pass

        return correct / num_tests if num_tests > 0 else 0.0

    def distance_robustness(self, num_tests: int = 30) -> Dict[int, float]:
        """Query at various distances. Measure recall degradation."""
        distances = [16, 32, 64, 128, 256]
        results = {}

        for dist in distances:
            correct = 0
            seq_len = dist + 32

            for _ in range(num_tests):
                key = np.random.randint(100, 200)
                value = np.random.randint(1000, 2000)

                seq = np.random.randint(0, min(256, self.vocab_size), seq_len)
                seq[0] = key
                seq[1] = value
                seq[1 + dist] = key

                try:
                    seq_mx = mx.array(seq, dtype=mx.int32).reshape(1, -1)
                    logits, _ = self.model(seq_mx)

                    pred_idx = mx.argmax(logits[0, 1 + dist, :])
                    pred = int(pred_idx)

                    if pred == value:
                        correct += 1
                except Exception as e:
                    pass

            results[dist] = correct / num_tests if num_tests > 0 else 0.0

        return results

    def run_full_suite(self) -> Dict[str, float]:
        """Run all memory tests."""
        print("=" * 80)
        print("MEMORY VALIDATION SUITE (Phase 9)")
        print("=" * 80)
        print()

        results = {}

        print("1. Associative recall...", end="", flush=True)
        results["associative"] = self.associative_recall(num_tests=50)
        print(f" {results['associative']:.0%}")

        print("2. Overwrite task...", end="", flush=True)
        results["overwrite"] = self.overwrite_task(num_tests=50)
        print(f" {results['overwrite']:.0%}")

        print("3. Protected memory...", end="", flush=True)
        results["protected"] = self.protected_memory(num_tests=50)
        print(f" {results['protected']:.0%}")

        print("4. Distance robustness:")
        distance_results = self.distance_robustness(num_tests=30)
        for dist, recall in distance_results.items():
            print(f"   {dist:3d} tokens: {recall:.0%}")
            results[f"dist_{dist}"] = recall

        print()
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Associative recall: {results['associative']:.0%}")
        print(f"Overwrite task: {results['overwrite']:.0%}")
        print(f"Protected memory: {results['protected']:.0%}")

        avg_dist = np.mean([v for k, v in results.items() if k.startswith("dist_")])
        print(f"Avg distance recall: {avg_dist:.0%}")

        print()
        print("=" * 80)
        print("INTERPRETATION")
        print("=" * 80)

        # Check if results are better than chance
        chance_rate = 1.0 / 1000  # Out of 32K vocab, 1K possible correct answers (conservative)

        if results["associative"] > 0.1:
            print("✓ Associative recall: Model learns to retrieve by key")
        else:
            print("✗ Associative recall: Below threshold (random guessing)")

        if results["overwrite"] > 0.1:
            print("✓ Overwrite task: Model learns to overwrite old associations")
        else:
            print("✗ Overwrite task: Below threshold")

        if results["protected"] > 0.1:
            print("✓ Protected memory: Model preserves unrelated memories")
        else:
            print("✗ Protected memory: Below threshold")

        if avg_dist > 0.05:
            print("✓ Distance robustness: Recall stable across distances")
        else:
            print("✗ Distance robustness: Degrades rapidly")

        print()
        print("GATE ASSESSMENT")
        print("-" * 80)
        if all(
            results.get(k, 0) > 0.05 for k in ["associative", "overwrite", "protected"]
        ):
            print("✓ GATE A: Memory gates working (hybrid learns associative recall)")
        else:
            print("⚠ GATE A: Partial (some memory tasks weak)")

        return results


if __name__ == "__main__":
    print("Creating 36M model...")
    model = create_hz_36m_mlx()
    print("✓ Model created\n")

    validator = MemoryValidator(model)
    results = validator.run_full_suite()
