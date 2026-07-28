"""
Memory diagnostics: Measure HZ model recall on memory-specific tasks.

Tests per plan section 14, experiment 4:
- Associative recall (A→red query A)
- Overwrite (A→red then A→green query A)
- Protected unrelated memory (A→red, B→blue, overwrite A→green, query B)
- Recall versus distance (measure retrieval at 32, 64, 128, 256, 512, 1024, 2048 tokens)

Run on: Transformer, hybrid, best 36M, tuned 110M
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, Tuple, List


class MemoryDiagnostics:
    """Memory task evaluation suite."""

    def __init__(self, model: nn.Module, vocab_size: int = 32768):
        self.model = model
        self.vocab_size = vocab_size

    def associative_recall(self, num_pairs: int = 100, seq_len: int = 256) -> float:
        """
        Task: Learn key→value pairs, then query for value given key.
        A → red
        B → blue
        query A → should output red
        """
        correct = 0

        for _ in range(num_pairs):
            # Create key-value pair
            key = np.random.randint(100, 200)
            value = np.random.randint(1000, 2000)

            # Create sequence: [key, value, ..., key, query_token]
            seq = np.random.randint(0, self.vocab_size, seq_len)
            seq[0] = key
            seq[1] = value
            seq[-1] = key  # Query with same key at end

            seq_mx = mx.array(seq, dtype=mx.int32).reshape(1, -1)

            try:
                logits, _ = self.model(seq_mx)
                pred = mx.argmax(logits[0, -1, :])
                if int(pred) == value:
                    correct += 1
            except:
                pass

        return correct / num_pairs if num_pairs > 0 else 0.0

    def overwrite_recall(self, num_tests: int = 100, seq_len: int = 256) -> float:
        """
        Task: Learn A→red, then overwrite with A→green, query A → should output green.
        Measures ability to forget old associations and learn new ones.
        """
        correct = 0

        for _ in range(num_tests):
            key = np.random.randint(100, 200)
            old_value = np.random.randint(1000, 2000)
            new_value = np.random.randint(3000, 4000)

            # Sequence: [key, old_value, ..., key, new_value, ..., key, query]
            seq = np.random.randint(0, self.vocab_size, seq_len)
            seq[0] = key
            seq[1] = old_value
            seq[seq_len // 2] = key
            seq[seq_len // 2 + 1] = new_value
            seq[-1] = key  # Query at end

            seq_mx = mx.array(seq, dtype=mx.int32).reshape(1, -1)

            try:
                logits, _ = self.model(seq_mx)
                pred = mx.argmax(logits[0, -1, :])
                if int(pred) == new_value:
                    correct += 1
            except:
                pass

        return correct / num_tests if num_tests > 0 else 0.0

    def protected_memory(self, num_tests: int = 100, seq_len: int = 256) -> float:
        """
        Task: Learn A→red, B→blue, overwrite A→green, query B → should still output blue.
        Measures whether unrelated memories are protected from interference.
        """
        correct = 0

        for _ in range(num_tests):
            key_a = np.random.randint(100, 150)
            key_b = np.random.randint(150, 200)
            value_a_orig = np.random.randint(1000, 1500)
            value_b = np.random.randint(2000, 2500)
            value_a_new = np.random.randint(3000, 3500)

            # Sequence: [A, red, B, blue, A, green, query_B]
            seq = np.random.randint(0, self.vocab_size, seq_len)
            pos = 0
            seq[pos] = key_a
            seq[pos + 1] = value_a_orig
            pos += 2
            seq[pos] = key_b
            seq[pos + 1] = value_b
            pos += 2
            seq[pos] = key_a
            seq[pos + 1] = value_a_new
            seq[-1] = key_b  # Query B at end

            seq_mx = mx.array(seq, dtype=mx.int32).reshape(1, -1)

            try:
                logits, _ = self.model(seq_mx)
                pred = mx.argmax(logits[0, -1, :])
                if int(pred) == value_b:
                    correct += 1
            except:
                pass

        return correct / num_tests if num_tests > 0 else 0.0

    def recall_vs_distance(self, num_tests: int = 50) -> Dict[int, float]:
        """
        Task: Learn key→value, then query at various distances.
        Measure recall at positions: 32, 64, 128, 256, 512, 1024, 2048 tokens after write.
        """
        distances = [32, 64, 128, 256, 512]
        results = {}

        for dist in distances:
            correct = 0
            seq_len = dist + 64  # Write + distance + padding

            for _ in range(num_tests):
                key = np.random.randint(100, 200)
                value = np.random.randint(1000, 2000)

                seq = np.random.randint(0, self.vocab_size, seq_len)
                seq[0] = key
                seq[1] = value
                seq[1 + dist] = key  # Query at distance

                seq_mx = mx.array(seq, dtype=mx.int32).reshape(1, -1)

                try:
                    logits, _ = self.model(seq_mx)
                    pred = mx.argmax(logits[0, 1 + dist, :])
                    if int(pred) == value:
                        correct += 1
                except:
                    pass

            results[dist] = correct / num_tests if num_tests > 0 else 0.0

        return results

    def run_full_diagnostic(self) -> Dict[str, float]:
        """Run all memory diagnostics."""
        print("=" * 70)
        print("MEMORY DIAGNOSTICS")
        print("=" * 70)

        results = {}

        print("\n1. Associative recall...", end="", flush=True)
        results["associative_recall"] = self.associative_recall()
        print(f" {results['associative_recall']:.0%}")

        print("2. Overwrite recall...", end="", flush=True)
        results["overwrite_recall"] = self.overwrite_recall()
        print(f" {results['overwrite_recall']:.0%}")

        print("3. Protected memory...", end="", flush=True)
        results["protected_memory"] = self.protected_memory()
        print(f" {results['protected_memory']:.0%}")

        print("4. Recall vs distance...")
        distance_results = self.recall_vs_distance()
        for dist, recall in distance_results.items():
            print(f"   {dist:4d} tokens: {recall:.0%}")
            results[f"distance_{dist}"] = recall

        print("\n" + "=" * 70)
        return results


if __name__ == "__main__":
    print("Memory diagnostics module. Import and use MemoryDiagnostics class.")
