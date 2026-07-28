"""
HZ-0B Phase 3: Memory diagnostics - hard-route logging + slot-match metrics.

Tracks:
- write_slot = argmax(write_scores)
- read_slot = argmax(read_scores)
- route_match = write_slot == read_slot
- Slot occupancy
- Slot collision rate
- Dead slots
- Value reconstruction error
- Recall conditional on route match
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class MemoryDiagnostics:
    """Single-step diagnostics for memory access."""

    write_slot: int
    read_slot: int
    write_scores: mx.array  # [num_slots]
    read_scores: mx.array
    key: mx.array
    value: mx.array
    retrieved: mx.array
    state_before: mx.array
    state_after: mx.array

    @property
    def route_match(self) -> bool:
        """Write and read routed to same slot."""
        return self.write_slot == self.read_slot

    @property
    def write_confidence(self) -> float:
        """Max score / sum scores."""
        scores = self.write_scores
        return float(mx.max(scores) / (mx.sum(scores) + 1e-8))

    @property
    def read_confidence(self) -> float:
        """Max score / sum scores."""
        scores = self.read_scores
        return float(mx.max(scores) / (mx.sum(scores) + 1e-8))

    @property
    def value_reconstruction_error(self) -> float:
        """L2 distance between stored and retrieved value."""
        error = mx.sqrt(mx.sum((self.retrieved - self.value) ** 2) + 1e-8)
        return float(error)


class DiagnosticsCollector:
    """Accumulate diagnostics over sequence."""

    def __init__(self):
        self.steps: List[MemoryDiagnostics] = []

    def add(self, diag: MemoryDiagnostics):
        """Record step diagnostics."""
        self.steps.append(diag)

    def summarize(self) -> Dict:
        """Aggregate statistics over sequence."""
        if not self.steps:
            return {}

        route_matches = [float(d.route_match) for d in self.steps]
        write_confs = [d.write_confidence for d in self.steps]
        read_confs = [d.read_confidence for d in self.steps]
        recon_errors = [d.value_reconstruction_error for d in self.steps]
        write_slots = [d.write_slot for d in self.steps]
        read_slots = [d.read_slot for d in self.steps]

        num_slots = len(self.steps[0].write_scores)

        return {
            "route_match_rate": np.mean(route_matches),
            "write_confidence_mean": np.mean(write_confs),
            "write_confidence_std": np.std(write_confs),
            "read_confidence_mean": np.mean(read_confs),
            "read_confidence_std": np.std(read_confs),
            "reconstruction_error_mean": np.mean(recon_errors),
            "reconstruction_error_max": np.max(recon_errors),
            "slot_occupancy": len(set(write_slots)) / num_slots,
            "slot_collision_rate": 1 - (len(set(write_slots)) / len(write_slots)) if write_slots else 0,
            "dead_slots": num_slots - len(set(write_slots)),
            "num_steps": len(self.steps),
        }


class OracleRoutingAblation:
    """Phase 4: Oracle routing - manually assign slots by key hash."""

    def __init__(self, num_slots: int):
        self.num_slots = num_slots

    def get_oracle_slot(self, key: mx.array) -> int:
        """Deterministic slot assignment via hash."""
        # Use first element as key ID
        key_id = int(key[0])
        return key_id % self.num_slots


class OracleStorageAblation:
    """Phase 4: Oracle storage - write correct target embedding directly."""

    def __init__(self, num_slots: int, slot_dim: int, target_embedding_fn=None):
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.target_embedding_fn = target_embedding_fn

    def get_oracle_value(self, target_id: int) -> mx.array:
        """Get ground-truth embedding for target."""
        if self.target_embedding_fn:
            return self.target_embedding_fn(target_id)
        else:
            # Synthetic: hash-based embedding
            rng = np.random.RandomState(target_id)
            return mx.array(rng.randn(self.slot_dim).astype(np.float32))


class OracleReadAblation:
    """Phase 4: Oracle read - learned writes, oracle-routed reads."""

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.oracle_routing = OracleRoutingAblation(num_slots)

    def read_with_oracle_slot(self, state: mx.array, key: mx.array) -> mx.array:
        """Read from oracle-determined slot."""
        slot = self.oracle_routing.get_oracle_slot(key)
        return state[:, slot, :] if state.ndim > 2 else state[slot, :]


def run_ablation_comparison(
    model,
    test_sequence: mx.array,
    test_targets: mx.array,
    oracle_routing: OracleRoutingAblation,
    oracle_storage: OracleStorageAblation,
) -> Dict:
    """
    Run model under different oracle conditions.

    Returns recall under:
    - Oracle routing (both write + read use oracle)
    - Oracle read (learned write, oracle read)
    - Oracle storage (oracle value, learned routing)
    - Baseline (learned everything)
    """
    results = {}

    # Baseline: learned routing + storage
    logits_baseline, _, diags_baseline = model(test_sequence)
    results["baseline"] = {"logits": logits_baseline, "diagnostics": diags_baseline}

    # Oracle routing ablation: manually assign slots
    # (requires modified forward pass - placeholder)
    results["oracle_routing"] = {"note": "requires forward_with_oracle_routing()"}
    results["oracle_read"] = {"note": "requires forward_with_oracle_read()"}
    results["oracle_storage"] = {"note": "requires forward_with_oracle_storage()"}

    return results
