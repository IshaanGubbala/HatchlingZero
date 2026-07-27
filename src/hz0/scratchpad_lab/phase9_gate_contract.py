"""
HZ-0B Phase 9: Gate contract - clear success criteria.

Separate HZ-0A (language backbone) from HZ-0B (memory) gates.
Define what "pass" actually means before shipping.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MemoryGate:
    """Single memory test gate."""
    name: str
    description: str
    threshold: float
    metric: str
    context: str


# HZ-0A gates (language modeling backbone)
HZ0A_GATES = {
    "language_quality": MemoryGate(
        name="language_quality",
        description="Perplexity on held-out text corpus",
        threshold=20.0,
        metric="perplexity",
        context="110M hybrid model, 5B token training",
    ),
    "decode_gap": MemoryGate(
        name="decode_gap",
        description="Single-token decode latency gap (hybrid vs transformer)",
        threshold=0.5,  # <= 2x
        metric="latency_ratio",
        context="Measure prefill + decode on seq_len=1024",
    ),
    "efficiency": MemoryGate(
        name="efficiency",
        description="FLOPs/second efficiency vs A100 baseline",
        threshold=0.8,  # >= 80% of baseline
        metric="efficiency_ratio",
        context="MLX on Mac M2 Pro vs CUDA on A100",
    ),
}

# HZ-0B gates (explicit memory layer)
HZ0B_GATES = {
    "associative_recall": MemoryGate(
        name="associative_recall",
        description="Recall accuracy on held-out key/value pairs",
        threshold=0.95,
        metric="recall",
        context="After curriculum training, 1000 unique pairs",
    ),
    "interference_resistance": MemoryGate(
        name="interference_resistance",
        description="Recall remains >90% with up to 8 concurrent unrelated writes",
        threshold=0.90,
        metric="recall",
        context="Protected memory stage with 8 slots",
    ),
    "overwrite_consistency": MemoryGate(
        name="overwrite_consistency",
        description="Correctly retrieve overwritten values",
        threshold=0.95,
        metric="recall",
        context="Sequential write detection + retrieval",
    ),
    "distance_robustness": MemoryGate(
        name="distance_robustness",
        description="Recall decays gracefully; >80% at max distance",
        threshold=0.80,
        metric="recall",
        context="Read at t/4, t/2, 3t/4 after write at t=0",
    ),
    "routing_consistency": MemoryGate(
        name="routing_consistency",
        description="Write and read route to same slot for identical keys",
        threshold=0.99,
        metric="route_match_rate",
        context="Track argmax(scores) for write vs read",
    ),
    "oracle_isolation": MemoryGate(
        name="oracle_isolation",
        description="Oracle routing shows >80% boost if learned routing fails",
        threshold=0.80,
        metric="recall_boost",
        context="Compare baseline vs oracle_routing on trained model",
    ),
}


class GateContract:
    """Formalize success criteria."""

    def __init__(self):
        self.hz0a = HZ0A_GATES
        self.hz0b = HZ0B_GATES

    def get_status(self, results: Dict[str, float]) -> Dict[str, bool]:
        """Check which gates pass."""
        status = {}

        for gate_name, gate in self.hz0a.items():
            if gate_name in results:
                status[f"hz0a_{gate_name}"] = results[gate_name] <= gate.threshold

        for gate_name, gate in self.hz0b.items():
            if gate_name in results:
                status[f"hz0b_{gate_name}"] = results[gate_name] >= gate.threshold

        return status

    def print_contract(self):
        """Print readable contract."""
        print("=" * 70)
        print("HZ-0A: LANGUAGE BACKBONE GATES")
        print("=" * 70)
        for gate_name, gate in self.hz0a.items():
            print(f"\n{gate_name.upper()}")
            print(f"  Criterion: {gate.description}")
            print(f"  Metric: {gate.metric}")
            print(f"  Threshold: {gate.threshold}")
            print(f"  Context: {gate.context}")

        print("\n" + "=" * 70)
        print("HZ-0B: MEMORY LAYER GATES")
        print("=" * 70)
        for gate_name, gate in self.hz0b.items():
            print(f"\n{gate_name.upper()}")
            print(f"  Criterion: {gate.description}")
            print(f"  Metric: {gate.metric}")
            print(f"  Threshold: >= {gate.threshold}")
            print(f"  Context: {gate.context}")

        print("\n" + "=" * 70)
        print("ARCHITECTURE STAGES")
        print("=" * 70)
        print("""
HZ-0A: Language backbone only
  - GDN-2 recurrent layer
  - Multi-head attention blocks
  - Feed-forward layers
  Status: Baseline language modeling performance

HZ-0B: Add explicit memory layer
  - HZ-0A backbone (frozen or fine-tuned)
  - Slot-addressed scratchpad
  - Routing + storage + readout
  Status: Declarative memory for hallucination reduction + efficiency

HZ-0C (future): Full integration
  - Joint training of backbone + memory
  - Cross-layer attention to scratchpad
  - Unified forward + backward
        """)


if __name__ == "__main__":
    contract = GateContract()
    contract.print_contract()
