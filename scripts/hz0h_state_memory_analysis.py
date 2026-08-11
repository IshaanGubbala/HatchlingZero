"""HZ Phase 2 (plans/HatchlingZero_Reality_Plan.md): reproduces the real,
measured BDH synaptic-state memory footprint from
docs/restart/hz0h_phase2_streaming_state_size_results.md -- state bytes
vs. model weight bytes, at several batch sizes, plus the crossover
context length where a real Transformer KV-cache would use the same
amount of memory as BDH's fixed-size state. Not estimated from a
formula alone -- uses reference/hz0h_bdh_torch.py's own
init_bdh_states to get real tensor byte counts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig, init_bdh_states


def state_bytes_per_batch_item(config: BDHConfig, dtype_bytes: int = 4) -> int:
    model = BDH(config)
    states = init_bdh_states(model, batch_size=1)
    return sum(state.numel() for state in states) * dtype_bytes


def crossover_context_length(config: BDHConfig, dtype_bytes: int = 4) -> float:
    """Context length at which a real KV-cache (n_layer * 2 * context *
    D * dtype_bytes) would use as much memory as BDH's fixed state."""
    state_bytes = state_bytes_per_batch_item(config, dtype_bytes)
    return state_bytes / (2 * config.n_embd * config.n_layer * dtype_bytes)


def _format_bytes(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    return f"{n / 1e6:.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--n-head", type=int, required=True)
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--batch-sizes", type=str, default="1,8,32")
    args = parser.parse_args()

    config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0)
    model = BDH(config)
    param_bytes = sum(p.numel() for p in model.parameters()) * 4
    per_batch = state_bytes_per_batch_item(config)
    crossover = crossover_context_length(config)

    print(f"parameter_count={sum(p.numel() for p in model.parameters()):,}")
    print(f"model_weight_bytes(fp32)={_format_bytes(param_bytes)}")
    print(f"state_bytes_per_batch_item(fp32)={_format_bytes(per_batch)}")
    print(f"state_to_weights_ratio={per_batch / param_bytes:.2f}x")
    print(f"kv_cache_crossover_context_length={crossover:,.0f} tokens")
    for batch_size in (int(b) for b in args.batch_sizes.split(",") if b.strip()):
        print(f"state_bytes(batch={batch_size})={_format_bytes(per_batch * batch_size)}")


if __name__ == "__main__":
    main()
