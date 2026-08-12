"""HZ-Core-1 real state-memory measurement for the value-bottleneck +
INT8 arms, mirroring scripts/hz0h_state_memory_analysis.py's exact-BDH
methodology exactly (same output format, same real-tensor-byte-count
approach via init_bdh_vb_states / init_bdh_vb_states_int8, not a
formula) so the two are directly comparable. Kept as a separate,
textually parallel file rather than adding an --architecture flag to
the existing script, matching this session's established convention
(see hz0h_stage2_runner_bdh.py vs hz0h_stage2_runner_bdh_vb.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_torch import BDHVB, init_bdh_vb_states, init_bdh_vb_states_int8
from reference.hz0h_state_v1 import hz_state_v1_config


def state_bytes_per_batch_item_fp32(config, dtype_bytes: int = 4) -> int:
    model = BDHVB(config)
    states = init_bdh_vb_states(model, batch_size=1)
    return sum(state.numel() for state in states) * dtype_bytes


def state_bytes_per_batch_item_int8(config) -> int:
    model = BDHVB(config)
    states = init_bdh_vb_states_int8(model, batch_size=1)
    # Real bytes: the quantized tensor itself (1 byte/element, int8) plus
    # the per-tensor scale scalar (fp32, 4 bytes) -- not just the int8
    # payload alone, to avoid understating the real footprint.
    return sum(state["q"].numel() * 1 + state["scale"].numel() * 4 for state in states)


def crossover_context_length(state_bytes: int, n_embd: int, n_layer: int, dtype_bytes: int = 4) -> float:
    """Context length at which a real KV-cache (n_layer * 2 * context *
    D * dtype_bytes) would use as much memory as this state."""
    return state_bytes / (2 * n_embd * n_layer * dtype_bytes)


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

    config = hz_state_v1_config(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size)
    model = BDHVB(config)
    param_bytes = sum(p.numel() for p in model.parameters()) * 4
    per_batch_fp32 = state_bytes_per_batch_item_fp32(config)
    per_batch_int8 = state_bytes_per_batch_item_int8(config)
    crossover_fp32 = crossover_context_length(per_batch_fp32, args.n_embd, args.n_layer)
    crossover_int8 = crossover_context_length(per_batch_int8, args.n_embd, args.n_layer)

    print(f"d_state={config.d_state}")
    print(f"parameter_count={sum(p.numel() for p in model.parameters()):,}")
    print(f"model_weight_bytes(fp32)={_format_bytes(param_bytes)}")
    print(f"state_bytes_per_batch_item(vb_fp32)={_format_bytes(per_batch_fp32)}")
    print(f"state_bytes_per_batch_item(vb_int8)={_format_bytes(per_batch_int8)}")
    print(f"state_to_weights_ratio(vb_fp32)={per_batch_fp32 / param_bytes:.3f}x")
    print(f"state_to_weights_ratio(vb_int8)={per_batch_int8 / param_bytes:.3f}x")
    print(f"kv_cache_crossover_context_length(vb_fp32)={crossover_fp32:,.0f} tokens")
    print(f"kv_cache_crossover_context_length(vb_int8)={crossover_int8:,.0f} tokens")
    for batch_size in (int(b) for b in args.batch_sizes.split(",") if b.strip()):
        print(f"state_bytes(vb_fp32,batch={batch_size})={_format_bytes(per_batch_fp32 * batch_size)}")
        print(f"state_bytes(vb_int8,batch={batch_size})={_format_bytes(per_batch_int8 * batch_size)}")


if __name__ == "__main__":
    main()
