#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def count_params(spec: dict) -> dict:
    vocab_size = int(spec["vocab_size"])
    d_model = int(spec["d_model"])
    num_heads = int(spec["num_heads"])
    d_k = int(spec["head_dim_qk"])
    d_v = int(spec["head_dim_v"])
    d_ff = int(spec["d_ff"])
    num_layers = int(spec["num_layers"])
    attention_layers = set(int(i) for i in spec["attention_layer_indices"])
    tie_embeddings = bool(spec["lm_head"]["tied_embeddings"])
    lm_head_bias = bool(spec["lm_head"]["bias"])

    if num_heads * d_k != d_model:
        raise ValueError("num_heads * head_dim_qk must equal d_model for attention blocks")

    if len(attention_layers) >= num_layers:
        raise ValueError("spec must include at least one recurrent layer")

    recurrent_layers = num_layers - len(attention_layers)

    embedding = vocab_size * d_model
    final_norm = d_model
    lm_head_weight = 0 if tie_embeddings else vocab_size * d_model
    lm_head_bias_params = vocab_size if lm_head_bias else 0

    # Recurrent block:
    # in_proj: q, k, v, decay, erase, write
    recurrent_in_features = num_heads * (4 * d_k + 2 * d_v)
    recurrent_in_proj = d_model * recurrent_in_features + recurrent_in_features
    recurrent_out_proj = (num_heads * d_v) * d_model + d_model
    recurrent_norms = 2 * d_model

    # SwiGLU MLP implemented as gate_proj, up_proj, down_proj
    mlp = (
        d_model * d_ff
        + d_ff
        + d_model * d_ff
        + d_ff
        + d_ff * d_model
        + d_model
    )
    recurrent_block = recurrent_in_proj + recurrent_out_proj + recurrent_norms + mlp

    # Attention block: q, k, v, out + same MLP/norm structure
    attention_qkv = d_model * (3 * num_heads * d_k) + (3 * num_heads * d_k)
    attention_out = (num_heads * d_k) * d_model + d_model
    attention_norms = 2 * d_model
    attention_block = attention_qkv + attention_out + attention_norms + mlp

    total = (
        embedding
        + final_norm
        + lm_head_weight
        + lm_head_bias_params
        + recurrent_layers * recurrent_block
        + len(attention_layers) * attention_block
    )

    return {
        "embedding": embedding,
        "final_norm": final_norm,
        "lm_head_weight": lm_head_weight,
        "lm_head_bias": lm_head_bias_params,
        "recurrent_block": recurrent_block,
        "attention_block": attention_block,
        "recurrent_layers": recurrent_layers,
        "attention_layers": len(attention_layers),
        "total_params": total,
        "total_params_millions": total / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Count parameters for the HZ-0A A1 spec.")
    parser.add_argument(
        "--spec",
        default="specs/hz0a_300m_a1.json",
        help="Path to the JSON spec file.",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    report = count_params(spec)
    arch_hash = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()

    output = {
        "spec_path": str(spec_path),
        "architecture_hash": arch_hash,
        "name": spec["name"],
        "version": spec["version"],
        **report,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
