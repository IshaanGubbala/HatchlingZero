#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_transformer_params(config: dict) -> int:
    v = int(config["vocab_size"])
    d = int(config["d_model"])
    h = int(config["num_heads"])
    head = int(config["head_dim"])
    ff = int(config["d_ff"])
    layers = int(config["num_layers"])
    if h * head != d:
        raise ValueError("num_heads * head_dim must equal d_model")
    embedding = v * d
    final_norm = d
    attention = d * (3 * h * head) + 3 * h * head + (h * head) * d + d + 2 * d
    mlp = 3 * d * ff + 2 * ff + d
    return embedding + final_norm + layers * (attention + mlp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count the HZ-0A matched-transformer parameters.")
    parser.add_argument("--config", default="configs/hz0a_transformer_matched.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    actual = count_transformer_params(config)
    output = {**config, "parameter_count_computed": actual, "count_matches_config": actual == config["parameter_count"]}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
