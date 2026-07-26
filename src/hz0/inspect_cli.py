from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hz0.config import Config
from hz0.model import build_model


def format_int(value: int) -> str:
    return f"{value:,}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-key", type=str, default="model")
    args = parser.parse_args()

    cfg = Config.load(args.config).raw
    model_cfg = cfg[args.model_key]
    model = build_model(model_cfg)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    bytes_per_param = torch.tensor([], dtype=torch.float32).element_size()
    weights_mebibytes = total_params * bytes_per_param / (1024 * 1024)

    print(f"config={args.config}")
    print(f"model_key={args.model_key}")
    print(f"architecture={model_cfg.get('architecture', 'hybrid')}")
    print(f"total_params={format_int(total_params)}")
    print(f"trainable_params={format_int(trainable_params)}")
    print(f"weights_fp32_mib={weights_mebibytes:.2f}")

    for key in ("d_model", "n_layers", "n_heads", "d_ff", "max_seq_len"):
        if key in model_cfg:
            print(f"{key}={model_cfg[key]}")


if __name__ == "__main__":
    main()
