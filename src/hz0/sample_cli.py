from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hz0.checkpoint import load_checkpoint
from hz0.config import Config
from hz0.generation import greedy_generate
from hz0.model import build_model
from hz0.tokenizer import ByteTokenizer
from hz0.utils import resolve_dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="HZ-0A ")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--model-key", type=str, default="model")
    args = parser.parse_args()

    cfg = Config.load(args.config).raw
    model_cfg = cfg[args.model_key]
    device = torch.device(cfg["device"])
    dtype = resolve_dtype(cfg["dtype"])
    model = build_model(model_cfg).to(device=device, dtype=dtype)
    payload = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(payload["model"])

    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode(args.prompt).unsqueeze(0).to(device)
    output = greedy_generate(
        model=model,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=model_cfg["max_seq_len"],
    )
    print(tokenizer.decode(output[0].cpu()))


if __name__ == "__main__":
    main()
