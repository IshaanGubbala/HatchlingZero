from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hz0.checkpoint import load_checkpoint
from hz0.config import Config
from hz0.generation import greedy_generate
from hz0.model import HybridLM
from hz0.tokenizer import ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="HZ-0A ")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    cfg = Config.load(args.config).raw
    device = torch.device(cfg["device"])
    model = HybridLM(**cfg["model"]).to(device)
    payload = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(payload["model"])

    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode(args.prompt).unsqueeze(0).to(device)
    output = greedy_generate(
        model=model,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=cfg["model"]["max_seq_len"],
    )
    print(tokenizer.decode(output[0].cpu()))


if __name__ == "__main__":
    main()
