from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hz0.checkpoint import load_checkpoint
from hz0.config import Config
from hz0.eval import benchmark_decode_latency, evaluate_copy_retrieval
from hz0.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--retrieval-samples", type=int, default=32)
    parser.add_argument("--model-key", type=str, default="model")
    args = parser.parse_args()

    cfg = Config.load(args.config).raw
    model_cfg = cfg[args.model_key]
    device = torch.device(cfg["device"])
    model = build_model(model_cfg).to(device)

    if args.checkpoint:
        payload = load_checkpoint(args.checkpoint, device)
        model.load_state_dict(payload["model"])

    metrics = {}
    metrics.update(
        benchmark_decode_latency(
            model=model,
            device=device,
            prompt_len=cfg["data"]["seq_len"],
            steps=args.decode_steps,
            vocab_size=cfg["data"]["vocab_size"],
        )
    )
    metrics.update(
        evaluate_copy_retrieval(
            model=model,
            device=device,
            seq_len=cfg["data"]["seq_len"],
            vocab_size=cfg["data"]["vocab_size"],
            num_samples=args.retrieval_samples,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
