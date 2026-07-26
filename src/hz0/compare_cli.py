from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hz0.checkpoint import load_checkpoint
from hz0.config import Config
from hz0.data import build_dataset
from hz0.eval import benchmark_decode_latency, evaluate_copy_retrieval, evaluate_language_model
from hz0.model import build_model


def collect_metrics(cfg: dict, model_cfg: dict, checkpoint: Path | None) -> dict[str, float]:
    device = torch.device(cfg["device"])
    dataset = build_dataset(
        cfg["data"]["val_text_path"],
        cfg["data"]["seq_len"],
        cfg["data"]["vocab_size"],
        cfg["data"]["val_length"],
        packed=True,
    )
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"])
    model = build_model(model_cfg).to(device)
    if checkpoint is not None:
        payload = load_checkpoint(checkpoint, device)
        model.load_state_dict(payload["model"])
    metrics = evaluate_language_model(model, loader, device)
    metrics.update(
        evaluate_copy_retrieval(
            model=model,
            device=device,
            seq_len=cfg["data"]["seq_len"],
            vocab_size=cfg["data"]["vocab_size"],
            num_samples=32,
        )
    )
    metrics.update(
        benchmark_decode_latency(
            model=model,
            device=device,
            prompt_len=min(cfg["data"]["seq_len"], model_cfg["max_seq_len"]),
            steps=16,
            vocab_size=cfg["data"]["vocab_size"],
        )
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, default=None)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config).raw
    result = {
        "hybrid": collect_metrics(cfg, cfg["model"], args.hybrid_checkpoint),
        "baseline": collect_metrics(cfg, cfg["baseline"], args.baseline_checkpoint),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
