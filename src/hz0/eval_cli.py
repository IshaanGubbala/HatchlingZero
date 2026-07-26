from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hz0.config import Config
from hz0.data import RandomTokenDataset, TextTokenDataset
from hz0.eval import evaluate_language_model
from hz0.model import HybridLM


def build_dataset(path: str | None, seq_len: int, vocab_size: int):
    if path:
        return TextTokenDataset(path, seq_len=seq_len, vocab_size=vocab_size)
    return RandomTokenDataset(seq_len=seq_len, vocab_size=vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = Config.load(args.config).raw
    device = torch.device(cfg["device"])
    dataset = build_dataset(
        cfg["data"]["val_text_path"],
        cfg["data"]["seq_len"],
        cfg["data"]["vocab_size"],
    )
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"])
    model = HybridLM(**cfg["model"]).to(device)
    metrics = evaluate_language_model(model, loader, device)
    print(metrics)


if __name__ == "__main__":
    main()
