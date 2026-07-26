from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hz0.config import Config
from hz0.data import RandomTokenDataset, TextTokenDataset
from hz0.eval import evaluate_language_model
from hz0.model import HybridLM
from hz0.utils import resolve_dtype, set_seed


def build_dataset(path: str | None, seq_len: int, vocab_size: int):
    if path:
        return TextTokenDataset(path, seq_len=seq_len, vocab_size=vocab_size)
    return RandomTokenDataset(seq_len=seq_len, vocab_size=vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config).raw
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"])
    dtype = resolve_dtype(cfg["dtype"])

    train_ds = build_dataset(
        cfg["data"]["train_text_path"],
        cfg["data"]["seq_len"],
        cfg["data"]["vocab_size"],
    )
    val_ds = build_dataset(
        cfg["data"]["val_text_path"],
        cfg["data"]["seq_len"],
        cfg["data"]["vocab_size"],
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"])

    model = HybridLM(**cfg["model"]).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )

    max_steps = args.max_steps or cfg["train"]["max_steps"]
    model.train()
    step = 0
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            optimizer.step()

            if step % cfg["train"]["log_every"] == 0:
                print(f"step={step} loss={loss.item():.4f}")

            if step > 0 and step % cfg["train"]["eval_every"] == 0:
                metrics = evaluate_language_model(model, val_loader, device)
                print(
                    "eval "
                    f"loss={metrics['loss']:.4f} "
                    f"perplexity={metrics['perplexity']:.2f}"
                )
                model.train()
            step += 1


if __name__ == "__main__":
    main()
