from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from hz0.checkpoint import load_checkpoint
from hz0.config import Config
from hz0.data import build_dataset
from hz0.eval import (
    evaluate_associative_recall,
    evaluate_overwrite_retrieval,
    evaluate_protected_memory_retrieval,
    evaluate_recall_by_distance,
)
from hz0.model import build_model
from hz0.runtime import autocast_context
from hz0.utils import resolve_dtype, set_seed


MemoryEval = Callable[[torch.nn.Module, torch.device, int, int, int], dict[str, float]]


def _eval_for_mode(mode: str) -> tuple[str, MemoryEval]:
    mapping: dict[str, tuple[str, MemoryEval]] = {
        "associative": ("associative_recall_accuracy", evaluate_associative_recall),
        "overwrite": ("overwrite_retrieval_accuracy", evaluate_overwrite_retrieval),
        "protected": ("protected_memory_accuracy", evaluate_protected_memory_retrieval),
        "distance": ("recall_distance_128_accuracy", _evaluate_distance_128),
    }
    if mode not in mapping:
        raise ValueError(f"Unsupported memory probe mode: {mode}")
    return mapping[mode]


def _evaluate_distance_128(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    num_samples: int,
) -> dict[str, float]:
    return evaluate_recall_by_distance(
        model=model,
        device=device,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_samples=num_samples,
        distances=[128],
    )


def _collect_metrics(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    num_samples: int,
    mode: str,
) -> dict[str, float]:
    _, fn = _eval_for_mode(mode)
    metrics = fn(
        model=model,
        device=device,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_samples=num_samples,
    )
    return {key: float(value) for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-mode", type=str, choices=["associative", "overwrite", "protected", "distance"], required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--probe-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config).raw
    set_seed(int(cfg["seed"]))
    torch.set_float32_matmul_precision("high")

    device = torch.device(cfg["device"])
    dtype = resolve_dtype(cfg["dtype"])
    model = build_model(cfg["model"]).to(device=device, dtype=dtype)
    payload = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(payload["model"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.probe_lr,
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )

    seq_len = int(cfg["data"]["seq_len"])
    vocab_size = int(cfg["data"]["vocab_size"])
    dataset = build_dataset(
        path=cfg["data"]["train_text_path"],
        seq_len=seq_len,
        vocab_size=vocab_size,
        random_length=int(cfg["data"]["train_length"]),
        packed=True,
        memory_mix_probability=1.0,
        memory_task_mode=args.task_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    data_iter = iter(loader)

    metric_name, _ = _eval_for_mode(args.task_mode)
    before = _collect_metrics(model, device, seq_len, vocab_size, args.eval_samples, args.task_mode)

    model.train()
    train_start = time.perf_counter()
    final_loss = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(max(1, args.grad_accum_steps)):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            with autocast_context(device, dtype):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits[:, -1, :], y[:, -1])
                loss = loss / max(1, args.grad_accum_steps)
            loss.backward()
            final_loss = float(loss.item() * max(1, args.grad_accum_steps))
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["optim"]["grad_clip"]))
        optimizer.step()

    after = _collect_metrics(model, device, seq_len, vocab_size, args.eval_samples, args.task_mode)
    elapsed = time.perf_counter() - train_start

    result = {
        "date": "2026-07-26",
        "checkpoint": str(args.checkpoint),
        "task_mode": args.task_mode,
        "probe_steps": int(args.steps),
        "probe_lr": float(args.probe_lr),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "elapsed_seconds": elapsed,
        "metric_name": metric_name,
        "before": before,
        "after": after,
        "delta": float(after.get(metric_name, 0.0) - before.get(metric_name, 0.0)),
        "final_last_token_loss": final_loss,
    }

    text = json.dumps(result, indent=2)
    if args.output_path is not None:
        args.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
