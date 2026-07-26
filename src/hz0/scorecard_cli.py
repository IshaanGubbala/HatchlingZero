from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from hz0.checkpoint import load_checkpoint
from hz0.config import Config
from hz0.data import build_dataset
from hz0.eval import benchmark_decode_by_context, benchmark_decode_latency, evaluate_copy_retrieval, evaluate_language_model, evaluate_multi_anchor_retrieval
from hz0.eval import (
    evaluate_associative_recall,
    evaluate_overwrite_retrieval,
    evaluate_protected_memory_retrieval,
    evaluate_recall_by_distance,
)
from hz0.model import build_model
from hz0.utils import resolve_dtype


def estimate_tokens_seen(cfg: dict[str, Any], step: int) -> int:
    seq_len = int(cfg["data"]["seq_len"])
    batch_size = int(cfg["data"]["batch_size"])
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    return step * seq_len * batch_size * grad_accum_steps


def load_sidecar_metrics(checkpoint_path: Path) -> dict[str, Any]:
    sidecar = checkpoint_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def collect_checkpoint_metrics(
    cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    checkpoint_path: Path,
    dtype: torch.dtype,
    context_lengths: list[int],
    decode_steps: int,
    retrieval_samples: int,
) -> dict[str, Any]:
    device = torch.device(cfg["device"])
    dataset = build_dataset(
        cfg["data"]["val_text_path"],
        cfg["data"]["seq_len"],
        cfg["data"]["vocab_size"],
        cfg["data"]["val_length"],
        packed=True,
    )
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"])
    model = build_model(model_cfg).to(device=device, dtype=dtype)
    payload = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(payload["model"])

    step = int(payload["step"])
    metrics = evaluate_language_model(model, loader, device, dtype=dtype)
    metrics.update(
        evaluate_associative_recall(
            model=model,
            device=device,
            seq_len=cfg["data"]["seq_len"],
            vocab_size=cfg["data"]["vocab_size"],
            num_samples=retrieval_samples,
        )
    )
    metrics.update(
        evaluate_copy_retrieval(
            model=model,
            device=device,
            seq_len=cfg["data"]["seq_len"],
            vocab_size=cfg["data"]["vocab_size"],
            num_samples=retrieval_samples,
        )
    )
    metrics.update(
        evaluate_overwrite_retrieval(
            model=model,
            device=device,
            seq_len=cfg["data"]["seq_len"],
            vocab_size=cfg["data"]["vocab_size"],
            num_samples=retrieval_samples,
        )
    )
    metrics.update(
        evaluate_protected_memory_retrieval(
            model=model,
            device=device,
            seq_len=cfg["data"]["seq_len"],
            vocab_size=cfg["data"]["vocab_size"],
            num_samples=retrieval_samples,
        )
    )
    metrics.update(
        evaluate_multi_anchor_retrieval(
            model=model,
            device=device,
            seq_len=cfg["data"]["seq_len"],
            vocab_size=cfg["data"]["vocab_size"],
            num_samples=retrieval_samples,
        )
    )
    metrics.update(
        evaluate_recall_by_distance(
            model=model,
            device=device,
            seq_len=cfg["data"]["seq_len"],
            vocab_size=cfg["data"]["vocab_size"],
            num_samples=retrieval_samples,
            distances=[32, 64, 128, 256],
        )
    )
    metrics.update(
        benchmark_decode_latency(
            model=model,
            device=device,
            prompt_len=min(cfg["data"]["seq_len"], model_cfg["max_seq_len"]),
            steps=decode_steps,
            vocab_size=cfg["data"]["vocab_size"],
        )
    )
    metrics.update(
        benchmark_decode_by_context(
            model=model,
            device=device,
            context_lengths=context_lengths,
            steps=decode_steps,
            vocab_size=cfg["data"]["vocab_size"],
        )
    )

    sidecar = load_sidecar_metrics(checkpoint_path)
    total_params = sum(param.numel() for param in model.parameters())
    metrics.update(
        {
            "step": step,
            "tokens_seen_estimate": float(estimate_tokens_seen(cfg, step)),
            "estimated_train_flops": float(6 * total_params * estimate_tokens_seen(cfg, step)),
            "wall_clock_seconds": float(sidecar.get("wall_clock_seconds", -1.0)),
            "grad_norm": float(sidecar.get("grad_norm", -1.0)),
            "peak_memory_bytes": float(sidecar.get("peak_memory_bytes", -1.0)),
        }
    )
    return metrics


def checkpoint_path_for_step(output_dir: Path, step: int) -> Path:
    return output_dir / f"step_{step:07d}.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hybrid-output-dir", type=Path, required=True)
    parser.add_argument("--baseline-output-dir", type=Path, required=True)
    parser.add_argument("--hybrid-steps", type=str, required=True)
    parser.add_argument("--baseline-steps", type=str, required=True)
    parser.add_argument("--context-lengths", type=str, default="64,128,256,512")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--retrieval-samples", type=int, default=64)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config).raw
    dtype = resolve_dtype(cfg["dtype"])
    context_lengths = [int(item) for item in args.context_lengths.split(",") if item.strip()]
    hybrid_steps = [int(item) for item in args.hybrid_steps.split(",") if item.strip()]
    baseline_steps = [int(item) for item in args.baseline_steps.split(",") if item.strip()]

    result = {
        "config": str(args.config),
        "context_lengths": context_lengths,
        "decode_steps": args.decode_steps,
        "retrieval_samples": args.retrieval_samples,
        "hybrid": {},
        "baseline": {},
    }

    for step in hybrid_steps:
        checkpoint = checkpoint_path_for_step(args.hybrid_output_dir, step)
        result["hybrid"][str(step)] = collect_checkpoint_metrics(
            cfg=cfg,
            model_cfg=cfg["model"],
            checkpoint_path=checkpoint,
            dtype=dtype,
            context_lengths=context_lengths,
            decode_steps=args.decode_steps,
            retrieval_samples=args.retrieval_samples,
        )

    for step in baseline_steps:
        checkpoint = checkpoint_path_for_step(args.baseline_output_dir, step)
        result["baseline"][str(step)] = collect_checkpoint_metrics(
            cfg=cfg,
            model_cfg=cfg["baseline"],
            checkpoint_path=checkpoint,
            dtype=dtype,
            context_lengths=context_lengths,
            decode_steps=args.decode_steps,
            retrieval_samples=args.retrieval_samples,
        )

    text = json.dumps(result, indent=2)
    if args.output_path is not None:
        args.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
