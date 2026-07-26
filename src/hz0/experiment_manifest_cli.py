from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from hz0.checkpoint import load_checkpoint
from hz0.config import Config
from hz0.model import build_model
from hz0.tokenizer import ByteTokenizer
from hz0.utils import resolve_dtype


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def file_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_manifest(
    *,
    config_path: Path,
    model_key: str,
    checkpoint_path: Path,
    experiment_name: str,
    seed: int | None,
    cwd: Path,
) -> dict[str, Any]:
    cfg = Config.load(config_path).raw
    model_cfg = cfg[model_key]
    dtype = resolve_dtype(cfg["dtype"])
    model = build_model(model_cfg).to(dtype=dtype)
    payload = load_checkpoint(checkpoint_path, torch.device("cpu"))
    model.load_state_dict(payload["model"])
    total_params = sum(param.numel() for param in model.parameters())
    tokenizer = ByteTokenizer()

    train_path = Path(cfg["data"]["train_text_path"])
    val_path = Path(cfg["data"]["val_text_path"])
    effective_tokens_per_update = (
        int(cfg["data"]["batch_size"]) * int(cfg["data"]["seq_len"]) * int(cfg["train"].get("grad_accum_steps", 1))
    )
    attention_every = model_cfg.get("attention_every")
    manifest = {
        "experiment_name": experiment_name,
        "created_at": "2026-07-26",
        "git_commit": git_commit(cwd),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config": cfg,
        "model_key": model_key,
        "seed": int(cfg["seed"] if seed is None else seed),
        "parameter_count": total_params,
        "trainable_parameter_count": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "layers": int(model_cfg["n_layers"]),
        "hidden_dim": int(model_cfg["d_model"]),
        "intermediate_dim": int(model_cfg["d_ff"]),
        "n_heads": int(model_cfg["n_heads"]),
        "attention_recurrent_schedule": {
            "architecture": model_cfg["architecture"],
            "attention_every": attention_every,
        },
        "sequence_length": int(cfg["data"]["seq_len"]),
        "microbatch_size": int(cfg["data"]["batch_size"]),
        "gradient_accumulation_steps": int(cfg["train"].get("grad_accum_steps", 1)),
        "effective_tokens_per_optimizer_update": effective_tokens_per_update,
        "learning_rate_schedule": {
            "base_lr": float(cfg["optim"]["lr"]),
            "weight_decay": float(cfg["optim"]["weight_decay"]),
            "betas": list(cfg["optim"]["betas"]),
        },
        "gradient_clip": float(cfg["optim"]["grad_clip"]),
        "precision": cfg["dtype"],
        "device": cfg["device"],
        "checkpoint": file_metadata(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "dataset_manifest": {
            "train": file_metadata(train_path),
            "val": file_metadata(val_path),
            "sources_path": str(train_path.parent / "hz0a_seed_sources.txt"),
        },
        "validation_split_definition": {
            "val_text_path": str(val_path),
            "val_length": int(cfg["data"]["val_length"]),
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "vocab_size": int(tokenizer.vocab_size),
            "spec_sha256": sha256_text("byte-tokenizer-v1-0-255"),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "macos": platform.mac_ver()[0],
            "platform": platform.platform(),
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--model-key", type=str, default="model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    manifest = build_manifest(
        config_path=args.config,
        model_key=args.model_key,
        checkpoint_path=args.checkpoint,
        experiment_name=args.experiment_name,
        seed=args.seed,
        cwd=Path.cwd(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "experiment_name": args.experiment_name}, indent=2))


if __name__ == "__main__":
    main()
