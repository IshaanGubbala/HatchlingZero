"""Run an auditable HZ-0A stage on the streaming packed dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restart.hz0a_dataset import StreamingResumablePackedDataset
from scripts.hz0a_stage_gate import stage_gate
from scripts.hz0a_tiny_training_comparison import TinyHybridLM, TinyTransformerLM, fingerprint, loss_for, parameter_bytes


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def packed_sequence_length(path: Path) -> int:
    with path.open(encoding="utf-8") as reader:
        first = json.loads(reader.readline())
    if not isinstance(first, list) or not first:
        raise ValueError("packed data must contain non-empty token sequences")
    return len(first)


def run_model(name: str, factory, data_path: Path, validation_data: Path, run_dir: Path, seed: int, steps: int, batch_size: int, vocab_size: int, checkpoint_interval: int, validation_interval: int, resume: bool, device: torch.device, dtype: torch.dtype) -> dict:
    seed_everything(seed)
    dataset = StreamingResumablePackedDataset(data_path, shuffle_seed=seed)
    validation_dataset = StreamingResumablePackedDataset(validation_data, shuffle_seed=0)
    # Keep fp32 master parameters and optimizer state; autocast only activations.
    model = factory(vocab_size=vocab_size).to(device)
    activation_dtype = dtype if dtype != torch.float32 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = run_dir / f"{name}.pt"
    initial_hash = fingerprint(model)
    metrics = []
    start_step = 0
    if resume:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        dataset = StreamingResumablePackedDataset.from_snapshot(data_path, payload["dataset_cursor"])
        validation_dataset = StreamingResumablePackedDataset(validation_data, shuffle_seed=0)
        initial_hash = payload["initial_parameter_sha256"]
        metrics = payload["metrics"]
        start_step = int(payload["step"])
        torch.set_rng_state(payload["torch_rng"])
    start = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        batch = torch.from_numpy(dataset.next_batch(batch_size)).remainder(vocab_size).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, batch, activation_dtype)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        validation_loss = None
        if step % validation_interval == 0 or step == steps:
            with torch.no_grad():
                validation_batch = torch.from_numpy(validation_dataset.next_batch(batch_size)).remainder(vocab_size).to(device)
                validation_loss = float(loss_for(model, validation_batch, activation_dtype).item())
                if not np.isfinite(validation_loss):
                    raise RuntimeError(f"non-finite validation loss at step {step}")
        metrics.append({"step": step, "loss": float(loss.item()), "validation_loss": validation_loss, "gradient_norm": gradient_norm, "batch_index": step - 1})
        if checkpoint_interval and step % checkpoint_interval == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "metrics": metrics, "dataset_cursor": dataset.snapshot(), "initial_parameter_sha256": initial_hash, "model_parameter_sha256": fingerprint(model), "torch_rng": torch.get_rng_state(), "device": str(device)}, checkpoint)
    elapsed = time.perf_counter() - start
    tokens_seen = steps * batch_size * (int(batch.shape[1]) - 1)
    return {"steps": steps, "tokens_seen": tokens_seen, "budget_complete": False, "metrics": metrics, "initial_parameter_sha256": initial_hash, "final_parameter_sha256": fingerprint(model), "parameters_changed": initial_hash != fingerprint(model), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "parameter_bytes": parameter_bytes(model), "final_loss": metrics[-1]["loss"], "validation_loss": float(loss_for(model, batch).item()), "training_seconds": elapsed, "tokens_per_second": tokens_seen / elapsed, "checkpoint": str(checkpoint), "resumed": resume}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an auditable tiny HZ-0A stage against streaming packed data.")
    parser.add_argument("--stage-config", default="configs/hz0a_training_stages.json", type=Path)
    parser.add_argument("--stage", default="stage1_validation")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, help="Optional bounded smoke run; omitted means the full stage budget.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()
    device_name = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    device = torch.device("mps" if device_name == "mps" else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    if dtype == torch.float16 and device.type == "cpu":
        raise RuntimeError("--dtype fp16 requires --device mps for this runner")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")
    gate = stage_gate(args.stage_config, args.data, args.stage)
    if not gate["sufficient"]:
        print(json.dumps(gate, indent=2, sort_keys=True))
        raise SystemExit(2)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    validation_data = args.validation_data or args.data
    if not validation_data.is_file():
        raise FileNotFoundError(validation_data)
    sequence_length = packed_sequence_length(args.data)
    tokens_per_step = args.batch_size * (sequence_length - 1)
    target_steps = (gate["required_tokens"] + tokens_per_step - 1) // tokens_per_step
    steps = args.steps if args.steps is not None else target_steps
    if steps <= 0:
        raise ValueError("steps must be positive")
    results = {}
    for name, factory in (("hybrid", TinyHybridLM), ("transformer", TinyTransformerLM)):
        result = run_model(name, factory, args.data, validation_data, args.run_dir, args.seed, steps, args.batch_size, args.vocab_size, args.checkpoint_interval, args.validation_interval, args.resume, device, dtype)
        result["device"] = str(device)
        result["dtype"] = str(dtype)
        result["budget_complete"] = result["tokens_seen"] >= gate["required_tokens"]
        results[name] = result
    report = {"stage": args.stage, "stage_gate": gate, "target_tokens": gate["required_tokens"], "smoke_run": args.steps is not None, "device": str(device), "dtype": str(dtype), "models": results}
    (args.run_dir / "stage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
