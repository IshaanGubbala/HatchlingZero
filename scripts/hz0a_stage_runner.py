"""Run an auditable HZ-0A stage on the streaming packed dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restart.hz0a_dataset import StreamingResumablePackedDataset
from reference.hz0a_torch_model import HZ0AConfig, HZ0AModel
from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM


def adamw_update_norm(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """Predict the AdamW update norm without cloning the model parameters."""
    total = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    for group in optimizer.param_groups:
        lr, (beta1, beta2), eps, weight_decay = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = optimizer.state[parameter]
            step = int(state["step"].item()) + 1 if "step" in state else 1
            old_exp_avg = state.get("exp_avg", torch.zeros_like(parameter))
            old_exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(parameter))
            grad = parameter.grad
            exp_avg = beta1 * old_exp_avg + (1.0 - beta1) * grad
            exp_avg_sq = beta2 * old_exp_avg_sq + (1.0 - beta2) * grad.square()
            denominator = (exp_avg_sq / (1.0 - beta2**step)).sqrt() + eps
            update = lr * exp_avg / ((1.0 - beta1**step) * denominator) + lr * weight_decay * parameter
            total = total + update.detach().double().square().sum()
    return float(total.sqrt().item())
from scripts.hz0a_stage_gate import stage_gate
from scripts.hz0a_tiny_training_comparison import TinyHybridLM, TinyTransformerLM, fingerprint, loss_for, parameter_bytes


def save_checkpoint(path: Path, payload: dict) -> None:
    """Write checkpoints atomically so an interruption cannot leave a false file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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


def run_model(name: str, factory, data_path: Path, validation_data: Path, run_dir: Path, seed: int, steps: int, batch_size: int, vocab_size: int, checkpoint_interval: int, validation_interval: int, resume: bool, device: torch.device, dtype: torch.dtype, record_update_norm: bool) -> dict:
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
    last_validation_loss = None
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
    peak_memory = 0
    for step in range(start_step + 1, steps + 1):
        batch = torch.from_numpy(dataset.next_batch(batch_size)).remainder(vocab_size).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, batch, activation_dtype)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        predicted_update_norm = adamw_update_norm(model, optimizer)
        old_parameters = [parameter.detach().clone() for parameter in model.parameters()] if record_update_norm else None
        optimizer.step()
        update_norm = predicted_update_norm
        if old_parameters is not None:
            update_norm = float(torch.sqrt(sum((parameter.detach() - old).square().sum() for parameter, old in zip(model.parameters(), old_parameters))).item())
        if device.type == "mps":
            peak_memory = max(peak_memory, int(torch.mps.current_allocated_memory()))
        validation_loss = None
        if step % validation_interval == 0 or step == steps:
            with torch.no_grad():
                validation_batch = torch.from_numpy(validation_dataset.next_batch(batch_size)).remainder(vocab_size).to(device)
                validation_loss = float(loss_for(model, validation_batch, activation_dtype).item())
                last_validation_loss = validation_loss
                if not np.isfinite(validation_loss):
                    raise RuntimeError(f"non-finite validation loss at step {step}")
        metrics.append({"step": step, "loss": float(loss.item()), "validation_loss": validation_loss, "validation_perplexity": float(np.exp(validation_loss)) if validation_loss is not None else None, "gradient_norm": gradient_norm, "update_norm": update_norm, "batch_index": step - 1})
        if checkpoint_interval and step % checkpoint_interval == 0:
            save_checkpoint(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "metrics": metrics, "dataset_cursor": dataset.snapshot(), "initial_parameter_sha256": initial_hash, "model_parameter_sha256": fingerprint(model), "torch_rng": torch.get_rng_state(), "device": str(device), "dtype": str(dtype)})
    elapsed = time.perf_counter() - start
    tokens_seen = steps * batch_size * (int(batch.shape[1]) - 1)
    final_validation_dataset = StreamingResumablePackedDataset(validation_data, shuffle_seed=0)
    final_validation_batch = torch.from_numpy(final_validation_dataset.next_batch(batch_size)).remainder(vocab_size).to(device)
    with torch.no_grad():
        final_validation_loss = float(loss_for(model, final_validation_batch, activation_dtype).item())
    return {"steps": steps, "tokens_seen": tokens_seen, "budget_complete": False, "metrics": metrics, "initial_parameter_sha256": initial_hash, "final_parameter_sha256": fingerprint(model), "parameters_changed": initial_hash != fingerprint(model), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "parameter_bytes": parameter_bytes(model), "final_loss": metrics[-1]["loss"], "validation_loss": final_validation_loss, "validation_perplexity": float(np.exp(final_validation_loss)), "training_seconds": elapsed, "tokens_per_second": tokens_seen / elapsed, "peak_memory_bytes": peak_memory, "checkpoint": str(checkpoint), "resumed": resume}


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
    parser.add_argument("--record-update-norm", action="store_true", help="Clone all parameters each step to measure exact update norm; expensive for large models")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--models", default="hybrid,transformer", help="Comma-separated model names to run")
    parser.add_argument("--model-config", type=Path, help="Locked HZ-0A JSON model config for the 'locked' model")
    parser.add_argument("--transformer-config", type=Path, help="Matched transformer JSON config for the 'matched_transformer' model")
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
    factories = {"hybrid": TinyHybridLM, "transformer": TinyTransformerLM}
    if args.model_config:
        config = HZ0AConfig.from_json(args.model_config)
        if config.vocab_size != args.vocab_size:
            raise ValueError(f"--vocab-size {args.vocab_size} disagrees with model config vocab_size {config.vocab_size}")
        factories["locked"] = lambda vocab_size: HZ0AModel(config)
    if args.transformer_config:
        transformer_config = MatchedTransformerConfig.from_json(args.transformer_config)
        if transformer_config.vocab_size != args.vocab_size:
            raise ValueError(f"--vocab-size {args.vocab_size} disagrees with transformer config vocab_size {transformer_config.vocab_size}")
        factories["matched_transformer"] = lambda vocab_size: MatchedTransformerLM(transformer_config)
    requested_models = [name.strip() for name in args.models.split(",") if name.strip()]
    if not requested_models or any(name not in factories for name in requested_models):
        raise ValueError(f"--models must contain only: {', '.join(factories)}")
    results = {}
    for name in requested_models:
        factory = factories[name]
        result = run_model(name, factory, args.data, validation_data, args.run_dir, args.seed, steps, args.batch_size, args.vocab_size, args.checkpoint_interval, args.validation_interval, args.resume, device, dtype, args.record_update_norm)
        result["device"] = str(device)
        result["dtype"] = str(dtype)
        result["budget_complete"] = result["tokens_seen"] >= gate["required_tokens"]
        results[name] = result
    report = {"stage": args.stage, "stage_gate": gate, "target_tokens": gate["required_tokens"], "smoke_run": args.steps is not None, "device": str(device), "dtype": str(dtype), "models_requested": requested_models, "models": results}
    (args.run_dir / "stage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
