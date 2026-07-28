"""Run a bounded full-parameter HZ-0A AdamW training smoke."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz0a_torch_model import HZ0AConfig, HZ0AModel


def run(config_path: Path, steps: int, batch_size: int, sequence_length: int, seed: int, learning_rate: float, device_name: str = "cpu") -> dict:
    torch.manual_seed(seed)
    config = HZ0AConfig.from_json(config_path)
    device = torch.device(device_name)
    model = HZ0AModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    initial_norm = float(torch.linalg.vector_norm(torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])))
    metrics = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        tokens = torch.randint(config.vocab_size, (batch_size, sequence_length + 1), device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(tokens[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("full-model loss is non-finite")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError("full-model gradient norm is non-finite")
        optimizer.step()
        metrics.append({"step": step, "loss": float(loss.detach()), "gradient_norm": gradient_norm})
    elapsed = time.perf_counter() - started
    final_norm = float(torch.linalg.vector_norm(torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])))
    return {"config": str(config_path), "device": device_name, "steps": steps, "batch_size": batch_size, "sequence_length": sequence_length, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "initial_parameter_l2": initial_norm, "final_parameter_l2": final_norm, "parameters_changed": initial_norm != final_norm, "metrics": metrics, "tokens_seen": steps * batch_size * sequence_length, "training_seconds": elapsed, "tokens_per_second": steps * batch_size * sequence_length / elapsed, "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded full-model HZ-0A AdamW smoke.")
    parser.add_argument("--config", default="specs/hz0a_300m_a1.json", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.config, args.steps, args.batch_size, args.sequence_length, args.seed, args.learning_rate, args.device)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
