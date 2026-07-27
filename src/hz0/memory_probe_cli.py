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
from hz0.eval.retrieval import retrieval_vocab_bounds, sample_distinct_tokens
from hz0.model import build_model
from hz0.model.session_scratchpad import ScratchpadLogEntry
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


@torch.no_grad()
def _collect_routing_diagnostics(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    num_samples: int,
    num_slots: int,
) -> dict[str, float]:
    """Phase-3 hard-route diagnostics from ``docs/hz0b-mem-fix-plan-2026-07-26.md``.

    The probe model is set to ``eval`` mode. We feed the same synthetic
    ``[key, value, filler*N, key]`` associative-recall prompt used by
    ``evaluate_associative_recall`` and inspect the per-token scratchpad
    routing log to compute:

    * ``route_match_rate``: fraction of positions where
      ``read_hard_idx[t] == write_hard_idx[t]``. A healthy slot-addressed
      scratchpad that is actually using different slots for read vs write
      should be near 1.0 (because the orthogonal slot_addresses init
      spreads points widely), but a routing-collapse failure mode
      collapses to a single slot for everything.
    * ``slot_occupancy``: fraction of slots that received at least one
      write during the probe span.
    * ``slot_collision_rate``: fraction of slots that received more than
      one write (read AND write, so collision spikes with repeated keys).
    * ``soft_routing_entropy_mean``: mean entropy of the soft routing
      distribution (high near orthogonal-init, low when the model has
      committed to a single slot).
    * ``dead_slot_fraction``: fraction of slots that received zero writes
      (collapses with slot-collision when the model gives up on the
      orthogonal structure and routes everything to one slot).

    Returns ``{}`` (empty dict) when the model has no scratchpad
    configured (``num_slots <= 0``). The caller treats that as "diagnostic
    not applicable" and omits the routing block from the JSON output
    rather than writing NaN values that propagate to ``nan - nan = nan``
    in the delta dict.
    """
    if num_slots <= 0:
        return {}
    model.eval()
    model_dtype = next(model.parameters()).dtype
    filler_low, filler_high = retrieval_vocab_bounds(vocab_size)
    filler_width = max(1, (seq_len - 4) // 2)

    route_match_count = 0
    total_compared = 0
    write_slot_count = torch.zeros(num_slots, dtype=torch.float32)
    read_slot_count = torch.zeros(num_slots, dtype=torch.float32)
    soft_entropy_sum = 0.0
    n_log_positions = 0

    for _ in range(num_samples):
        key, value = sample_distinct_tokens(device, vocab_size, 2)
        filler = torch.randint(filler_low, filler_high, (1, filler_width), device=device)
        prompt = torch.cat([key, value, filler, key], dim=1)[:, : max(2, seq_len - 1)]
        with autocast_context(device, model_dtype):
            _, logs = model.forward_with_optional_logs(prompt, return_scratchpad_logs=True)
        for entry in logs:
            write_idx = entry.write_hard_idx.detach().to(torch.long)
            read_idx = entry.read_hard_idx.detach().to(torch.long)
            route_match_count += int((write_idx == read_idx).sum().item())
            total_compared += int(write_idx.numel())
            write_slot_count.scatter_add_(
                0, write_idx, torch.ones(write_idx.shape[0], dtype=torch.float32)
            )
            read_slot_count.scatter_add_(
                0, read_idx, torch.ones(read_idx.shape[0], dtype=torch.float32)
            )
            soft = entry.read_weights.detach()
            soft = torch.clamp(soft, min=1e-12)
            entropy = -(soft * torch.log(soft)).sum(dim=-1)
            soft_entropy_sum += float(entropy.sum().item())
            n_log_positions += int(soft.shape[0])

    write_slots_used = (write_slot_count > 0).sum().item()
    read_slots_used = (read_slot_count > 0).sum().item()
    write_collisions = (write_slot_count > 1).sum().item()
    read_collisions = (read_slot_count > 1).sum().item()

    return {
        "route_match_rate": route_match_count / max(total_compared, 1),
        "slot_occupancy": max(write_slots_used, read_slots_used) / max(num_slots, 1),
        "slot_collision_rate": (write_collisions + read_collisions) / max(2 * num_slots, 1),
        "soft_routing_entropy_mean": soft_entropy_sum / max(n_log_positions, 1),
        "dead_slot_fraction": (write_slot_count == 0).sum().item() / max(num_slots, 1),
        "diagnostic_samples": float(num_samples),
    }


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
    scratchpad = getattr(model, "scratchpad", None)
    num_slots = int(getattr(scratchpad, "num_slots", 0)) if scratchpad is not None else 0
    has_scratchpad = num_slots > 0
    before_routing = (
        _collect_routing_diagnostics(
            model, device, seq_len, vocab_size, args.eval_samples, num_slots
        )
        if has_scratchpad
        else {}
    )

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
    after_routing = (
        _collect_routing_diagnostics(
            model, device, seq_len, vocab_size, args.eval_samples, num_slots
        )
        if has_scratchpad
        else {}
    )
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
    if has_scratchpad:
        result["routing_diagnostics"] = {
            "before": before_routing,
            "after": after_routing,
            "delta": {
                key: float(after_routing[key] - before_routing[key])
                for key in before_routing.keys()
                if key != "diagnostic_samples"
            },
        }

    text = json.dumps(result, indent=2)
    if args.output_path is not None:
        args.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
