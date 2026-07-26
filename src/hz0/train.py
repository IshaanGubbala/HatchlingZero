from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hz0.checkpoint import load_checkpoint, save_checkpoint
from hz0.config import Config
from hz0.data import build_dataset
from hz0.eval import benchmark_decode_latency, evaluate_copy_retrieval, evaluate_language_model, evaluate_multi_anchor_retrieval
from hz0.generation import greedy_generate
from hz0.model import build_model
from hz0.runtime import autocast_context, current_memory_bytes
from hz0.tokenizer import ByteTokenizer
from hz0.utils import resolve_dtype, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--model-key", type=str, default="model")
    args = parser.parse_args()

    cfg = Config.load(args.config).raw
    model_cfg = cfg[args.model_key]
    set_seed(cfg["seed"])
    torch.set_float32_matmul_precision("high")

    device = torch.device(cfg["device"])
    dtype = resolve_dtype(cfg["dtype"])

    train_ds = build_dataset(
        cfg["data"]["train_text_path"],
        cfg["data"]["seq_len"],
        cfg["data"]["vocab_size"],
        cfg["data"]["train_length"],
        packed=True,
        retrieval_mix_probability=float(cfg["data"].get("retrieval_mix_probability", 0.0)),
        retrieval_num_anchors=int(cfg["data"].get("retrieval_num_anchors", 3)),
        memory_mix_probability=float(cfg["data"].get("memory_mix_probability", 0.0)),
        memory_task_mode=str(cfg["data"].get("memory_task_mode", "mixed")),
    )
    num_workers = cfg["data"].get("num_workers", 0)
    persistent_workers = bool(num_workers > 0 and cfg["data"].get("persistent_workers", True))
    memory_aux_weight = float(cfg["train"].get("memory_aux_weight", 0.0))
    memory_aux_last_token_weight = float(cfg["train"].get("memory_aux_last_token_weight", 0.0))
    memory_aux_loss_mode = str(cfg["train"].get("memory_aux_loss_mode", "blend")).lower()
    if memory_aux_loss_mode not in {"blend", "full", "last_token_only"}:
        raise ValueError(f"Unsupported memory_aux_loss_mode: {memory_aux_loss_mode}")
    memory_aux_loader = None
    memory_aux_iter = None
    if memory_aux_weight > 0.0:
        memory_aux_ds = build_dataset(
            cfg["data"]["train_text_path"],
            cfg["data"]["seq_len"],
            cfg["data"]["vocab_size"],
            cfg["data"]["train_length"],
            packed=True,
            retrieval_mix_probability=float(cfg["train"].get("memory_aux_retrieval_mix_probability", 0.0)),
            retrieval_num_anchors=int(cfg["data"].get("retrieval_num_anchors", 3)),
            memory_mix_probability=float(cfg["train"].get("memory_aux_memory_mix_probability", 1.0)),
            memory_task_mode=str(cfg["train"].get("memory_aux_task_mode", cfg["data"].get("memory_task_mode", "mixed"))),
        )
        memory_aux_loader = DataLoader(
            memory_aux_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        memory_aux_iter = iter(memory_aux_loader)
    val_ds = build_dataset(
        cfg["data"]["val_text_path"],
        cfg["data"]["seq_len"],
        cfg["data"]["vocab_size"],
        cfg["data"]["val_length"],
        packed=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    model = build_model(model_cfg).to(device=device, dtype=dtype)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"model_params={total_params} trainable_params={trainable_params}")
    if cfg["train"].get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model, mode=cfg["train"].get("compile_mode", "reduce-overhead"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )

    max_steps = args.max_steps or cfg["train"]["max_steps"]
    grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    output_dir = Path(cfg["train"]["output_dir"])
    if args.model_key != "model":
        output_dir = output_dir.parent / f"{output_dir.name}-{args.model_key}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tokenizer = ByteTokenizer()
    start_step = 0

    if args.resume:
        payload = load_checkpoint(args.resume, device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"]) + 1
        print(f"resumed_from={args.resume} step={start_step}")

    model.train()
    step = start_step
    running_tokens = 0
    running_start = time.perf_counter()
    train_start = time.perf_counter()
    total_tokens_seen = start_step * cfg["data"]["batch_size"] * cfg["data"]["seq_len"] * grad_accum_steps
    peak_memory_bytes = current_memory_bytes(device)
    train_iter = iter(train_loader)
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_value = None
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            with autocast_context(device, dtype):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                if memory_aux_loader is not None and memory_aux_iter is not None:
                    try:
                        aux_batch = next(memory_aux_iter)
                    except StopIteration:
                        memory_aux_iter = iter(memory_aux_loader)
                        aux_batch = next(memory_aux_iter)
                    aux_batch = aux_batch.to(device)
                    aux_x = aux_batch[:, :-1]
                    aux_y = aux_batch[:, 1:]
                    aux_logits = model(aux_x)
                    aux_full_loss = torch.nn.functional.cross_entropy(
                        aux_logits.reshape(-1, aux_logits.size(-1)),
                        aux_y.reshape(-1),
                    )
                    aux_last_loss = torch.nn.functional.cross_entropy(
                        aux_logits[:, -1, :],
                        aux_y[:, -1],
                    )
                    if memory_aux_loss_mode == "last_token_only":
                        aux_loss = aux_last_loss
                    elif memory_aux_loss_mode == "full":
                        aux_loss = aux_full_loss
                    else:
                        aux_loss = aux_full_loss
                        if memory_aux_last_token_weight > 0.0:
                            aux_loss = aux_loss + memory_aux_last_token_weight * aux_last_loss
                    loss = loss + memory_aux_weight * aux_loss
                loss = loss / grad_accum_steps
            loss.backward()
            running_tokens += x.numel()
            total_tokens_seen += x.numel()
            loss_value = loss.item() * grad_accum_steps

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"]).item())
        optimizer.step()
        peak_memory_bytes = max(peak_memory_bytes, current_memory_bytes(device))

        if step % cfg["train"]["log_every"] == 0:
            elapsed = max(time.perf_counter() - running_start, 1e-8)
            train_toks = running_tokens / elapsed
            wall_clock = time.perf_counter() - train_start
            print(
                f"step={step} loss={loss_value:.4f} "
                f"train_tokens_per_second={train_toks:.2f} "
                f"tokens_seen={total_tokens_seen} "
                f"grad_norm={grad_norm:.4f} "
                f"wall_clock_seconds={wall_clock:.2f} "
                f"peak_memory_bytes={peak_memory_bytes}"
            )
            running_tokens = 0
            running_start = time.perf_counter()

        if step > 0 and step % cfg["train"]["eval_every"] == 0:
            metrics = evaluate_language_model(model, val_loader, device)
            metrics.update(
                evaluate_copy_retrieval(
                    model=model,
                    device=device,
                    seq_len=cfg["data"]["seq_len"],
                    vocab_size=cfg["data"]["vocab_size"],
                    num_samples=16,
                )
            )
            metrics.update(
                evaluate_multi_anchor_retrieval(
                    model=model,
                    device=device,
                    seq_len=cfg["data"]["seq_len"],
                    vocab_size=cfg["data"]["vocab_size"],
                    num_samples=16,
                )
            )
            metrics.update(
                benchmark_decode_latency(
                    model=model,
                    device=device,
                    prompt_len=min(cfg["data"]["seq_len"], model_cfg["max_seq_len"]),
                    steps=8,
                    vocab_size=cfg["data"]["vocab_size"],
                )
            )
            metrics.update(
                {
                    "tokens_seen": float(total_tokens_seen),
                    "grad_norm": grad_norm,
                    "wall_clock_seconds": time.perf_counter() - train_start,
                    "peak_memory_bytes": float(peak_memory_bytes),
                }
            )
            print(
                "eval "
                f"loss={metrics['loss']:.4f} "
                f"perplexity={metrics['perplexity']:.2f} "
                f"copy_retrieval_accuracy={metrics['copy_retrieval_accuracy']:.3f} "
                f"tokens_per_second={metrics['tokens_per_second']:.2f}"
            )
            save_checkpoint(output_dir, step, model, optimizer, cfg, metrics)
            model.train()

        if step > 0 and step % cfg["train"]["sample_every"] == 0:
            prompt = tokenizer.encode(cfg["train"]["sample_prompt"]).unsqueeze(0).to(device)
            generated = greedy_generate(
                model=model,
                prompt=prompt,
                max_new_tokens=cfg["train"]["sample_tokens"],
                max_seq_len=model_cfg["max_seq_len"],
            )
            text = tokenizer.decode(generated[0].cpu())
            print(f"sample step={step} text={text!r}")

        if step > 0 and step % cfg["train"]["save_every"] == 0:
            save_checkpoint(output_dir, step, model, optimizer, cfg)
        step += 1

    save_checkpoint(output_dir, step, model, optimizer, cfg)


if __name__ == "__main__":
    main()
