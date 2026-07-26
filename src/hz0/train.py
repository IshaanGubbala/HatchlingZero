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
from hz0.eval import benchmark_decode_latency, evaluate_copy_retrieval, evaluate_language_model
from hz0.generation import greedy_generate
from hz0.model import build_model
from hz0.runtime import autocast_context
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
    )
    val_ds = build_dataset(
        cfg["data"]["val_text_path"],
        cfg["data"]["seq_len"],
        cfg["data"]["vocab_size"],
        cfg["data"]["val_length"],
        packed=True,
    )
    num_workers = cfg["data"].get("num_workers", 0)
    persistent_workers = bool(num_workers > 0 and cfg["data"].get("persistent_workers", True))
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
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            optimizer.step()
            running_tokens += x.numel()

            if step % cfg["train"]["log_every"] == 0:
                elapsed = max(time.perf_counter() - running_start, 1e-8)
                train_toks = running_tokens / elapsed
                print(f"step={step} loss={loss.item():.4f} train_tokens_per_second={train_toks:.2f}")
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
                    benchmark_decode_latency(
                        model=model,
                        device=device,
                        prompt_len=min(cfg["data"]["seq_len"], model_cfg["max_seq_len"]),
                        steps=8,
                        vocab_size=cfg["data"]["vocab_size"],
                    )
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
