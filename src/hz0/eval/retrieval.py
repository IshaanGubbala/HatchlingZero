from __future__ import annotations

import time

import torch


@torch.no_grad()
def evaluate_copy_retrieval(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    num_samples: int = 32,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    prefix_len = max(4, seq_len // 2)
    filler_low = 32 if vocab_size > 64 else 0
    filler_high = min(vocab_size, 127) if vocab_size > 127 else vocab_size

    for _ in range(num_samples):
        needle = torch.randint(0, vocab_size, (1, 1), device=device)
        filler = torch.randint(filler_low, filler_high, (1, prefix_len - 2), device=device)
        prompt = torch.cat([needle, filler, needle], dim=1)
        logits = model(prompt[:, :-1])
        pred = torch.argmax(logits[:, -1, :], dim=-1)
        correct += int((pred == prompt[:, -1]).item())
        total += 1
    return {"copy_retrieval_accuracy": correct / max(total, 1), "samples": float(total)}


@torch.no_grad()
def benchmark_decode_latency(
    model: torch.nn.Module,
    device: torch.device,
    prompt_len: int,
    steps: int,
    vocab_size: int,
) -> dict[str, float]:
    model.eval()
    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)
    start = time.perf_counter()
    for _ in range(steps):
        logits = model(prompt)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        prompt = torch.cat([prompt[:, 1:], next_token], dim=1)
    elapsed = time.perf_counter() - start
    tokens_per_second = steps / max(elapsed, 1e-8)
    return {
        "decode_steps": float(steps),
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
    }
