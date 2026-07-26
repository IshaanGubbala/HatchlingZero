from __future__ import annotations

import time

import torch

from hz0.runtime import autocast_context, maybe_sync_device


def _retrieval_vocab_bounds(vocab_size: int) -> tuple[int, int]:
    filler_low = 32 if vocab_size > 64 else 0
    filler_high = min(vocab_size, 127) if vocab_size > 127 else vocab_size
    return filler_low, filler_high


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
    filler_low, filler_high = _retrieval_vocab_bounds(vocab_size)
    model_dtype = next(model.parameters()).dtype

    for _ in range(num_samples):
        needle = torch.randint(0, vocab_size, (1, 1), device=device)
        filler = torch.randint(filler_low, filler_high, (1, prefix_len - 2), device=device)
        prompt = torch.cat([needle, filler, needle], dim=1)
        with autocast_context(device, model_dtype):
            logits = model(prompt[:, :-1])
        pred = torch.argmax(logits[:, -1, :], dim=-1)
        correct += int((pred == prompt[:, -1]).item())
        total += 1
    return {"copy_retrieval_accuracy": correct / max(total, 1), "samples": float(total)}


@torch.no_grad()
def evaluate_multi_anchor_retrieval(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    num_samples: int = 32,
    num_anchors: int = 3,
) -> dict[str, float]:
    model.eval()
    exact_correct = 0
    anchor_set_correct = 0
    total = 0
    model_dtype = next(model.parameters()).dtype
    filler_low, filler_high = _retrieval_vocab_bounds(vocab_size)
    working_len = max(seq_len, num_anchors * 4 + 4)
    segment_len = max(2, (working_len - 1) // max(num_anchors, 1))

    for _ in range(num_samples):
        anchors = torch.randint(0, vocab_size, (1, num_anchors), device=device)
        pieces = []
        answer_index = torch.randint(0, num_anchors, (1,), device=device).item()
        for idx in range(num_anchors):
            filler = torch.randint(filler_low, filler_high, (1, segment_len - 1), device=device)
            pieces.append(torch.cat([anchors[:, idx : idx + 1], filler], dim=1))
        query_token = anchors[:, answer_index : answer_index + 1]
        prompt = torch.cat([*pieces, query_token], dim=1)
        prompt = prompt[:, : working_len - 1]
        target = anchors[:, (answer_index + 1) % num_anchors]

        with autocast_context(device, model_dtype):
            logits = model(prompt)
        pred = torch.argmax(logits[:, -1, :], dim=-1)
        exact_correct += int((pred == target).item())
        anchor_set_correct += int(torch.isin(pred, anchors).item())
        total += 1

    return {
        "multi_anchor_retrieval_accuracy": exact_correct / max(total, 1),
        "multi_anchor_anchor_set_accuracy": anchor_set_correct / max(total, 1),
        "multi_anchor_samples": float(total),
    }


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
    model_dtype = next(model.parameters()).dtype
    maybe_sync_device(device)
    start = time.perf_counter()
    for _ in range(steps):
        with autocast_context(device, model_dtype):
            logits = model(prompt)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        prompt = torch.cat([prompt[:, 1:], next_token], dim=1)
    maybe_sync_device(device)
    elapsed = time.perf_counter() - start
    tokens_per_second = steps / max(elapsed, 1e-8)
    return {
        "decode_steps": float(steps),
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
    }
