from __future__ import annotations

import time

import torch

from hz0.runtime import autocast_context, maybe_sync_device


def _retrieval_vocab_bounds(vocab_size: int) -> tuple[int, int]:
    filler_low = 32 if vocab_size > 64 else 0
    filler_high = min(vocab_size, 127) if vocab_size > 127 else vocab_size
    return filler_low, filler_high


def _sample_non_filler_token(device: torch.device, vocab_size: int) -> torch.Tensor:
    low = min(128, max(vocab_size - 1, 0))
    return torch.randint(low, vocab_size, (1, 1), device=device)


def _sample_distinct_tokens(device: torch.device, vocab_size: int, count: int) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    seen: set[int] = set()
    while len(values) < count:
        token = _sample_non_filler_token(device, vocab_size)
        token_value = int(token.item())
        if token_value in seen:
            continue
        values.append(token)
        seen.add(token_value)
    return values


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
def evaluate_associative_recall(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    num_samples: int = 32,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    model_dtype = next(model.parameters()).dtype
    filler_low, filler_high = _retrieval_vocab_bounds(vocab_size)
    filler_width = max(1, (seq_len - 4) // 2)

    for _ in range(num_samples):
        key, value = _sample_distinct_tokens(device, vocab_size, 2)
        filler = torch.randint(filler_low, filler_high, (1, filler_width), device=device)
        prompt = torch.cat([key, value, filler, key], dim=1)
        prompt = prompt[:, : max(2, seq_len - 1)]
        with autocast_context(device, model_dtype):
            logits = model(prompt)
        pred = torch.argmax(logits[:, -1, :], dim=-1)
        correct += int((pred == value[:, 0]).item())
        total += 1

    return {"associative_recall_accuracy": correct / max(total, 1), "associative_recall_samples": float(total)}


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
def evaluate_overwrite_retrieval(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    num_samples: int = 32,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    model_dtype = next(model.parameters()).dtype
    filler_low, filler_high = _retrieval_vocab_bounds(vocab_size)
    filler_width = max(1, (seq_len - 5) // 3)

    for _ in range(num_samples):
        key, old_value, new_value = _sample_distinct_tokens(device, vocab_size, 3)
        filler_a = torch.randint(filler_low, filler_high, (1, filler_width), device=device)
        filler_b = torch.randint(filler_low, filler_high, (1, filler_width), device=device)
        prompt = torch.cat([key, old_value, filler_a, key, new_value, filler_b, key], dim=1)
        prompt = prompt[:, : max(2, seq_len - 1)]
        with autocast_context(device, model_dtype):
            logits = model(prompt)
        pred = torch.argmax(logits[:, -1, :], dim=-1)
        correct += int((pred == new_value[:, 0]).item())
        total += 1

    return {"overwrite_retrieval_accuracy": correct / max(total, 1), "overwrite_samples": float(total)}


@torch.no_grad()
def evaluate_protected_memory_retrieval(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    num_samples: int = 32,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    model_dtype = next(model.parameters()).dtype
    filler_low, filler_high = _retrieval_vocab_bounds(vocab_size)
    filler_width = max(1, (seq_len - 8) // 4)

    for _ in range(num_samples):
        key_a, value_a_old, key_b, value_b, value_a_new = _sample_distinct_tokens(device, vocab_size, 5)
        filler_a = torch.randint(filler_low, filler_high, (1, filler_width), device=device)
        filler_b = torch.randint(filler_low, filler_high, (1, filler_width), device=device)
        filler_c = torch.randint(filler_low, filler_high, (1, filler_width), device=device)
        prompt = torch.cat(
            [key_a, value_a_old, filler_a, key_b, value_b, filler_b, key_a, value_a_new, filler_c, key_b],
            dim=1,
        )
        prompt = prompt[:, : max(2, seq_len - 1)]
        with autocast_context(device, model_dtype):
            logits = model(prompt)
        pred = torch.argmax(logits[:, -1, :], dim=-1)
        correct += int((pred == value_b[:, 0]).item())
        total += 1

    return {"protected_memory_accuracy": correct / max(total, 1), "protected_memory_samples": float(total)}


@torch.no_grad()
def evaluate_recall_by_distance(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    num_samples: int = 32,
    distances: list[int] | None = None,
) -> dict[str, float]:
    model.eval()
    model_dtype = next(model.parameters()).dtype
    filler_low, filler_high = _retrieval_vocab_bounds(vocab_size)
    if distances is None:
        distances = [32, 64, 128, 256, 512, 1024, 2048]

    metrics: dict[str, float] = {}
    for distance in distances:
        correct = 0
        total = 0
        filler_width = max(1, min(distance, max(seq_len - 3, 1)))
        for _ in range(num_samples):
            key, value = _sample_distinct_tokens(device, vocab_size, 2)
            filler = torch.randint(filler_low, filler_high, (1, filler_width), device=device)
            prompt = torch.cat([key, value, filler, key], dim=1)
            prompt = prompt[:, : max(2, seq_len - 1)]
            with autocast_context(device, model_dtype):
                logits = model(prompt)
            pred = torch.argmax(logits[:, -1, :], dim=-1)
            correct += int((pred == value[:, 0]).item())
            total += 1
        metrics[f"recall_distance_{distance}_accuracy"] = correct / max(total, 1)
    metrics["recall_distance_samples"] = float(num_samples)
    return metrics


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


@torch.no_grad()
def benchmark_decode_by_context(
    model: torch.nn.Module,
    device: torch.device,
    context_lengths: list[int],
    steps: int,
    vocab_size: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for prompt_len in context_lengths:
        result = benchmark_decode_latency(
            model=model,
            device=device,
            prompt_len=prompt_len,
            steps=steps,
            vocab_size=vocab_size,
        )
        key = f"context_{prompt_len}_tokens_per_second"
        metrics[key] = result["tokens_per_second"]
    return metrics
