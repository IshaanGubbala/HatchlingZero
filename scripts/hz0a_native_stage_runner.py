"""Resumable native-Metal MLX Stage 1 runner for the locked HZ-0A topology."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from reference.hz0a_mlx_model import HZ0AMlxModel


def read_batch(handle, batch_size: int, sequence_length: int):
    values = []
    while len(values) < batch_size:
        line = handle.readline()
        if not line:
            handle.seek(0)
            line = handle.readline()
        tokens = json.loads(line)
        if len(tokens) < sequence_length:
            continue
        values.append(tokens[:sequence_length])
    return mx.array(values, dtype=mx.int32)


def model_fingerprint(model) -> str:
    values = [np.asarray(value).tobytes() for _, value in tree_flatten(model.parameters())]
    return hashlib.sha256(b"".join(values)).hexdigest()


def detach_state(state):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(detach_state(item) for item in state)
    return mx.stop_gradient(state)


def lr_at_step(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Pure function of step -> learning rate: linear warmup then cosine decay.

    Deterministic and stateless by design -- resuming only needs to restore
    ``step`` (already checkpointed), not any separate scheduler state.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return max_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = max_lr * min_lr_ratio
    return min_lr + (max_lr - min_lr) * cosine


def assert_finite(label, values):
    arrays = values if isinstance(values, (list, tuple)) else [values]
    mx.eval(*arrays)
    if not all(bool(mx.all(mx.isfinite(value))) for value in arrays):
        raise FloatingPointError(f"native Stage 1 produced non-finite {label}")


def save_checkpoint(path: Path, model, optimizer, step: int, tokens_seen: int, batch_index: int, metrics: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model_values = tree_flatten(model.parameters())
    optimizer_values = tree_flatten(optimizer.state)
    arrays = []
    for group, values in (("model", model_values), ("optimizer", optimizer_values)):
        for index, (key, value) in enumerate(values):
            filename = f"{group}-{index:04d}.npy"
            mx.save(str(path / filename), value)
            arrays.append({"group": group, "key": key, "shape": list(value.shape), "file": filename})
    (path / "state.json").write_text(json.dumps({"step": step, "tokens_seen": tokens_seen, "batch_index": batch_index, "metrics": metrics, "arrays": arrays}), encoding="utf-8")


def restore_checkpoint(path: Path, model, optimizer) -> dict:
    payload = json.loads((path / "state.json").read_text(encoding="utf-8"))
    groups = {"model": [], "optimizer": []}
    for item in payload["arrays"]:
        groups[item["group"]].append((item["key"], mx.load(str(path / item["file"]))))
    model.update(tree_unflatten(groups["model"]))
    optimizer.state = tree_unflatten(groups["optimizer"])
    mx.eval(model.parameters(), optimizer.state)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chunk-length", type=int, default=128)
    parser.add_argument("--truncate-backward", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=24576)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=31)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=2304)
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--reset-attention-state", action="store_true")
    parser.add_argument("--carry-attention-state", action="store_true", help="Retain periodic-attention KV caches across truncated training chunks; off by default")
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--exact-update-norm", action="store_true", help="Clone parameters for exact update norm; expensive and off by default")
    parser.add_argument("--gradient-accumulation-chunks", type=int, default=1)
    parser.add_argument("--gradient-accumulation-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--max-lr", type=float, default=1e-4, help="Peak learning rate after warmup (1e-4 is the Phase 6 sweep optimum)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1, help="Cosine floor as a fraction of --max-lr")
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--architecture", choices=("hybrid", "transformer"), default="hybrid", help="hybrid = periodic GDN-2 recurrence + attention (the A1 spec); transformer = every layer is causal attention (the A10 matched baseline)")
    args = parser.parse_args()
    if args.gradient_accumulation_chunks <= 0:
        raise ValueError("--gradient-accumulation-chunks must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.activation_checkpoint and args.truncate_backward and (args.carry_attention_state and not args.reset_attention_state):
        raise ValueError("--activation-checkpoint is not compatible with carried attention state; use --reset-attention-state or disable activation checkpointing")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    memory_log = args.run_dir / "native_metal_memory.jsonl"
    checkpoint = args.run_dir / "native_metal_checkpoint"
    if not (args.resume and checkpoint.exists()):
        memory_log.write_text("", encoding="utf-8")
    sequence_length = args.sequence_length or len(json.loads(args.data.open().readline()))
    if args.truncate_backward:
        total_chunks = math.ceil(args.target_tokens / (args.batch_size * args.chunk_length))
        total_optimizer_steps = max(1, math.ceil(total_chunks / args.gradient_accumulation_chunks))
    else:
        total_optimizer_steps = max(1, math.ceil(args.target_tokens / (args.batch_size * sequence_length)))
    mx.random.seed(7)
    if args.architecture == "transformer":
        attention = tuple(range(args.layers))
    else:
        attention = tuple(index for index in (4, 9, 14, 19, 24, 29) if index < args.layers)
    model = HZ0AMlxModel(args.vocab_size, args.dim, args.layers, args.heads, args.d_ff, attention, native_metal=True, checkpoint_blocks=args.activation_checkpoint)
    if args.dtype == "float16":
        model.update(tree_unflatten([(key, value.astype(mx.float16)) for key, value in tree_flatten(model.parameters())]))
    def current_lr(at_step: int) -> float:
        if args.lr_schedule == "constant":
            return args.max_lr
        return lr_at_step(at_step, total_optimizer_steps, args.warmup_steps, args.max_lr, args.lr_min_ratio)

    optimizer = optim.AdamW(learning_rate=current_lr(0), weight_decay=0.01)
    metrics, step, tokens_seen, batch_index = [], 0, 0, 0
    if args.resume and checkpoint.exists():
        payload = restore_checkpoint(checkpoint, model, optimizer)
        metrics, step, tokens_seen, batch_index = payload["metrics"], payload["step"], payload["tokens_seen"], payload["batch_index"]
    last_lr = current_lr(step)
    started = time.perf_counter()
    carry_attention_state = args.carry_attention_state and not args.reset_attention_state
    with args.data.open() as train, args.validation_data.open() as validation:
        for _ in range(batch_index):
            read_batch(train, args.batch_size, sequence_length)
        def loss_fn(current, tokens):
            states = None
            logits_parts = []
            for start in range(0, sequence_length, args.chunk_length):
                logits, states = current(tokens[:, start:start + args.chunk_length], states)
                logits_parts.append(logits)
            logits = mx.concatenate(logits_parts, axis=1)
            return mx.mean(nn.losses.cross_entropy(logits[:, :-1], tokens[:, 1:]))
        value_and_grad = nn.value_and_grad(model, loss_fn)
        def chunk_loss(current, chunk, carry):
            logits, _ = current(chunk, carry)
            return mx.mean(nn.losses.cross_entropy(logits[:, :-1], chunk[:, 1:]))
        chunk_value_and_grad = nn.value_and_grad(model, chunk_loss)
        while tokens_seen < args.target_tokens:
            tokens = read_batch(train, args.batch_size, sequence_length)
            chunk_metrics = []
            if args.truncate_backward:
                states = None
                accumulated_grads = None
                accumulated_count = 0
                chunks = range(0, sequence_length, args.chunk_length)
                for start in chunks:
                    chunk = tokens[:, start:start + args.chunk_length]
                    logits, states = model(chunk, states)
                    states = [detach_state(state) if (carry_attention_state or not isinstance(state, tuple)) else None for state in states]
                    mx.eval(*[state for state in states if state is not None for state in (state if isinstance(state, tuple) else (state,))])
                    del logits
                    loss, grads = chunk_value_and_grad(model, chunk, states)
                    mx.eval(loss, grads)
                    assert_finite("loss/gradients", [loss] + [value for _, value in tree_flatten(grads)])
                    flat_grads = [value for _, value in tree_flatten(grads)]
                    grad_norm = float(mx.sqrt(sum(mx.sum(value * value) for value in flat_grads)))
                    if accumulated_grads is None:
                        accumulator_dtype = mx.float16 if args.gradient_accumulation_dtype == "float16" else mx.float32
                        accumulated_grads = [(key, value.astype(accumulator_dtype)) for key, value in tree_flatten(grads)]
                        mx.eval(*[value for _, value in accumulated_grads])
                    else:
                        previous_grads = accumulated_grads
                        accumulated_grads = []
                        for (key, previous), (_, current) in zip(previous_grads, tree_flatten(grads)):
                            value = previous + current.astype(previous.dtype)
                            mx.eval(value)
                            accumulated_grads.append((key, mx.array(value)))
                        mx.eval(*[value for _, value in accumulated_grads])
                        del previous_grads
                    accumulated_count += 1
                    final_chunk = start + args.chunk_length >= sequence_length
                    should_update = accumulated_count >= args.gradient_accumulation_chunks or final_chunk
                    old = [mx.array(value) for _, value in tree_flatten(model.parameters())] if should_update and args.exact_update_norm else None
                    new = []
                    update_norm = None
                    if should_update:
                        last_lr = current_lr(step)
                        optimizer.learning_rate = last_lr
                        optimizer.update(model, tree_unflatten([(key, (value / accumulated_count).astype(mx.float32)) for key, value in accumulated_grads]))
                        mx.eval(loss, model.parameters(), optimizer.state)
                        new = [value for _, value in tree_flatten(model.parameters())]
                        assert_finite("updated parameters", new)
                        update_norm = float(mx.sqrt(sum(mx.sum((a - b) * (a - b)) for a, b in zip(new, old)))) if old is not None else None
                        step += 1
                        accumulated_grads = None
                        accumulated_count = 0
                    mx.clear_cache()
                    gc.collect()
                    active_memory = int(mx.get_active_memory())
                    cache_memory = int(mx.get_cache_memory())
                    peak_memory = int(mx.get_peak_memory())
                    tokens_seen += args.batch_size * chunk.shape[1]
                    chunk_metrics.append({"step": step, "tokens_seen": tokens_seen, "loss": float(loss), "gradient_norm": grad_norm, "update_norm": update_norm, "lr": last_lr, "wall_time": time.perf_counter() - started, "active_memory_bytes": active_memory, "cache_memory_bytes": cache_memory, "peak_memory_bytes": peak_memory})
                    with memory_log.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(chunk_metrics[-1]) + "\n")
                    del grads, old, new, loss, chunk
                    gc.collect()
                batch_index += 1
                item = chunk_metrics[-1]
            else:
                loss, grads = value_and_grad(model, tokens)
                mx.eval(loss, grads)
                assert_finite("loss/gradients", [loss] + [value for _, value in tree_flatten(grads)])
                flat_grads = [value for _, value in tree_flatten(grads)]
                grad_norm = float(mx.sqrt(sum(mx.sum(value * value) for value in flat_grads)))
                old = [mx.array(value) for _, value in tree_flatten(model.parameters())] if args.exact_update_norm else None
                last_lr = current_lr(step)
                optimizer.learning_rate = last_lr
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)
                new = [value for _, value in tree_flatten(model.parameters())]
                assert_finite("updated parameters", new)
                update_norm = float(mx.sqrt(sum(mx.sum((a - b) * (a - b)) for a, b in zip(new, old)))) if old is not None else None
                step += 1; batch_index += 1; tokens_seen += args.batch_size * sequence_length
                item = {"step": step, "tokens_seen": tokens_seen, "loss": float(loss), "gradient_norm": grad_norm, "update_norm": update_norm, "lr": last_lr, "wall_time": time.perf_counter() - started}
                with memory_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item) + "\n")
                gc.collect()
            if step % args.validation_interval == 0 or tokens_seen >= args.target_tokens:
                validation_tokens = read_batch(validation, 1, sequence_length)
                validation_logits, _ = model(validation_tokens)
                validation_loss = mx.mean(nn.losses.cross_entropy(validation_logits[:, :-1], validation_tokens[:, 1:]))
                mx.eval(validation_loss)
                item["validation_loss"] = float(validation_loss)
            metrics.append(item)
            if step % args.checkpoint_interval == 0 or tokens_seen >= args.target_tokens:
                save_checkpoint(checkpoint, model, optimizer, step, tokens_seen, batch_index, metrics)
    report = {"backend": "native_metal_mlx", "architecture": args.architecture, "stage": "stage1_validation", "dtype": args.dtype, "activation_checkpoint": args.activation_checkpoint, "chunk_length": args.chunk_length, "gradient_accumulation_chunks": args.gradient_accumulation_chunks, "gradient_accumulation_dtype": args.gradient_accumulation_dtype, "reset_attention_state": args.reset_attention_state, "carry_attention_state": carry_attention_state, "lr_schedule": args.lr_schedule, "max_lr": args.max_lr, "warmup_steps": args.warmup_steps, "lr_min_ratio": args.lr_min_ratio, "total_optimizer_steps_estimate": total_optimizer_steps, "final_lr": last_lr, "steps": step, "tokens_seen": tokens_seen, "target_tokens": args.target_tokens, "budget_complete": tokens_seen >= args.target_tokens, "parameter_count": sum(value.size for _, value in tree_flatten(model.parameters())), "initialization_seed": 7, "final_parameter_sha256": model_fingerprint(model), "metrics": metrics, "checkpoint": str(checkpoint), "training_seconds": time.perf_counter() - started, "tokens_per_second": tokens_seen / max(time.perf_counter() - started, 1e-9), "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (args.run_dir / "native_metal.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
