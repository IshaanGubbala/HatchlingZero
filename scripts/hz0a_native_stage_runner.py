"""Resumable native-Metal MLX Stage 1 runner for the locked HZ-0A topology."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from reference.hz0a_mlx_model import HZ0AMlxModel


def read_batch(handle, batch_size: int, sequence_length: int, epoch_counter: list[int] | None = None):
    values = []
    while len(values) < batch_size:
        line = handle.readline()
        if not line:
            handle.seek(0)
            if epoch_counter is not None:
                epoch_counter[0] += 1
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
    """One aggregate host round-trip instead of one bool() per parameter leaf.

    The naive per-leaf `bool(mx.all(mx.isfinite(v)))` loop forces a separate
    synchronous device-to-host transfer for every leaf (hundreds for a
    300M-param model, called at least twice per chunk) -- stacking the
    per-leaf flags and checking once cuts that to a single sync point.
    """
    arrays = values if isinstance(values, (list, tuple)) else [values]
    finite_flags = mx.stack([mx.all(mx.isfinite(value)) for value in arrays])
    if not bool(mx.all(finite_flags)):
        raise FloatingPointError(f"native Stage 1 produced non-finite {label}")


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    step: int,
    tokens_seen: int,
    batch_index: int,
    metrics: list[dict],
    microbatch_count: int = 0,
    epoch_or_data_pass: int = 0,
    best_validation_loss: float | None = None,
    milestones_hit: list[int] | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model_values = tree_flatten(model.parameters())
    optimizer_values = tree_flatten(optimizer.state)
    arrays = []
    for group, values in (("model", model_values), ("optimizer", optimizer_values)):
        for index, (key, value) in enumerate(values):
            filename = f"{group}-{index:04d}.npy"
            mx.save(str(path / filename), value)
            arrays.append({"group": group, "key": key, "shape": list(value.shape), "file": filename})
    payload = {
        "step": step,
        "tokens_seen": tokens_seen,
        "batch_index": batch_index,
        "microbatch_count": microbatch_count,
        "epoch_or_data_pass": epoch_or_data_pass,
        "best_validation_loss": best_validation_loss,
        "milestones_hit": milestones_hit or [],
        "metrics": metrics,
        "arrays": arrays,
    }
    (path / "state.json").write_text(json.dumps(payload), encoding="utf-8")


def snapshot_checkpoint(source: Path, destination: Path) -> None:
    """Copy a just-written checkpoint directory to a separate, never-overwritten
    path -- for milestone and best-validation snapshots (the earlier Stage 1
    run only kept one rolling checkpoint slot, so its "best" mid-run weights
    were unrecoverably overwritten by later saves; this fixes that)."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


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
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32", help="float16 was rejected earlier this session for NaN; bfloat16 is a separate, verified-stable lower-precision path with a wider exponent range")
    parser.add_argument("--reset-attention-state", action="store_true")
    parser.add_argument("--carry-attention-state", action="store_true", help="Retain periodic-attention KV caches across truncated training chunks; off by default")
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--exact-update-norm", action="store_true", help="Clone parameters for exact update norm; expensive and off by default")
    parser.add_argument("--compile-step", action="store_true", help="mx.compile the per-chunk forward+backward and the optimizer update as two separate compiled functions (verified bit-exact against the uncompiled path; real ~2.65x throughput at the locked 301M scale in isolated benchmarks). Off by default -- opt in explicitly.")
    parser.add_argument("--gradient-accumulation-chunks", type=int, default=1)
    parser.add_argument("--gradient-accumulation-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--max-lr", type=float, default=1e-4, help="Peak learning rate after warmup (1e-4 is the Phase 6 sweep optimum)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1, help="Cosine floor as a fraction of --max-lr")
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--architecture", choices=("hybrid", "transformer"), default="hybrid", help="hybrid = periodic GDN-2 recurrence + attention (the A1 spec); transformer = every layer is causal attention (the A10 matched baseline)")
    parser.add_argument("--mixer", choices=("gdn2", "gdn2_fix"), default="gdn2", help="recurrent mixer; gdn2 is the frozen baseline, gdn2_fix is the opt-in exact vector-gated successor")
    parser.add_argument("--seed", type=int, default=7, help="Initialization seed (data order is deterministic/sequential regardless of seed)")
    parser.add_argument("--validation-batch-size", type=int, default=32, help="Fixed number of validation sequences read once at startup and reused for every validation check (was a single rotating sequence -- high variance, not comparable across runs)")
    parser.add_argument("--milestone-tokens", type=str, default="", help="Comma-separated token counts (e.g. 25000000,50000000,75000000,100000000) to preserve as separate, never-overwritten checkpoint snapshots -- in addition to the regular rolling checkpoint. Fixes the earlier mistake of only keeping one overwritten checkpoint slot.")
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
        effective_batch_tokens = args.batch_size * args.chunk_length * args.gradient_accumulation_chunks
    else:
        total_optimizer_steps = max(1, math.ceil(args.target_tokens / (args.batch_size * sequence_length)))
        effective_batch_tokens = args.batch_size * sequence_length
    # Configuration snapshot per run (A7 requirement): written before training
    # starts so a mid-run crash still leaves a record of exactly what ran.
    config_snapshot = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config_snapshot["sequence_length_resolved"] = sequence_length
    config_snapshot["effective_batch_tokens"] = effective_batch_tokens
    config_snapshot["total_optimizer_steps_estimate"] = total_optimizer_steps
    (args.run_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2, sort_keys=True), encoding="utf-8")
    mx.random.seed(args.seed)
    if args.architecture == "transformer":
        attention = tuple(range(args.layers))
    else:
        attention = tuple(index for index in (4, 9, 14, 19, 24, 29) if index < args.layers)
    model = HZ0AMlxModel(args.vocab_size, args.dim, args.layers, args.heads, args.d_ff, attention, native_metal=True, checkpoint_blocks=args.activation_checkpoint, mixer=args.mixer)
    if args.dtype == "float16":
        model.update(tree_unflatten([(key, value.astype(mx.float16)) for key, value in tree_flatten(model.parameters())]))
    elif args.dtype == "bfloat16":
        # The GDN-2 kernel's recurrent-state accumulator is hardcoded fp32
        # internally (thread float state[64] in reference/hz0a_mlx_metal.py,
        # regardless of the templated I/O DType) -- casting parameters to
        # bf16 does not touch that. Loss reduction is separately forced back
        # to fp32 below regardless of model dtype, matching the same
        # mixed-precision policy already used for the optimizer's own
        # gradient input (already always fp32, see the accumulation loop).
        model.update(tree_unflatten([(key, value.astype(mx.bfloat16)) for key, value in tree_flatten(model.parameters())]))
    def current_lr(at_step: int) -> float:
        if args.lr_schedule == "constant":
            return args.max_lr
        return lr_at_step(at_step, total_optimizer_steps, args.warmup_steps, args.max_lr, args.lr_min_ratio)

    optimizer = optim.AdamW(learning_rate=current_lr(0), weight_decay=0.01)
    metrics, step, tokens_seen, batch_index = [], 0, 0, 0
    microbatch_count, epoch_or_data_pass = 0, 0
    best_validation_loss, milestones_hit = None, []
    milestone_tokens = sorted({int(value) for value in args.milestone_tokens.split(",") if value.strip()})
    if args.resume and checkpoint.exists():
        payload = restore_checkpoint(checkpoint, model, optimizer)
        metrics, step, tokens_seen, batch_index = payload["metrics"], payload["step"], payload["tokens_seen"], payload["batch_index"]
        microbatch_count = payload.get("microbatch_count", 0)
        epoch_or_data_pass = payload.get("epoch_or_data_pass", 0)
        best_validation_loss = payload.get("best_validation_loss")
        milestones_hit = payload.get("milestones_hit", [])
    last_lr = current_lr(step)
    started = time.perf_counter()
    carry_attention_state = args.carry_attention_state and not args.reset_attention_state
    epoch_counter = [epoch_or_data_pass]
    with args.data.open() as train, args.validation_data.open() as validation:
        for _ in range(batch_index):
            # Fast-forwarding the file cursor to a previously-reached resume
            # position -- these reads already happened once and are already
            # reflected in the restored epoch_or_data_pass, so use a
            # throwaway counter here rather than double-counting wraps.
            read_batch(train, args.batch_size, sequence_length, [0])
        # Fixed, deterministic validation set read once and reused for every
        # check -- previously this was a single sequence off a rotating
        # cursor (read_batch(validation, 1, ...) called fresh each time),
        # which made validation_loss a high-variance single-sample estimate
        # rather than a stable signal comparable across checkpoints/runs.
        fixed_validation_tokens = read_batch(validation, args.validation_batch_size, sequence_length)
        mx.eval(fixed_validation_tokens)
        def evaluate_fixed_validation(current, sub_batch: int = 8) -> float:
            total, count = 0.0, 0
            for start in range(0, fixed_validation_tokens.shape[0], sub_batch):
                chunk = fixed_validation_tokens[start:start + sub_batch]
                logits, _ = current(chunk)
                loss = mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), chunk[:, 1:]))
                mx.eval(loss)
                total += float(loss)
                count += 1
            return total / count
        def loss_fn(current, tokens):
            states = None
            logits_parts = []
            for start in range(0, sequence_length, args.chunk_length):
                logits, states = current(tokens[:, start:start + args.chunk_length], states)
                logits_parts.append(logits)
            logits = mx.concatenate(logits_parts, axis=1)
            return mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), tokens[:, 1:]))
        value_and_grad = nn.value_and_grad(model, loss_fn)
        def chunk_loss(current, chunk, carry):
            # next_state returned as an aux output (mx.value_and_grad only
            # differentiates the first/scalar element of a returned tuple)
            # so the caller gets the CORRECT post-chunk state from the same
            # forward pass used for loss/grad, instead of running a second,
            # separate throwaway forward just to compute it -- that second
            # forward previously used a mismatched carry-in (see the fix
            # commit this accompanies for the full bug writeup).
            logits, next_state = current(chunk, carry)
            # Loss reduction forced to fp32 regardless of model dtype (the
            # softmax/log-sum-exp inside cross_entropy is precision-sensitive;
            # bf16 logits are fine as matmul/activation storage but risk real
            # numerical error accumulated over a full-vocab reduction).
            loss = mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), chunk[:, 1:]))
            return loss, next_state
        chunk_value_and_grad = nn.value_and_grad(model, chunk_loss)

        def chunk_forward_backward(chunk, carry):
            return chunk_value_and_grad(model, chunk, carry)

        def apply_optimizer_update(averaged_grads):
            optimizer.update(model, averaged_grads)

        if args.compile_step:
            # mx.compile requires the SAME arrays in both `inputs=` and
            # `outputs=` whenever a value produced by one call feeds back as
            # an argument to a later call (here: `carry`/`states` -- the
            # model's own recurrent state -- round-tripping across chunk
            # iterations) -- omitting `outputs=` here raises "Attempting to
            # eval an array without a primitive" on the second call, even
            # though this specific function does not itself mutate
            # model.state (confirmed by direct experiment before adopting
            # this, not assumed from documentation alone).
            chunk_forward_backward = mx.compile(chunk_forward_backward, inputs=model.state, outputs=model.state)
            # Kept as a SEPARATE compiled function from the forward/backward
            # above (rather than one combined compiled step) because the
            # optimizer update only happens every gradient_accumulation_chunks
            # chunks, not every chunk -- a single compiled function can't
            # cleanly express that data-independent-but-call-count-dependent
            # branch. Verified bit-exact against the fully uncompiled
            # accumulation path before adopting (see the commit this
            # accompanies).
            apply_optimizer_update = mx.compile(apply_optimizer_update, inputs=[model.state, optimizer.state], outputs=[model.state, optimizer.state])
        while tokens_seen < args.target_tokens:
            tokens = read_batch(train, args.batch_size, sequence_length, epoch_counter)
            chunk_metrics = []
            if args.truncate_backward:
                states = None
                accumulated_grads = None
                accumulated_count = 0
                chunks = range(0, sequence_length, args.chunk_length)
                for start in chunks:
                    chunk = tokens[:, start:start + args.chunk_length]
                    (loss, next_state), grads = chunk_forward_backward(chunk, states)
                    mx.eval(loss, grads, *[state for state in next_state if state is not None for state in (state if isinstance(state, tuple) else (state,))])
                    states = [detach_state(state) if (carry_attention_state or not isinstance(state, tuple)) else None for state in next_state]
                    assert_finite("loss/gradients", [loss] + [value for _, value in tree_flatten(grads)])
                    flat_grads = [value for _, value in tree_flatten(grads)]
                    grad_norm = float(mx.sqrt(sum(mx.sum(value * value) for value in flat_grads)))
                    if accumulated_grads is None:
                        accumulator_dtype = mx.float16 if args.gradient_accumulation_dtype == "float16" else mx.float32
                        accumulated_grads = [(key, value.astype(accumulator_dtype)) for key, value in tree_flatten(grads)]
                        mx.eval(*[value for _, value in accumulated_grads])
                    else:
                        previous_grads = accumulated_grads
                        # previous/current are both already materialized (evaluated
                        # on the accumulator-init branch and on line ~290
                        # respectively), so summing them is a shallow op with no
                        # lazy-graph buildup risk -- one batched eval after the loop
                        # is sufficient; evaluating each parameter individually
                        # inside the loop (as a since-fixed refactor accidentally
                        # left this) was a redundant sync point per parameter tensor.
                        accumulated_grads = [(key, previous + current.astype(previous.dtype)) for (key, previous), (_, current) in zip(previous_grads, tree_flatten(grads))]
                        mx.eval(*[value for _, value in accumulated_grads])
                        del previous_grads
                    accumulated_count += 1
                    microbatch_count += 1
                    final_chunk = start + args.chunk_length >= sequence_length
                    should_update = accumulated_count >= args.gradient_accumulation_chunks or final_chunk
                    old = [mx.array(value) for _, value in tree_flatten(model.parameters())] if should_update and args.exact_update_norm else None
                    new = []
                    update_norm = None
                    if should_update:
                        last_lr = current_lr(step)
                        optimizer.learning_rate = last_lr
                        apply_optimizer_update(tree_unflatten([(key, (value / accumulated_count).astype(mx.float32)) for key, value in accumulated_grads]))
                        mx.eval(loss, model.parameters(), optimizer.state)
                        new = [value for _, value in tree_flatten(model.parameters())]
                        assert_finite("updated parameters", new)
                        update_norm = float(mx.sqrt(sum(mx.sum((a - b) * (a - b)) for a, b in zip(new, old)))) if old is not None else None
                        step += 1
                        accumulated_grads = None
                        accumulated_count = 0
                        # Once per optimizer step (their original intent per the
                        # commits that added them: "Stabilize native Stage 1 chunk
                        # allocator" / "reduce native runner peak memory"), not
                        # every chunk -- a later accumulation-loop refactor left
                        # these unindented, accidentally doubling their frequency
                        # (at gradient_accumulation_chunks=2) with no memory-safety
                        # benefit over the originally-intended cadence.
                        mx.clear_cache()
                        gc.collect()
                    active_memory = int(mx.get_active_memory())
                    cache_memory = int(mx.get_cache_memory())
                    peak_memory = int(mx.get_peak_memory())
                    tokens_seen += args.batch_size * chunk.shape[1]
                    chunk_metrics.append({"step": step, "tokens_seen": tokens_seen, "loss": float(loss), "gradient_norm": grad_norm, "update_norm": update_norm, "lr": last_lr, "wall_time": time.perf_counter() - started, "active_memory_bytes": active_memory, "cache_memory_bytes": cache_memory, "peak_memory_bytes": peak_memory, "microbatch_count": microbatch_count, "epoch_or_data_pass": epoch_counter[0]})
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
                microbatch_count += 1
                item = {"step": step, "tokens_seen": tokens_seen, "loss": float(loss), "gradient_norm": grad_norm, "update_norm": update_norm, "lr": last_lr, "wall_time": time.perf_counter() - started, "microbatch_count": microbatch_count, "epoch_or_data_pass": epoch_counter[0]}
                with memory_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item) + "\n")
                gc.collect()
            is_new_best = False
            if step % args.validation_interval == 0 or tokens_seen >= args.target_tokens:
                item["validation_loss"] = evaluate_fixed_validation(model)
                if best_validation_loss is None or item["validation_loss"] < best_validation_loss:
                    best_validation_loss = item["validation_loss"]
                    is_new_best = True
            metrics.append(item)
            if step % args.checkpoint_interval == 0 or tokens_seen >= args.target_tokens:
                save_checkpoint(checkpoint, model, optimizer, step, tokens_seen, batch_index, metrics, microbatch_count=microbatch_count, epoch_or_data_pass=epoch_counter[0], best_validation_loss=best_validation_loss, milestones_hit=milestones_hit)
                if is_new_best:
                    snapshot_checkpoint(checkpoint, checkpoint.parent / f"{checkpoint.name}_best")
                for target in milestone_tokens:
                    if tokens_seen >= target and target not in milestones_hit:
                        snapshot_checkpoint(checkpoint, checkpoint.parent / f"{checkpoint.name}_milestone_{target}")
                        milestones_hit.append(target)
    report = {"backend": "native_metal_mlx", "architecture": args.architecture, "mixer": args.mixer, "stage": "stage1_validation", "dtype": args.dtype, "activation_checkpoint": args.activation_checkpoint, "chunk_length": args.chunk_length, "gradient_accumulation_chunks": args.gradient_accumulation_chunks, "gradient_accumulation_dtype": args.gradient_accumulation_dtype, "reset_attention_state": args.reset_attention_state, "carry_attention_state": carry_attention_state, "lr_schedule": args.lr_schedule, "max_lr": args.max_lr, "warmup_steps": args.warmup_steps, "lr_min_ratio": args.lr_min_ratio, "total_optimizer_steps_estimate": total_optimizer_steps, "final_lr": last_lr, "validation_batch_size": args.validation_batch_size, "steps": step, "microbatch_count": microbatch_count, "epoch_or_data_pass": epoch_counter[0], "best_validation_loss": best_validation_loss, "milestones_hit": milestones_hit, "tokens_seen": tokens_seen, "target_tokens": args.target_tokens, "budget_complete": tokens_seen >= args.target_tokens, "parameter_count": sum(value.size for _, value in tree_flatten(model.parameters())), "initialization_seed": args.seed, "final_parameter_sha256": model_fingerprint(model), "metrics": metrics, "checkpoint": str(checkpoint), "training_seconds": time.perf_counter() - started, "tokens_per_second": tokens_seen / max(time.perf_counter() - started, 1e-9), "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (args.run_dir / "native_metal.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
