"""HZ-Core-1 real training runner for the Value Bottleneck arm
(`plans/HZ Integrated Candidate Plan.md` Step 6): identical CLI,
data-loading, milestone/checkpoint, and report-JSON conventions to
`scripts/hz0h_stage2_runner_bdh.py` (itself matching
`scripts/hz0a_torch_stage2_runner.py`), so a BDH run, a BDHVB run, and a
`--architecture transformer` run are all directly comparable side by
side -- same corpus, same batching, same LR schedule family, same
report shape. Deliberately NOT a generic "pass a model class" refactor
of the exact-BDH runner -- keeping the two scripts textually parallel
means either can be read/audited standalone, matching this session's
established convention (see `reference/hz0h_bdh_gs_torch.py` vs
`reference/hz0h_bdh_soft_grouped_state_torch.py`, kept as separate
files for the same reason).

Uses `reference/hz0h_bdh_vb_torch.py`'s `BDHVB`/`BDHVBConfig` --
confirmed fully vectorized (same whole-sequence forward shape as exact
BDH, no per-token streaming loop needed for training, unlike BDHGSP/
BDHSoftGroupedState). `d_state` is locked via
`reference/hz0h_state_v1.py`'s `hz_state_v1_config` (`n_embd // 4`, the
SAFE point on the reassignment-task compression cliff, not the more
aggressive D/8 setting) -- this IS the "value bottleneck" arm of
HZ-Core-1's 4-way ablation. INT8 state quantization is NOT applied
during training here (state only matters across streaming calls, and
training uses the whole-sequence vectorized forward with no persistent
state at all) -- INT8's real effect is evaluated separately, post-hoc,
on this SAME trained checkpoint via `bdh_vb_stream_chunk_int8_state`
(streaming-only), matching how every other INT8 result this session was
produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_train_torch import build_optimizer, shifted_target_batch
from reference.hz0h_energy import TrainingEnergySampler
from reference.hz0h_bdh_vb_torch import BDHVB
from reference.hz0h_state_v1 import hz_state_v1_config


def read_batch(handle, batch_size: int, sequence_length: int, device, epoch_counter: list[int] | None = None) -> torch.Tensor:
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
    return torch.tensor(np.asarray(values, dtype=np.int64), device=device)


def lr_at_step(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr_ratio: float = 0.1) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return max_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = max_lr * min_lr_ratio
    return min_lr + (max_lr - min_lr) * cosine


def model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().to("cpu", torch.float32).numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path: Path, model, optimizer, step, tokens_seen, batch_index, metrics, epoch_or_data_pass, best_validation_loss, milestones_hit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, str(path) + ".pt")
    payload = {
        "step": step, "tokens_seen": tokens_seen, "batch_index": batch_index,
        "epoch_or_data_pass": epoch_or_data_pass, "best_validation_loss": best_validation_loss,
        "milestones_hit": milestones_hit or [], "metrics": metrics,
    }
    Path(str(path) + ".json").write_text(json.dumps(payload), encoding="utf-8")


def restore_checkpoint(path: Path, model, optimizer, device) -> dict:
    blob = torch.load(str(path) + ".pt", map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    return json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))


def snapshot_checkpoint(path: Path, destination: Path) -> None:
    import shutil
    shutil.copy(str(path) + ".pt", str(destination) + ".pt")
    shutil.copy(str(path) + ".json", str(destination) + ".json")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


_mps_peak_bytes = [0]


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        current = int(torch.mps.current_allocated_memory())
        _mps_peak_bytes[0] = max(_mps_peak_bytes[0], current)
        return _mps_peak_bytes[0]
    if device.type == "cpu":
        import resource
        import sys
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return maxrss if sys.platform == "darwin" else maxrss * 1024
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=25_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--validation-interval", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.1, help="real train.py default")
    parser.add_argument("--max-lr", type=float, default=1e-3, help="real train.py default")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--milestone-tokens", type=str, default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    memory_log = args.run_dir / "bdh_vb_stage2_memory.jsonl"
    checkpoint = args.run_dir / "bdh_vb_stage2_checkpoint"
    if not (args.resume and Path(str(checkpoint) + ".pt").exists()):
        memory_log.write_text("", encoding="utf-8")

    sequence_length = args.sequence_length or len(json.loads(args.data.open().readline()))
    total_optimizer_steps = max(1, math.ceil(args.target_tokens / (args.batch_size * sequence_length)))
    effective_batch_tokens = args.batch_size * sequence_length

    torch.manual_seed(args.seed)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    bdh_vb_config = hz_state_v1_config(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size, dropout=args.dropout,
    )
    model = BDHVB(bdh_vb_config).to(device=device, dtype=torch_dtype)
    # Same real dtype-cast subtlety as the exact-BDH runner: RoPE's
    # frequency buffer must stay float32 (Attention.forward's own
    # `assert self.freqs.dtype == torch.float32`).
    model.attn.freqs = model.attn.freqs.to(torch.float32)

    config_snapshot = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config_snapshot.update(sequence_length_resolved=sequence_length, effective_batch_tokens=effective_batch_tokens, total_optimizer_steps_estimate=total_optimizer_steps, resolved_device=str(device), backend="torch", architecture="bdh_vb", d_state=bdh_vb_config.d_state, parameter_count=sum(p.numel() for p in model.parameters()))
    (args.run_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2, sort_keys=True), encoding="utf-8")

    def current_lr(at_step: int) -> float:
        if args.lr_schedule == "constant":
            return args.max_lr
        return lr_at_step(at_step, total_optimizer_steps, args.warmup_steps, args.max_lr, args.lr_min_ratio)

    optimizer = build_optimizer(model, lr=current_lr(0), weight_decay=args.weight_decay)
    metrics, step, tokens_seen, batch_index = [], 0, 0, 0
    epoch_or_data_pass = 0
    best_validation_loss, milestones_hit = None, []
    milestone_tokens = sorted({int(value) for value in args.milestone_tokens.split(",") if value.strip()})

    if args.resume and Path(str(checkpoint) + ".pt").exists():
        payload = restore_checkpoint(checkpoint, model, optimizer, device)
        metrics, step, tokens_seen, batch_index = payload["metrics"], payload["step"], payload["tokens_seen"], payload["batch_index"]
        epoch_or_data_pass = payload.get("epoch_or_data_pass", 0)
        best_validation_loss = payload.get("best_validation_loss")
        milestones_hit = payload.get("milestones_hit", [])

    last_lr = current_lr(step)
    started = time.perf_counter()
    energy_sampler = TrainingEnergySampler()
    energy_sampler.start()
    epoch_counter = [epoch_or_data_pass]

    with args.data.open() as train, args.validation_data.open() as validation:
        for _ in range(batch_index):
            read_batch(train, args.batch_size, sequence_length, device, [0])
        fixed_validation_tokens = read_batch(validation, args.validation_batch_size, sequence_length, device)

        @torch.no_grad()
        def evaluate_fixed_validation(sub_batch: int = 8) -> float:
            model.eval()
            total, count = 0.0, 0
            for start in range(0, fixed_validation_tokens.shape[0], sub_batch):
                chunk = fixed_validation_tokens[start:start + sub_batch]
                x, y = shifted_target_batch(chunk)
                _logits, loss = model(x, targets=y)
                total += float(loss)
                count += 1
            model.train()
            return total / count

        reset_peak_memory(device)
        model.train()
        while tokens_seen < args.target_tokens:
            tokens = read_batch(train, args.batch_size, sequence_length, device, epoch_counter)
            x, y = shifted_target_batch(tokens)
            optimizer.zero_grad(set_to_none=True)
            _logits, loss = model(x, targets=y)
            loss.backward()
            grad_norm = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in model.parameters() if p.grad is not None], 2.0))))
            last_lr = current_lr(step)
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            step += 1
            batch_index += 1
            tokens_seen += args.batch_size * sequence_length
            item = {"step": step, "tokens_seen": tokens_seen, "loss": float(loss), "gradient_norm": grad_norm, "lr": last_lr, "wall_time": time.perf_counter() - started, "epoch_or_data_pass": epoch_counter[0], "peak_memory_bytes": peak_memory_bytes(device)}

            is_new_best = False
            if step % args.validation_interval == 0 or tokens_seen >= args.target_tokens:
                item["validation_loss"] = evaluate_fixed_validation()
                if best_validation_loss is None or item["validation_loss"] < best_validation_loss:
                    best_validation_loss = item["validation_loss"]
                    is_new_best = True
            with memory_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item) + "\n")
            metrics.append(item)
            if step % args.checkpoint_interval == 0 or tokens_seen >= args.target_tokens:
                save_checkpoint(checkpoint, model, optimizer, step, tokens_seen, batch_index, metrics, epoch_counter[0], best_validation_loss, milestones_hit)
                if is_new_best:
                    snapshot_checkpoint(checkpoint, checkpoint.parent / f"{checkpoint.name}_best")
                for target in milestone_tokens:
                    if tokens_seen >= target and target not in milestones_hit:
                        snapshot_checkpoint(checkpoint, checkpoint.parent / f"{checkpoint.name}_milestone_{target}")
                        milestones_hit.append(target)

    report = {
        "backend": "torch", "device": str(device), "architecture": "bdh_vb", "dtype": args.dtype,
        "d_state": bdh_vb_config.d_state,
        "lr_schedule": args.lr_schedule, "max_lr": args.max_lr, "warmup_steps": args.warmup_steps, "lr_min_ratio": args.lr_min_ratio,
        "total_optimizer_steps_estimate": total_optimizer_steps, "final_lr": last_lr, "validation_batch_size": args.validation_batch_size,
        "steps": step, "epoch_or_data_pass": epoch_counter[0],
        "best_validation_loss": best_validation_loss, "milestones_hit": milestones_hit, "tokens_seen": tokens_seen,
        "target_tokens": args.target_tokens, "budget_complete": tokens_seen >= args.target_tokens,
        "parameter_count": sum(p.numel() for p in model.parameters()), "initialization_seed": args.seed,
        "final_parameter_sha256": model_fingerprint(model), "metrics": metrics, "checkpoint": str(checkpoint),
        "training_seconds": time.perf_counter() - started, "tokens_per_second": tokens_seen / max(time.perf_counter() - started, 1e-9),
        "peak_memory_bytes": peak_memory_bytes(device),
    }
    report.update(energy_sampler.stop(tokens=tokens_seen))
    (args.run_dir / "bdh_vb_stage2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
