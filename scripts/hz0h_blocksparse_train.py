"""HZ-0H initial test: matched-scale BDH-GPU runner, mirroring
`scripts/hz0a_torch_stage2_runner.py`'s data-loading, milestone/checkpoint,
and report-JSON conventions exactly (same `read_batch`, same
`lr_at_step` cosine schedule, same `torch_stage2.json` report shape) so a
BDH run and a `--architecture transformer` run of that script are directly
comparable side by side -- same corpus, same batching, same LR schedule
family, same report format.

Uses the verbatim-real `reference/hz0h_bdh_torch.py` model and
`reference/hz0h_bdh_train_torch.py`'s real recipe pieces
(`shifted_target_batch`, `build_optimizer`) -- the real official
`AdamW(model.parameters(), lr, weight_decay=0.1)` over every parameter,
no separate treatment of the shared/tied `encoder`/`encoder_v`/`decoder`.
Diverges from the real upstream recipe in one deliberate way: uses a
cosine LR schedule with warmup (matching this project's other matched
runs) instead of upstream's own constant LR, so the comparison against
the Transformer runner isn't confounded by a different schedule family --
disclosed, not silent.

BDH's own `forward(idx, targets)` computes cross-entropy internally
(unlike `HZ0AModel`/`MatchedTransformerLM`, which return logits only);
this script still calls `shifted_target_batch` on the loaded
sequence-length tokens and passes `(x, y)` in, matching the Transformer
runner's `logits[:, :-1]` vs `tokens[:, 1:]` supervision exactly (both
get sequence_length-1 supervised positions per row) and counts
`tokens_seen` the same way (`batch_size * sequence_length`, i.e. every
loaded token, not just supervised ones) for a directly comparable token
budget.
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

from reference.hz0h_bdh_torch import BDH, BDHConfig, compute_activation_and_state_diagnostics
from reference.hz0h_bdh_train_torch import shifted_target_batch, build_optimizer
from reference.hz0h_energy import TrainingEnergySampler
from reference.hz0h_bdh_blocksparse_torch import (
    bdh_blocksparse_forward, block_balance_loss, compute_active_blocks,
)


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
    # MPS/CPU: no native peak-reset API -- peak_memory_bytes() below tracks
    # a running max manually for MPS, and CPU's ru_maxrss is a whole-process
    # lifetime peak that can't be reset mid-process anyway (real platform
    # limitation, not an oversight).


_mps_peak_bytes = [0]


def peak_memory_bytes(device: torch.device) -> int | None:
    """Real peak memory, not an estimate. CUDA has a native peak counter.
    MPS has no equivalent (torch.mps only exposes a current-allocation
    snapshot) -- tracked here by keeping a running max of that snapshot
    across calls, which misses any peak that occurs strictly BETWEEN two
    calls (e.g. mid-backward-pass) -- a real, disclosed undercount, not a
    silent one. CPU uses resource.getrusage's maxrss (whole-process
    lifetime peak, platform units differ: bytes on macOS, KB on Linux)."""
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
    parser.add_argument("--compile-step", action="store_true", help="torch.compile the model's forward call. Checkpoints are still saved from the ORIGINAL (uncompiled) model object -- torch.compile wraps but does not copy parameters, so this stays a plain, portable BDH state_dict, loadable by every existing eval script unchanged. Off by default; see docs/restart/hz0h_phase6_depth_curriculum_results.md for the real speed/correctness measurement this flag was validated against on CUDA before being recommended.")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default", help="torch.compile's `mode` argument, only used if --compile-step is set. `reduce-overhead` uses CUDA graphs to eliminate per-kernel Python/driver launch overhead -- real candidate for BDH specifically, since its forward pass is many small sequential matmuls in a Python for-loop over n_layer (likely launch-overhead-bound at small batch sizes on consumer GPUs), unlike `default` mode which mainly fuses operators but still issues per-op CUDA calls normally. CUDA graphs require STATIC input shapes across calls (true here: batch/sequence-length are fixed for the whole run) and static memory addresses for the graphed tensors -- a real, disclosed constraint of this mode, not a free upgrade in general, but a good fit for this runner's own fixed-shape training loop specifically.")
    parser.add_argument("--fused-optimizer", action="store_true", help="Use CUDA fused AdamW where available.")
    parser.add_argument("--block-size", type=int, default=16, help="Even contiguous latent columns per routed block.")
    parser.add_argument("--active-fraction", type=float, default=0.5, help="Fraction of blocks selected each batch (0,1].")
    parser.add_argument("--balance-loss-weight", type=float, default=0.0, help="Optional encoder block-load balance auxiliary-loss weight.")
    parser.add_argument("--router-exploration-noise", type=float, default=0.0, help="Gumbel score noise used only while training.")
    args = parser.parse_args()

    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.block_size <= 0 or args.block_size % 2:
        raise ValueError("--block-size must be positive and even because RoPE pairs columns")
    if not 0 < args.active_fraction <= 1:
        raise ValueError("--active-fraction must be in (0, 1]")
    if args.balance_loss_weight < 0:
        raise ValueError("--balance-loss-weight must be non-negative")
    latent_width = args.n_embd * args.mlp_internal_dim_multiplier // args.n_head
    if latent_width % args.block_size:
        raise ValueError(f"--block-size ({args.block_size}) must divide latent width N={latent_width}")
    if args.compile_step:
        raise ValueError("--compile-step is not supported for dynamic BlockBDH routing; do not compare a compiled dense arm against this eager derivative")

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")
    hardware_id = torch.cuda.get_device_name(device) if device.type == "cuda" else ("Apple MPS" if device.type == "mps" else "CPU")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    memory_log = args.run_dir / "block_bdh_memory.jsonl"
    checkpoint = args.run_dir / "block_bdh_checkpoint"
    if not (args.resume and Path(str(checkpoint) + ".pt").exists()):
        memory_log.write_text("", encoding="utf-8")

    sequence_length = args.sequence_length or len(json.loads(args.data.open().readline()))
    total_optimizer_steps = max(1, math.ceil(args.target_tokens / (args.batch_size * sequence_length)))
    effective_batch_tokens = args.batch_size * sequence_length

    torch.manual_seed(args.seed)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    bdh_config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size, dropout=args.dropout,
    )
    model = BDH(bdh_config).to(device=device, dtype=torch_dtype)
    # RoPE's frequency buffer must stay float32 (BDH.attn.forward's own
    # `assert self.freqs.dtype == torch.float32`) -- the blanket .to(dtype=...)
    # above downcasts every buffer/parameter including this one when
    # --dtype is float16/bfloat16, which crashes on the very first forward
    # call. Restore it after the cast rather than skip the cast for the
    # rest of the model.
    model.attn.freqs = model.attn.freqs.to(torch.float32)
    # torch.compile wraps model's __call__ but shares the SAME parameter
    # tensors -- forward_model is used for every training/eval step below,
    # while `model` itself (uncompiled) is what gets checkpointed, so saved
    # weights stay a plain, portable BDH state_dict either way.
    # Dynamic index selection is deliberately eager; compile is rejected above.
    forward_model = model

    config_snapshot = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config_snapshot.update(sequence_length_resolved=sequence_length, effective_batch_tokens=effective_batch_tokens, total_optimizer_steps_estimate=total_optimizer_steps, resolved_device=str(device), backend="torch", architecture="bdh", parameter_count=sum(p.numel() for p in model.parameters()))
    (args.run_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2, sort_keys=True), encoding="utf-8")

    def routed_forward(inputs: torch.Tensor, targets: torch.Tensor | None = None, *, training: bool = False):
        active = compute_active_blocks(model, inputs, args.block_size, args.active_fraction,
            exploration_noise=args.router_exploration_noise if training else 0.0)
        logits, lm_loss = bdh_blocksparse_forward(model, inputs, active, args.block_size, targets=targets)
        return logits, lm_loss, active

    def current_lr(at_step: int) -> float:
        if args.lr_schedule == "constant":
            return args.max_lr
        return lr_at_step(at_step, total_optimizer_steps, args.warmup_steps, args.max_lr, args.lr_min_ratio)

    if args.fused_optimizer:
        if device.type != "cuda":
            raise ValueError("--fused-optimizer requires --device cuda (fused AdamW is a CUDA-only kernel)")
        optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr(0), weight_decay=args.weight_decay, fused=True)
    else:
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
                _logits, loss, _active = routed_forward(x, targets=y, training=False)
                total += float(loss)
                count += 1
            model.train()
            return total / count

        reset_peak_memory(device)
        model.train()
        previous_active_blocks: set[int] | None = None
        while tokens_seen < args.target_tokens:
            tokens = read_batch(train, args.batch_size, sequence_length, device, epoch_counter)
            x, y = shifted_target_batch(tokens)
            optimizer.zero_grad(set_to_none=True)
            _logits, lm_loss, active_blocks = routed_forward(x, targets=y, training=True)
            balance_loss = block_balance_loss(model, x, args.block_size) if args.balance_loss_weight else None
            loss = lm_loss if balance_loss is None else lm_loss + args.balance_loss_weight * balance_loss
            loss.backward()
            grad_norm = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in model.parameters() if p.grad is not None], 2.0))))
            last_lr = current_lr(step)
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            step += 1
            batch_index += 1
            tokens_seen += args.batch_size * sequence_length
            active_block_indices = [int(value) for value in active_blocks.detach().cpu().tolist()]
            active_block_set = set(active_block_indices)
            route_jaccard_previous = None if previous_active_blocks is None else len(active_block_set & previous_active_blocks) / len(active_block_set | previous_active_blocks)
            previous_active_blocks = active_block_set
            item = {"step": step, "tokens_seen": tokens_seen, "loss": float(loss.detach()), "lm_loss": float(lm_loss.detach()), "balance_loss": None if balance_loss is None else float(balance_loss.detach()), "active_block_count": int(active_blocks.numel()), "active_block_indices": active_block_indices, "route_jaccard_previous": route_jaccard_previous, "gradient_norm": grad_norm, "lr": last_lr, "wall_time": time.perf_counter() - started, "epoch_or_data_pass": epoch_counter[0], "peak_memory_bytes": peak_memory_bytes(device)}

            is_new_best = False
            if step % args.validation_interval == 0 or tokens_seen >= args.target_tokens:
                item["validation_loss"] = evaluate_fixed_validation()
                if best_validation_loss is None or item["validation_loss"] < best_validation_loss:
                    best_validation_loss = item["validation_loss"]
                    is_new_best = True
                # Phase 1 metrics (plans/HatchlingZero_Reality_Plan.md):
                # activation sparsity + synaptic-state norms. Only at
                # validation_interval, not every step -- runs a second
                # (read-only, no_grad) forward via bdh_stream_chunk, real
                # but non-trivial extra cost.
                item["activation_state_diagnostics"] = compute_activation_and_state_diagnostics(
                    model, fixed_validation_tokens[: min(8, fixed_validation_tokens.shape[0])]
                )
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
        "backend": "torch", "device": str(device), "hardware_id": hardware_id, "effective_batch_tokens": effective_batch_tokens,
        "compile_step": False, "compile_mode": None, "fused_optimizer": args.fused_optimizer,
        "architecture": "block_bdh_derivative", "exact_bdh": False, "claim_eligible": False, "dtype": args.dtype,
        "block_size": args.block_size, "active_fraction": args.active_fraction, "balance_loss_weight": args.balance_loss_weight, "router_exploration_noise": args.router_exploration_noise,
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
    (args.run_dir / "block_bdh_training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
