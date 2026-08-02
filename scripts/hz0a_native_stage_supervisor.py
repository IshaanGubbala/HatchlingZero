"""Sequentially resume the native Stage 1 runner to bound graph lifetime.

The runner can terminate after several hundred chunks on long Apple-GPU jobs
without corrupting its last checkpoint. This supervisor never overlaps workers:
it starts one runner, reads its checkpoint, and resumes the same run directory
until the requested token budget is complete.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def checkpoint_tokens(run_dir: Path) -> int:
    state = run_dir / "native_metal_checkpoint" / "state.json"
    if not state.exists():
        return 0
    return int(json.loads(state.read_text())["tokens_seen"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=10_000_000)
    parser.add_argument("--child-tokens", type=int, default=128_000, help="Maximum new tokens per child process; keeps GPU graph lifetime bounded")
    parser.add_argument("--max-restarts", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--chunk-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=24576)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=31)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=2304)
    parser.add_argument("--mixer", choices=("gdn2", "gdn2_fix"), default="gdn2_fix")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--compile-step", action="store_true")
    args = parser.parse_args()
    if args.child_tokens <= 0:
        raise ValueError("--child-tokens must be positive")
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    previous = checkpoint_tokens(run_dir)
    for restart in range(args.max_restarts + 1):
        child_target = min(args.target_tokens, previous + args.child_tokens)
        command = [
            sys.executable, "scripts/hz0a_native_stage_runner.py",
            "--data", str(args.data), "--validation-data", str(args.validation_data),
            "--run-dir", str(run_dir), "--target-tokens", str(args.target_tokens),
            "--stop-tokens", str(child_target),
            "--batch-size", str(args.batch_size), "--checkpoint-interval", str(args.checkpoint_interval),
            "--validation-interval", str(args.validation_interval), "--chunk-length", str(args.chunk_length),
            "--truncate-backward", "--vocab-size", str(args.vocab_size), "--dim", str(args.dim),
            "--layers", str(args.layers), "--heads", str(args.heads), "--d-ff", str(args.d_ff),
            "--dtype", "float32", "--mixer", args.mixer, "--reset-attention-state", "--seed", str(args.seed),
        ]
        if restart or previous:
            command.append("--resume")
        if args.activation_checkpoint:
            command.append("--activation-checkpoint")
        if args.compile_step:
            command.append("--compile-step")
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            start_new_session=True,
        )
        try:
            returncode = process.wait()
        except BaseException:
            # Keep an interrupted supervisor from orphaning a GPU runner.
            try:
                os.killpg(process.pid, 15)
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, 9)
                except OSError:
                    pass
            raise
        current = checkpoint_tokens(run_dir)
        history.append({"restart": restart, "exit_code": returncode, "checkpoint_tokens": current, "seconds": time.perf_counter() - started})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if current >= args.target_tokens:
            break
        if current <= previous:
            raise RuntimeError(f"runner made no checkpoint progress: previous={previous}, current={current}")
        previous = current
    else:
        raise RuntimeError(f"restart budget exhausted at {previous} tokens")
    report = run_dir / "native_metal.json"
    print(json.dumps({"target_tokens": args.target_tokens, "checkpoint_tokens": checkpoint_tokens(run_dir), "budget_complete": checkpoint_tokens(run_dir) >= args.target_tokens, "restarts": len(history), "history": history, "report_exists": report.exists()}, indent=2))


if __name__ == "__main__":
    main()
