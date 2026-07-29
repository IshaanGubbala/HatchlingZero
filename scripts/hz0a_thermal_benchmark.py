"""Thermally-honest hybrid-vs-transformer throughput benchmark.

Cold-start GPU throughput on Apple Silicon (~2.6k tok/s observed earlier
this project) is not sustainable; steady-state after sustained load is
materially lower (~1.4-1.5k tok/s observed). Comparing one architecture's
cold-start number against another's steady-state number silently biases
any "architecture X is faster" claim. This script removes that bias by
alternating which architecture goes first/hot/cold:

    hybrid (cold) -> transformer (hot, inherits hybrid's heat)
    -> cooldown -> transformer (warm-restart) -> hybrid (hot, inherits
    transformer's heat)

Each architecture ends up with one cold-ish slot and one hot slot, and
within every phase the first `--ramp-fraction` of steps is excluded from
the steady-state statistics (kept only for the raw time series) so a
phase's own internal ramp-up doesn't skew its steady-state median.

Reports, per architecture, pooled across both of its phases: steady-state
median/p10/p90 tokens/sec, forward/backward/optimizer ms, and peak
active/cache memory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from reference.hz0a_mlx_model import HZ0AMlxModel


def build_model(architecture: str, dim: int, layers: int, heads: int, hybrid_d_ff: int, transformer_d_ff: int):
    if architecture == "transformer":
        attention = tuple(range(layers))
        d_ff = transformer_d_ff
    else:
        attention = tuple(index for index in (4, 9, 14, 19, 24, 29) if index < layers)
        d_ff = hybrid_d_ff
    model = HZ0AMlxModel(24576, dim, layers, heads, d_ff, attention, native_metal=True)
    mx.eval(model.parameters())
    return model


def capture_power_state() -> dict:
    """Best-effort snapshot of macOS power/thermal-relevant state, so a
    throughput comparison can be checked against "was this run on battery /
    in low-power mode / about to sleep" rather than assumed clean."""
    def run(command: list[str]) -> str:
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception as error:  # best-effort diagnostics only, never fail the benchmark over this
            return f"<unavailable: {error}>"
    return {
        "pmset_active_assertions": run(["pmset", "-g"]),
        "pmset_battery": run(["pmset", "-g", "batt"]),
    }


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[index]


def run_phase(architecture: str, args, label: str) -> dict:
    mx.random.seed(7)
    model = build_model(architecture, args.dim, args.layers, args.heads, args.hybrid_d_ff, args.transformer_d_ff)
    tokens = mx.random.randint(0, 24576, (args.batch_size, args.sequence_length))
    mx.eval(tokens)

    def loss_fn(current, toks):
        states = None
        parts = []
        for start in range(0, args.sequence_length, args.chunk_length):
            logits, states = current(toks[:, start:start + args.chunk_length], states)
            parts.append(logits)
        logits = mx.concatenate(parts, axis=1)
        return mx.mean(nn.losses.cross_entropy(logits[:, :-1], toks[:, 1:]))

    value_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=1e-4)
    mx.reset_peak_memory()

    forward_ms, combined_ms, optimizer_ms, tokens_per_sec = [], [], [], []
    elapsed_at_step = []
    phase_started = time.perf_counter()
    print(f"[{label}] {architecture}: running {args.steps} steps...", flush=True)
    for step in range(args.steps):
        t0 = time.perf_counter()
        loss = loss_fn(model, tokens)
        mx.eval(loss)
        t1 = time.perf_counter()

        loss, grads = value_and_grad(model, tokens)
        mx.eval(loss, grads)
        t2 = time.perf_counter()

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        t3 = time.perf_counter()

        forward_ms.append((t1 - t0) * 1000)
        combined_ms.append((t2 - t1) * 1000)
        optimizer_ms.append((t3 - t2) * 1000)
        tokens_per_sec.append((args.batch_size * args.sequence_length) / (t3 - t0))
        elapsed_at_step.append(time.perf_counter() - phase_started)
        if (step + 1) % max(1, args.steps // 5) == 0:
            print(f"[{label}] {architecture}: step {step + 1}/{args.steps}, last tok/s={tokens_per_sec[-1]:.1f}", flush=True)

    # Ramp exclusion is the LARGER of the step-fraction rule and a minimum
    # wall-clock warmup duration -- a fixed step fraction alone
    # under-excludes when a phase happens to run fast (few seconds isn't
    # long enough for thermal/cache state to actually stabilize).
    steps_within_min_warmup = sum(1 for elapsed in elapsed_at_step if elapsed < args.min_warmup_seconds)
    ramp = max(1, int(args.steps * args.ramp_fraction), min(steps_within_min_warmup, args.steps - 1))
    steady_forward = forward_ms[ramp:]
    steady_combined = combined_ms[ramp:]
    steady_optimizer = optimizer_ms[ramp:]
    steady_tps = tokens_per_sec[ramp:]

    result = {
        "phase_label": label,
        "architecture": architecture,
        "parameter_count": sum(v.size for _, v in tree_flatten(model.parameters())),
        "steps_total": args.steps,
        "steps_excluded_as_ramp": ramp,
        "raw_tokens_per_second": tokens_per_sec,
        "steady_state": {
            "tokens_per_second_median": percentile(steady_tps, 50),
            "tokens_per_second_p10": percentile(steady_tps, 10),
            "tokens_per_second_p90": percentile(steady_tps, 90),
            "forward_ms_median": percentile(steady_forward, 50),
            "backward_ms_median": max(percentile(steady_combined, 50) - percentile(steady_forward, 50), 0.0),
            "optimizer_ms_median": percentile(steady_optimizer, 50),
        },
        "peak_active_memory_gb": mx.get_active_memory() / 1e9,
        "peak_cache_memory_gb": mx.get_cache_memory() / 1e9,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }
    del model, tokens
    mx.clear_cache()
    return result


def pool_architecture(phases: list[dict]) -> dict:
    steady_tps = []
    for phase in phases:
        raw = phase["raw_tokens_per_second"]
        ramp = phase["steps_excluded_as_ramp"]
        steady_tps.extend(raw[ramp:])
    return {
        "phases": [phase["phase_label"] for phase in phases],
        "tokens_per_second_median": percentile(steady_tps, 50),
        "tokens_per_second_p10": percentile(steady_tps, 10),
        "tokens_per_second_p90": percentile(steady_tps, 90),
        "peak_memory_gb_max_across_phases": max(phase["peak_memory_gb"] for phase in phases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--chunk-length", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=31)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--hybrid-d-ff", type=int, default=2304)
    parser.add_argument("--transformer-d-ff", type=int, default=2944)
    parser.add_argument("--steps", type=int, default=150, help="Steps per phase")
    parser.add_argument("--ramp-fraction", type=float, default=0.3, help="Fraction of each phase's steps excluded from steady-state stats")
    parser.add_argument("--cooldown-seconds", type=float, default=90.0)
    parser.add_argument("--min-warmup-seconds", type=float, default=60.0, help="Minimum wall-clock time per phase excluded from steady-state stats, in addition to --ramp-fraction (shorter than the ideal 10-15 minutes for a quick sweep; lengthen for a final, authoritative comparison)")
    parser.add_argument("--output", type=Path, default=Path("outputs/hz0a_thermal_benchmark.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    power_before = capture_power_state()
    phase1 = run_phase("hybrid", args, "phase1_hybrid_cold")
    phase2 = run_phase("transformer", args, "phase2_transformer_hot_after_hybrid")
    print(f"Cooling down {args.cooldown_seconds}s before the warm-restart pair...", flush=True)
    time.sleep(args.cooldown_seconds)
    phase3 = run_phase("transformer", args, "phase3_transformer_warm_restart")
    phase4 = run_phase("hybrid", args, "phase4_hybrid_hot_after_transformer")
    power_after = capture_power_state()

    report = {
        "config": vars(args) | {"output": str(args.output)},
        "phases": [phase1, phase2, phase3, phase4],
        "pooled": {
            "hybrid": pool_architecture([phase1, phase4]),
            "transformer": pool_architecture([phase2, phase3]),
        },
        "power_state_before": power_before,
        "power_state_after": power_after,
        "total_wall_seconds": time.perf_counter() - started,
    }
    print(json.dumps(report["pooled"], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
