#!/usr/bin/env python3
"""Real diagnostic for a genuine, unexplained anomaly: raw BDH's own
forward+backward+optimizer.step() measured 397.6 tok/s (7.7 seconds per
step -- pathologically slow for a 25M-param model on an RTX3060) via
scripts/hz0h_bdh_native_kernel_benchmark.py, but ~6,900-6,990 tok/s
(~0.044 seconds per step) via scripts/hz0h_gpu_native_ablation_benchmark.py,
same config, same GPU, runs minutes apart with the GPU independently
confirmed idle/unthrottled (P8, 35C, no slowdown flags) immediately
before the slow run. Not a thermal/power regime drifting over time --
something structural differs between how the two scripts construct or
run what should be the identical raw BDH.forward() call.

This script reproduces BOTH code paths' raw-BDH construction pattern
side by side in ONE process, polling `nvidia-smi` for SM clock speed and
GPU utilization at ~0.5s intervals DURING each timed loop (not just
before), so the comparison is automatic and exact rather than relying on
manually coordinating a separate background poller against two separate
script invocations.

Path A ("native_script_style"): mirrors
scripts/hz0h_bdh_native_kernel_benchmark.py's structure exactly --
creates a SECOND model (`native`) alongside `raw` before any timing (as
that script does for its own native/triton comparison), and runs one
extra parity forward+backward on both BEFORE the timed raw loop starts,
via a plain oracle-native pair (no Triton needed for this diagnostic --
using the tiled/native backend is enough to reproduce the same "two
models + one pre-timing parity call" structure).

Path B ("ablation_script_style"): mirrors
scripts/hz0h_gpu_native_ablation_benchmark.py's structure -- ONE model,
no pre-timing parity call, timed loop starts immediately after its own
warmup.

If Path A reproduces the slow (~400 tok/s) number and Path B reproduces
the fast (~6900 tok/s) number in this SAME process, that isolates the
real cause to one of: the extra model construction, the pre-timing
parity forward+backward, or some interaction between them -- not GPU
clock state (which this script polls directly to rule in or out).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from reference.hz0h_bdh_native_kernel_attention_torch import bdh_native_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _sync() -> None:
    torch.cuda.synchronize()


class ClockPoller:
    """Polls nvidia-smi for SM clock (MHz) and GPU utilization (%) on a
    background thread at a fixed interval, real subprocess calls, no
    fabricated data."""

    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = interval_seconds
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> dict | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu,power.draw,pstate,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0,
            )
            if out.returncode != 0:
                return None
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            return {
                "t": time.perf_counter(),
                "sm_clock_mhz": parts[0], "gpu_util_pct": parts[1],
                "power_draw_w": parts[2], "pstate": parts[3], "temp_c": parts[4],
            }
        except Exception as exc:  # real, disclosed: nvidia-smi may not be on PATH or may error
            return {"t": time.perf_counter(), "error": str(exc)}

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._poll_once()
            if sample is not None:
                self.samples.append(sample)
            time.sleep(self.interval_seconds)

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return self.samples


def run_native_script_style(config: BDHConfig, idx, targets, device, dtype, warmup: int, steps: int, lr: float, poller: ClockPoller) -> dict:
    """Reproduces hz0h_bdh_native_kernel_benchmark.py's structure: raw +
    native both constructed up front, one parity forward+backward on
    both BEFORE the timed raw loop."""
    raw = BDH(config).to(device=device, dtype=dtype)
    native = BDH(config).to(device=device, dtype=dtype)
    native.load_state_dict(raw.state_dict())
    raw.attn.freqs = raw.attn.freqs.to(torch.float32)
    native.attn.freqs = native.attn.freqs.to(torch.float32)

    raw.zero_grad(set_to_none=True)
    native.zero_grad(set_to_none=True)
    raw_logits, raw_loss = raw(idx, targets)
    native_logits, native_loss = bdh_native_forward(native, idx, targets)
    raw_loss.backward()
    native_loss.backward()
    _sync()

    optimizer = torch.optim.AdamW(raw.parameters(), lr=lr, weight_decay=0.1, fused=True)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        _, loss = raw(idx, targets)
        loss.backward()
        optimizer.step()
    _sync()
    torch.cuda.reset_peak_memory_stats()
    poller.start()
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = raw(idx, targets)
        loss.backward()
        optimizer.step()
    _sync()
    elapsed = time.perf_counter() - started
    samples = poller.stop()
    tokens = idx.numel() * steps
    return {
        "seconds": elapsed, "tokens_per_second": tokens / elapsed,
        "seconds_per_step": elapsed / steps,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "clock_samples_during_timed_loop": samples,
    }


def run_ablation_script_style(config: BDHConfig, idx, targets, device, dtype, warmup: int, steps: int, lr: float, poller: ClockPoller) -> dict:
    """Reproduces hz0h_gpu_native_ablation_benchmark.py's raw stage: one
    model, no pre-timing parity call."""
    model = BDH(config).to(device=device, dtype=dtype)
    model.attn.freqs = model.attn.freqs.to(torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=True)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        loss.backward()
        optimizer.step()
    _sync()
    torch.cuda.reset_peak_memory_stats()
    poller.start()
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        loss.backward()
        optimizer.step()
    _sync()
    elapsed = time.perf_counter() - started
    samples = poller.stop()
    tokens = idx.numel() * steps
    return {
        "seconds": elapsed, "tokens_per_second": tokens / elapsed,
        "seconds_per_step": elapsed / steps,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "clock_samples_during_timed_loop": samples,
    }


def summarize_clocks(samples: list[dict]) -> dict:
    clocks = [int(s["sm_clock_mhz"]) for s in samples if "sm_clock_mhz" in s and s["sm_clock_mhz"].isdigit()]
    utils = [int(s["gpu_util_pct"]) for s in samples if "gpu_util_pct" in s and s["gpu_util_pct"].isdigit()]
    if not clocks:
        return {"n_samples": len(samples), "note": "no valid nvidia-smi samples parsed"}
    return {
        "n_samples": len(samples),
        "sm_clock_mhz_min": min(clocks), "sm_clock_mhz_max": max(clocks),
        "sm_clock_mhz_mean": sum(clocks) / len(clocks),
        "gpu_util_pct_min": min(utils) if utils else None,
        "gpu_util_pct_max": max(utils) if utils else None,
        "gpu_util_pct_mean": sum(utils) / len(utils) if utils else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this diagnostic requires real CUDA hardware")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    idx = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device)
    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size, dropout=0.0,
    )
    poller = ClockPoller(interval_seconds=args.poll_interval)

    print("Running native_script_style (raw+native pre-built, parity pre-pass)...", file=sys.stderr)
    native_style_result = run_native_script_style(config, idx, targets, device, dtype, args.warmup, args.steps, args.learning_rate, poller)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2.0)  # real settle gap between the two comparisons

    print("Running ablation_script_style (single model, no parity pre-pass)...", file=sys.stderr)
    ablation_style_result = run_ablation_script_style(config, idx, targets, device, dtype, args.warmup, args.steps, args.learning_rate, poller)

    results = {
        "hardware_id": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size, "sequence_length": args.sequence_length,
        "warmup_steps": args.warmup, "timed_steps": args.steps,
        "native_script_style": {
            "seconds": native_style_result["seconds"],
            "tokens_per_second": native_style_result["tokens_per_second"],
            "seconds_per_step": native_style_result["seconds_per_step"],
            "peak_memory_bytes": native_style_result["peak_memory_bytes"],
            "clock_summary": summarize_clocks(native_style_result["clock_samples_during_timed_loop"]),
        },
        "ablation_script_style": {
            "seconds": ablation_style_result["seconds"],
            "tokens_per_second": ablation_style_result["tokens_per_second"],
            "seconds_per_step": ablation_style_result["seconds_per_step"],
            "peak_memory_bytes": ablation_style_result["peak_memory_bytes"],
            "clock_summary": summarize_clocks(ablation_style_result["clock_samples_during_timed_loop"]),
        },
        "raw_clock_samples": {
            "native_script_style": native_style_result["clock_samples_during_timed_loop"],
            "ablation_script_style": ablation_style_result["clock_samples_during_timed_loop"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "raw_clock_samples"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
