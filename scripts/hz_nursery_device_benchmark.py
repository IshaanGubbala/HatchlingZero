#!/usr/bin/env python3
"""Phase 6 (systems/speed), explicit user priority 2026-09-05: CPU
baseline vs MPS reference on the Language Nursery's ACTUAL training
loop, not an isolated microbenchmark. Extends this project's earlier
real finding (FSM harness AND room-navigation BC training: MPS was
SLOWER than CPU for that tiny sequential workload, because CPU-side
Python data generation dominates cost and per-step host-to-device
transfer overhead erases any GPU compute advantage at this scale) to
the Nursery's own per-token/per-episode training loops, which have
never been benchmarked this way -- they're architecturally similar
(small batch, many sequential Python-level steps) but a genuinely
different codepath, so the old finding is not assumed to carry over
unchanged, it's re-measured directly.

Runs REAL L1 grounding training steps (the same generator + forward/
backward/optimizer-step hz_nursery_train.py's l1_train_step performs,
reimplemented here standalone so tensors can be explicitly placed on
each device without touching that production script) on CPU and, if
available, MPS, and reports real wall-clock time and steps/sec for
each -- no isolated op-level microbenchmark, no assumption.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz_language_model_torch import HZLanguageModel
from hatchling_world.language.tokenizer import NurseryTokenizer, NOUNS, COLORS, SIZES, POSITIONS
from hatchling_world.language.nursery_generator import generate_l1_grounding_episode


def run_benchmark(device: str, tok: NurseryTokenizer, args) -> dict:
    torch.manual_seed(0)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = random.Random(0)

    # Warmup (first-call overhead, especially real on MPS: kernel
    # compilation/allocator warmup -- excluded from the timed region so
    # this measures steady-state throughput, not one-time setup cost).
    for _ in range(args.warmup_steps):
        ep = generate_l1_grounding_episode(rng, n_objects=args.n_objects)
        instr_ids = torch.tensor([tok.encode(ep["instruction"])], device=device)
        type_idx = torch.tensor([[NOUNS.index(o["type"]) for o in ep["objects"]]], device=device)
        color_idx = torch.tensor([[COLORS.index(o["color"]) for o in ep["objects"]]], device=device)
        size_idx = torch.tensor([[SIZES.index(o["size"]) for o in ep["objects"]]], device=device)
        pos_idx = torch.tensor([[POSITIONS.index(o["position"]) for o in ep["objects"]]], device=device)
        target = torch.tensor([ep["target_idx"]], device=device)
        logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(args.timed_steps):
        ep = generate_l1_grounding_episode(rng, n_objects=args.n_objects)
        instr_ids = torch.tensor([tok.encode(ep["instruction"])], device=device)
        type_idx = torch.tensor([[NOUNS.index(o["type"]) for o in ep["objects"]]], device=device)
        color_idx = torch.tensor([[COLORS.index(o["color"]) for o in ep["objects"]]], device=device)
        size_idx = torch.tensor([[SIZES.index(o["size"]) for o in ep["objects"]]], device=device)
        pos_idx = torch.tensor([[POSITIONS.index(o["position"]) for o in ep["objects"]]], device=device)
        target = torch.tensor([ep["target_idx"]], device=device)
        logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    return {"device": device, "timed_steps": args.timed_steps, "elapsed_s": elapsed,
            "steps_per_sec": args.timed_steps / elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--n-objects", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--timed-steps", type=int, default=2000)
    args = parser.parse_args()

    tok = NurseryTokenizer()
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    else:
        print("[bench] MPS not available on this machine -- CPU only", flush=True)

    results = {}
    for device in devices:
        print(f"[bench] running {args.timed_steps} real L1 training steps on {device} "
              f"({args.warmup_steps} warmup steps excluded)...", flush=True)
        r = run_benchmark(device, tok, args)
        results[device] = r
        print(f"[bench] {device}: {r['elapsed_s']:.2f}s total, {r['steps_per_sec']:.1f} steps/sec", flush=True)

    print("\n[bench] === SUMMARY ===")
    for device, r in results.items():
        print(f"{device:>6}: {r['elapsed_s']:>8.2f}s  ({r['steps_per_sec']:.1f} steps/sec)")
    if "mps" in results:
        ratio = results["cpu"]["elapsed_s"] / results["mps"]["elapsed_s"]
        verdict = "MPS faster" if ratio > 1 else "CPU faster"
        print(f"\n[bench] {verdict} -- CPU took {ratio:.2f}x {'more' if ratio > 1 else 'less'} time than MPS "
              f"for this workload.")


if __name__ == "__main__":
    main()
