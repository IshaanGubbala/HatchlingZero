#!/usr/bin/env python3
"""Real training + R-scaling gate evaluation for HZCQ0 on the
reassignment task (reference/hz0h_cq0_tasks_torch.py).

This is CQ-0's own decisive gate
(`plans/Deep Reserach Plan.md`'s "CQ-0: the concrete first build"
section): train with R sampled in-path from {1,2,4,8}, then evaluate
the SAME trained checkpoint at each R and check whether accuracy grows
with R -- and specifically grows MORE as task dependency depth
(num_overwrites) increases. That is the real signature this script's
own final report is checking for, not just "did the loss go down."

Real, disclosed scope: this trains ONE (model, difficulty) pair per
run. To build the plan doc's own `A(R, d)` plot, run this multiple
times at different `--num-overwrites` values and compare the resulting
`accuracy_by_r` tables across runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_cq0_torch import HZCQ0, HZCQ0Config
from reference.hz0h_cq0_tasks_torch import ReassignmentTaskConfig, generate_reassignment_batch


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def run_eval(model: HZCQ0, task_config: ReassignmentTaskConfig, r_iterations: int, num_batches: int, batch_size: int, device: torch.device, gen: torch.Generator) -> float:
    """Real accuracy at a fixed R, averaged over num_batches fresh
    (unseen during this call) real task instances."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(num_batches):
            demo, query, targets = generate_reassignment_batch(batch_size, task_config, gen, device=device)
            logits, _ = model(demo, query, r_iterations=r_iterations)
            predictions = logits[:, 0, :].argmax(dim=-1)  # slot 0 is the answer slot (M=1)
            correct += int((predictions == targets).sum())
            total += batch_size
    model.train()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--m-slots", type=int, default=1, help="1 for this single-value-retrieval task; the answer is always read from slot 0.")
    parser.add_argument("--num-keys", type=int, default=8)
    parser.add_argument("--num-overwrites", type=int, default=3, help="the real dependency-depth axis for the R-scaling gate.")
    parser.add_argument("--num-distractor-keys", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--r-choices", type=str, default="1,2,4,8", help="comma-separated R values sampled in-path during training.")
    parser.add_argument("--eval-r-choices", type=str, default="1,2,4,8", help="comma-separated R values to evaluate the SAME trained checkpoint at, for the final gate report.")
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    task_config = ReassignmentTaskConfig(
        num_keys=args.num_keys, num_overwrites=args.num_overwrites, num_distractor_keys=args.num_distractor_keys,
    )
    model_config = HZCQ0Config(
        n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size, m_slots=args.m_slots, dropout=0.0,
    )
    model = HZCQ0(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    r_choices = [int(r) for r in args.r_choices.split(",")]
    eval_r_choices = [int(r) for r in args.eval_r_choices.split(",")]

    print(f"Training HZCQ0 ({sum(p.numel() for p in model.parameters()):,} params) on reassignment "
          f"(num_overwrites={args.num_overwrites}, num_distractor_keys={args.num_distractor_keys}) "
          f"on {device}, R sampled from {r_choices}", file=sys.stderr)

    started = time.perf_counter()
    losses = []
    for step in range(1, args.train_steps + 1):
        r = r_choices[int(torch.randint(0, len(r_choices), (1,), generator=gen).item())]
        demo, query, targets = generate_reassignment_batch(args.batch_size, task_config, gen, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(demo, query, r_iterations=r, targets=targets.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

        if step % args.log_interval == 0:
            recent = sum(losses[-args.log_interval:]) / len(losses[-args.log_interval:])
            print(f"step {step}/{args.train_steps} mean_loss(last {args.log_interval})={recent:.4f}", file=sys.stderr)

    training_seconds = time.perf_counter() - started

    print("Evaluating the SAME trained checkpoint at each R (the real CQ-0 gate check)...", file=sys.stderr)
    accuracy_by_r = {}
    for r in eval_r_choices:
        acc = run_eval(model, task_config, r, args.eval_batches, args.batch_size, device, gen)
        accuracy_by_r[r] = acc
        print(f"  R={r}: accuracy={acc:.4f}", file=sys.stderr)

    r_sorted = sorted(accuracy_by_r)
    monotonic_nondecreasing = all(accuracy_by_r[r_sorted[i]] <= accuracy_by_r[r_sorted[i + 1]] + 1e-9 for i in range(len(r_sorted) - 1))
    accuracy_gain_low_to_high = accuracy_by_r[r_sorted[-1]] - accuracy_by_r[r_sorted[0]]

    report = {
        "task": "reassignment",
        "task_config": vars(task_config),
        "model_config": vars(model_config),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "train_steps": args.train_steps,
        "r_choices_trained_on": r_choices,
        "training_seconds": training_seconds,
        "final_train_loss_mean_last_100": sum(losses[-100:]) / len(losses[-100:]),
        "accuracy_by_r": accuracy_by_r,
        "accuracy_monotonic_nondecreasing_in_r": monotonic_nondecreasing,
        "accuracy_gain_lowest_to_highest_r": accuracy_gain_low_to_high,
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
