#!/usr/bin/env python3
"""Trains a real learned jump operator J2 (`JumpOperator`, jump_size=2)
by distillation against a frozen BDH's own real 2-step trajectories,
then tests the decision-relevant question Part 5's closed-form
diagnostic could not: does substituting J2 for real recurrent
iterations preserve actual validation loss, not just raw state
prediction accuracy.

Real motivation: Part 5's linearizability diagnostic
(`docs/restart/hz0h_inherited_choices_audit_results.md`) found a
CLOSED-FORM linear operator already composes almost exactly at k=2
(gap <1% vs a directly-fit 2-step operator). This script asks whether a
small TRAINED (nonlinear, residual) operator does better, and -- the
real test -- whether it's good enough to actually replace real
recurrent compute without hurting downstream quality.

Method:
1. Train a small real BDH (reuses `train_small_model` from
   `scripts/hz0h_bdh_linearizability_diagnostic.py` so the recipe can't
   drift), CPU by default.
2. Train `JumpOperator` via distillation: for each step, capture the
   frozen model's REAL trajectory (`bdh_forward_with_trajectory`,
   no_grad -- this is the teacher), pick a random even starting depth r
   in {0,2,4,6}, and train `jump(x_r) ~= x_{r+2}` (state MSE) plus a
   logits-KL term (does the jump's prediction, decoded through the
   SAME shared lm_head, preserve the real next-state's downstream
   distribution, not just match its raw magnitude).
3. Evaluate on held-out validation: compare REAL validation loss (not
   just state-prediction error) across arms that reach the same
   depth-equivalent (8) via different real/jump splits:
   - `real_depth8`: the true oracle-equivalent (ground truth ceiling).
   - `all_jumps`: 0 real iterations, 4 jumps (pure J2, most aggressive).
   - `hybrid_2real_6jump`: 2 real iterations (let it reach Part 5's
     "settled" regime first) + 3 jumps.
   - `hybrid_4real_4jump`: 4 real + 2 jumps (a more conservative point).
   Also measures real wall-clock forward-pass time per arm -- the
   actual efficiency claim, since a jump this small should be far
   cheaper than a real BDH iteration's three wide GEMMs.

Real, disclosed limits: single model, CPU, small scale -- a prototype
answering "does this look viable at all," not a 3-seed production
claim.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_jump_operator_torch import JumpOperator, jump_bdh_forward
from reference.hz0h_bdh_trajectory_torch import bdh_forward_with_trajectory
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from scripts.hz0h_bdh_linearizability_diagnostic import train_small_model
from scripts.hz0h_factorized_curriculum_full_comparison import read_batch


def train_jump_operator(model, jump, args, device) -> dict:
    optimizer = torch.optim.AdamW(jump.parameters(), lr=args.jump_learning_rate)
    epochs = [0]
    starting_depths = [r for r in range(0, args.n_layer - 1, 2)]
    history = []
    started = time.perf_counter()
    with args.data.open() as handle:
        for step in range(args.jump_steps):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx = data[:, :-1].contiguous()
            with torch.no_grad():
                _, _, x_states, _ = bdh_forward_with_trajectory(model, idx, args.n_layer)
            r = random.choice(starting_depths)
            x_r = x_states[r].detach()
            x_target = x_states[r + 2].detach()

            optimizer.zero_grad(set_to_none=True)
            predicted = jump(x_r)
            state_loss = F.mse_loss(predicted, x_target)

            B, _, T, D = predicted.shape
            with torch.no_grad():
                target_logits = x_target.view(B, T, D) @ model.lm_head
            predicted_logits = predicted.view(B, T, D) @ model.lm_head
            logits_loss = F.kl_div(
                F.log_softmax(predicted_logits, dim=-1),
                F.softmax(target_logits, dim=-1),
                reduction="batchmean",
            )
            loss = state_loss + args.logits_loss_weight * logits_loss
            loss.backward()
            optimizer.step()

            if (step + 1) % max(1, args.jump_steps // 10) == 0:
                history.append({
                    "step": step + 1, "starting_depth": r,
                    "state_loss": float(state_loss), "logits_loss": float(logits_loss),
                })
                print(f"[jump train] step={step+1} r={r} state_loss={float(state_loss):.4f} "
                      f"logits_loss={float(logits_loss):.4f}", flush=True)
    return {"training_seconds": time.perf_counter() - started, "history": history}


def evaluate_arm(model, jump, args, device, real_prefix: int, num_jumps: int) -> dict:
    epochs = [0]
    losses = []
    started = time.perf_counter()
    with args.validation_data.open() as handle, torch.no_grad():
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            if jump is None:
                _, loss = bdh_variable_depth_forward(model, idx, real_prefix, target)
            else:
                _, loss = jump_bdh_forward(model, jump, idx, real_prefix, num_jumps, target)
            losses.append(float(loss))
    elapsed = time.perf_counter() - started
    return {
        "validation_loss": sum(losses) / len(losses),
        "real_prefix_iterations": real_prefix,
        "num_jumps": num_jumps,
        "depth_equivalent": real_prefix + num_jumps * 2,
        "eval_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-tokens", type=int, default=1_500_000, help="BDH pretraining budget.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--jump-steps", type=int, default=500)
    parser.add_argument("--jump-learning-rate", type=float, default=1e-3)
    parser.add_argument("--jump-hidden-mult", type=int, default=4)
    parser.add_argument("--logits-loss-weight", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(args.device)
    random.seed(args.seed)

    print("[stage 1/3] training BDH teacher ...", flush=True)
    model = train_small_model(args, device)

    print("[stage 2/3] training JumpOperator via distillation ...", flush=True)
    jump = JumpOperator(d_model=args.n_embd, hidden_mult=args.jump_hidden_mult, jump_size=2).to(device)
    jump_train_report = train_jump_operator(model, jump, args, device)

    print("[stage 3/3] evaluating arms ...", flush=True)
    arms = {
        "real_depth8": evaluate_arm(model, None, args, device, real_prefix=args.n_layer, num_jumps=0),
        "all_jumps": evaluate_arm(model, jump, args, device, real_prefix=0, num_jumps=args.n_layer // 2),
        "hybrid_2real_6jump": evaluate_arm(model, jump, args, device, real_prefix=2, num_jumps=3),
        "hybrid_4real_4jump": evaluate_arm(model, jump, args, device, real_prefix=4, num_jumps=2),
    }
    baseline = arms["real_depth8"]["validation_loss"]
    for name, arm in arms.items():
        arm["validation_loss_minus_real_depth8"] = arm["validation_loss"] - baseline
        print(f"[{name}] loss={arm['validation_loss']:.4f} "
              f"delta={arm['validation_loss_minus_real_depth8']:+.4f} "
              f"depth_eq={arm['depth_equivalent']} eval_seconds={arm['eval_seconds']:.2f}", flush=True)

    report = {
        "device": str(device),
        "single_model_prototype_not_a_3seed_claim": (
            "One trained BDH + one trained jump operator. Structural/prototype result, "
            "not a statistical claim -- answers 'does this look viable at all.'"
        ),
        "shape": {
            "n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head, "mult": args.mult,
            "jump_hidden_mult": args.jump_hidden_mult, "jump_steps": args.jump_steps,
        },
        "jump_training": jump_train_report,
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
