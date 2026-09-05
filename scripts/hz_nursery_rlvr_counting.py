#!/usr/bin/env python3
"""Phase 7 (RL), plans/Hatchling world.md's first real RLVR/GRPO
experiment. Real, verifiable reward loop (checklist item 1) + group-
relative trajectory optimization (checklist item 3), on-policy
(checklist item 2 -- fresh episodes every step, no replay buffer;
replay/off-policy stays deferred per the plan's own ordering,
checklist item 4).

Testbed: L4 counting-verification (`verify_count_forward`), chosen
because its reward is GENUINELY verifiable -- a single binary correct/
incorrect signal computed directly from the generator's own ground
truth, no learned reward model, no human judgment. It also already has
a real, established supervised baseline from this session (~60-72%
held-out accuracy via BCE loss on the same task), so RLVR training can
be compared directly against a known number, not assumed to work.

Method: the model's single logit defines a Bernoulli policy
pi(a) = sigmoid(logit) over {predict TRUE, predict FALSE}. For each
training step, sample a GROUP of K actions from that SAME state (same
episode), score each against the real verifiable reward (+1 correct,
-1 incorrect), and use the GROUP MEAN reward as a variance-reducing
baseline (advantage = reward - group_mean) -- literally GRPO's own
normalization trick, applied here at the smallest possible scale (one
episode, K samples) rather than skipped. Policy-gradient loss:
-mean_k(log pi(a_k) * advantage_k).

Real question: does a pure RL signal, with no supervised gradient at
all, reach anywhere near the same ceiling as the established BCE
baseline on the exact same task? If RLVR also plateaus in the same
60-72% band, that's a real confirmation the ceiling is a genuine model/
task-capacity limit, not an artifact of the supervised loss choice
specifically.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hz_nursery_train as nt  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.tokenizer import NurseryTokenizer  # noqa: E402
from hatchling_world.language.nursery_generator import generate_l4_counting_episode  # noqa: E402

# Real, established supervised (BCE) baseline for this exact task, this
# session (plans/Hatchling world.md's L4-counting writeup) -- printed
# alongside the RLVR result for a direct, honest comparison.
SUPERVISED_BASELINE_ACC = 0.675


def rlvr_train_step(model, opt, tok, rng, n_objects, group_size):
    ep = generate_l4_counting_episode(rng, n_objects=n_objects)
    instr_ids, type_idx, color_idx, size_idx, pos_idx, label = nt.l4_counting_tensors(tok, ep)
    logit = model.verify_count_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)  # (1,)
    prob_true = torch.sigmoid(logit)  # Bernoulli(prob_true) over "predict TRUE"

    # Sample a GROUP of K actions from this SAME state -- real GRPO-style
    # group, not a single-sample REINFORCE estimate.
    dist = torch.distributions.Bernoulli(probs=prob_true.expand(group_size))
    actions = dist.sample()  # (K,) in {0.,1.}
    log_probs = dist.log_prob(actions)  # (K,)

    correct = (actions == label.expand(group_size)).float()
    rewards = correct * 2.0 - 1.0  # +1 correct, -1 incorrect -- the real verifiable signal
    baseline = rewards.mean()  # group-relative baseline (GRPO's own normalization)
    advantages = rewards - baseline

    loss = -(log_probs * advantages.detach()).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    greedy_correct = int(((logit > 0).float() == label).item())
    return loss.item(), greedy_correct, rewards.mean().item()


def rlvr_eval(model, tok, rng, n_objects, n_episodes):
    """Eval uses the GREEDY action (argmax/threshold), standard
    practice -- sampling is a training-time exploration device, not
    how a deployed policy should act."""
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l4_counting_episode(rng, n_objects=n_objects)
            instr_ids, type_idx, color_idx, size_idx, pos_idx, label = nt.l4_counting_tensors(tok, ep)
            logit = model.verify_count_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            correct += int(((logit > 0).float() == label).item())
    return correct / n_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--n-objects", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_rlvr_counting.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + nt.TEST_SEED_OFFSET)

    history = []
    recent_greedy, recent_reward = [], []
    for step in range(args.steps):
        loss, greedy_correct, mean_reward = rlvr_train_step(model, opt, tok, train_rng, args.n_objects, args.group_size)
        recent_greedy.append(greedy_correct); recent_greedy[:] = recent_greedy[-200:]
        recent_reward.append(mean_reward); recent_reward[:] = recent_reward[-200:]

        if (step + 1) % args.eval_every == 0:
            held_out = rlvr_eval(model, tok, eval_rng, args.n_objects, args.eval_episodes)
            print(f"[rlvr-counting] step={step+1}/{args.steps} "
                  f"train_greedy_acc={sum(recent_greedy)/len(recent_greedy):.3f} "
                  f"mean_group_reward={sum(recent_reward)/len(recent_reward):+.3f} "
                  f"held_out_acc={held_out:.3f}", flush=True)
            history.append({"step": step + 1, "held_out_acc": held_out})

    final_acc = rlvr_eval(model, tok, eval_rng, args.n_objects, args.eval_episodes * 2)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump({"history": history, "final_acc": final_acc,
                    "supervised_baseline_acc": SUPERVISED_BASELINE_ACC}, f, indent=2)
    print(f"\n[rlvr-counting] wrote {args.results_file}", flush=True)

    print("\n[rlvr-counting] === SUMMARY ===")
    print(f"RLVR (group-relative policy gradient) final held_out_acc: {final_acc:.3f}")
    print(f"Supervised (BCE) baseline, this session, same task:       {SUPERVISED_BASELINE_ACC:.3f}")
    print(f"Chance:                                                    0.500")


if __name__ == "__main__":
    main()
