#!/usr/bin/env python3
"""School-0 Physics training, plans/Hatchling world.md section 8.2/
Phase 9. "Comparative-magnitude reasoning": a general rule ("a large
object needs more force than a small object") taught once per episode,
then applied to two per-episode objects named only by color -- the
answer is which of TWO entities the rule picks out, not a single
premise's conclusion (`generate_rule_episode`'s task). Real question:
can the model learn to apply an abstract comparative rule to arbitrary
color-bound instances, generalizing across all color pairs and both
question-orderings?
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz_language_model_torch import HZLanguageModel
from hatchling_world.language.tokenizer import NurseryTokenizer, COLORS
from hatchling_world.school.generator import generate_physics_episode

TEST_SEED_OFFSET = 10_000_000


def physics_tensors(tok, ep):
    teach_ids = torch.tensor([tok.encode(ep["teach"])])
    scenario_ids = torch.tensor([tok.encode(ep["scenario"])])
    question_ids = torch.tensor([tok.encode(ep["question"])])
    target = torch.tensor([ep["answer_idx"]])
    return teach_ids, scenario_ids, question_ids, target


def physics_train_step(model, opt, tok, rng):
    ep = generate_physics_episode(rng)
    teach_ids, scenario_ids, question_ids, target = physics_tensors(tok, ep)
    logits = model.physics_forward(teach_ids, scenario_ids, question_ids)
    loss = F.cross_entropy(logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item(), (logits.argmax(-1) == target).float().item()


def physics_eval(model, tok, rng, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_physics_episode(rng)
            teach_ids, scenario_ids, question_ids, target = physics_tensors(tok, ep)
            logits = model.physics_forward(teach_ids, scenario_ids, question_ids)
            correct += int((logits.argmax(-1) == target).item())
    return correct / n_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_read_labels=len(COLORS))
    print(f"[school0-physics] vocab_size={tok.vocab_size} n_params={sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + TEST_SEED_OFFSET)
    recent = []
    for step in range(args.steps):
        loss, acc = physics_train_step(model, opt, tok, train_rng)
        recent.append(acc); recent[:] = recent[-200:]
        if (step + 1) % args.eval_every == 0:
            held_out = physics_eval(model, tok, eval_rng, args.eval_episodes)
            print(f"[school0-physics] step={step+1}/{args.steps} train_acc={sum(recent)/len(recent):.3f} "
                  f"held_out_acc={held_out:.3f} (chance={1.0/len(COLORS):.3f})", flush=True)


if __name__ == "__main__":
    main()
