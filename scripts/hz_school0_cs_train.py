#!/usr/bin/env python3
"""School-0 Computer Science training, plans/Hatchling world.md
section 8.2/Phase 9. "Program execution": two variable assignments
("x is {a}", "y is {b}") must be tracked as a real symbol table (TWO
simultaneous facts -- directly at the edge this session's own L5
memory-stress diagnostic found S struggling with) before their values
can be substituted into "what is x plus y" and summed. Genuinely
different skill from raw arithmetic (generate_arithmetic_episode
states both operands directly in one instruction; this requires
retrieval first).
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
from hatchling_world.language.tokenizer import NurseryTokenizer, NUMBERS, SIZES
from hatchling_world.school.generator import generate_cs_program_episode

TEST_SEED_OFFSET = 10_000_000


def cs_train_step(model, opt, tok, rng):
    ep = generate_cs_program_episode(rng)
    statement_ids_list = [torch.tensor([tok.encode(s)]) for s in ep["program"]]
    question_ids = torch.tensor([tok.encode(ep["question"])])
    target = torch.tensor([ep["sum_idx"]])
    logits = model.cs_program_forward(statement_ids_list, question_ids)
    loss = F.cross_entropy(logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item(), (logits.argmax(-1) == target).float().item()


def cs_eval(model, tok, rng, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_cs_program_episode(rng)
            statement_ids_list = [torch.tensor([tok.encode(s)]) for s in ep["program"]]
            question_ids = torch.tensor([tok.encode(ep["question"])])
            target = torch.tensor([ep["sum_idx"]])
            logits = model.cs_program_forward(statement_ids_list, question_ids)
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
                             n_arith_labels=len(NUMBERS), n_read_labels=len(SIZES))
    print(f"[school0-cs] vocab_size={tok.vocab_size} n_params={sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + TEST_SEED_OFFSET)
    recent = []
    for step in range(args.steps):
        loss, acc = cs_train_step(model, opt, tok, train_rng)
        recent.append(acc); recent[:] = recent[-200:]
        if (step + 1) % args.eval_every == 0:
            held_out = cs_eval(model, tok, eval_rng, args.eval_episodes)
            print(f"[school0-cs] step={step+1}/{args.steps} train_acc={sum(recent)/len(recent):.3f} "
                  f"held_out_acc={held_out:.3f} (chance={1.0/len(NUMBERS):.3f})", flush=True)


if __name__ == "__main__":
    main()
