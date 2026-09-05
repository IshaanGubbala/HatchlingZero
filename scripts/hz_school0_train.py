#!/usr/bin/env python3
"""School-0 training, plans/Hatchling world.md section 8.2, explicit
user request 2026-09-05: "build the first actual School-0 reasoning
curriculum -- not just language tasks anymore: simple arithmetic,
logic, causal rules, then teach -> quiz -> apply." A minimal, real
slice of section 8.3's full 8-step pipeline (Teach->Demonstrate->
Recall->Reason->Apply->Experiment->Correct->Generalize): arithmetic's
"quiz" IS its "apply" (the held-out operand-pair split tests
generalization directly), and the rule task's teach/question turns are
literally Teach->Apply with Reason folded into H's reasoning over S.

Two real, distinct tasks:
  - Arithmetic: "{a} plus {b} equals" -> the sum, with a real held-out
    (a, b) split (ARITH_HELD_OUT_PAIRS) so held-out accuracy measures
    genuine generalization to unseen operand pairs, not memorized sums.
  - Rule/logic: teach a GENERAL conditional ("if an object is {color}
    then it is {size}"), then ask about a specific instance identified
    by the rule's premise -- answerable only by applying the rule
    (never stated directly), a real deduction test.
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
from hatchling_world.school.generator import generate_arithmetic_episode, generate_rule_episode

TEST_SEED_OFFSET = 10_000_000


def arith_train_step(model, opt, tok, rng):
    ep = generate_arithmetic_episode(rng, split="train")
    ids = torch.tensor([tok.encode(ep["instruction"])])
    target = torch.tensor([ep["sum_idx"]])
    logits = model.arithmetic_forward(ids)
    loss = F.cross_entropy(logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item(), (logits.argmax(-1) == target).float().item()


def arith_eval(model, tok, rng, n_episodes, split):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_arithmetic_episode(rng, split=split)
            ids = torch.tensor([tok.encode(ep["instruction"])])
            target = torch.tensor([ep["sum_idx"]])
            logits = model.arithmetic_forward(ids)
            correct += int((logits.argmax(-1) == target).item())
    return correct / n_episodes


def rule_train_step(model, opt, tok, rng):
    ep = generate_rule_episode(rng)
    rule_ids = torch.tensor([tok.encode(ep["rule"])])
    question_ids = torch.tensor([tok.encode(ep["question"])])
    target = torch.tensor([ep["answer_idx"]])
    logits = model.rule_forward(rule_ids, question_ids)
    loss = F.cross_entropy(logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item(), (logits.argmax(-1) == target).float().item()


def rule_eval(model, tok, rng, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_rule_episode(rng)
            rule_ids = torch.tensor([tok.encode(ep["rule"])])
            question_ids = torch.tensor([tok.encode(ep["question"])])
            target = torch.tensor([ep["answer_idx"]])
            logits = model.rule_forward(rule_ids, question_ids)
            correct += int((logits.argmax(-1) == target).item())
    return correct / n_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["arith", "rule", "both"], default="both")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--arith-steps", type=int, default=2500)
    parser.add_argument("--rule-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_arith_labels=len(NUMBERS), n_read_labels=len(SIZES))
    print(f"[school0] vocab_size={tok.vocab_size} n_params={sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.stage in ("arith", "both"):
        train_rng = random.Random(args.seed + 1)
        eval_seen_rng = random.Random(args.seed + 1 + TEST_SEED_OFFSET)
        eval_unseen_rng = random.Random(args.seed + 1 + 2 * TEST_SEED_OFFSET)
        recent = []
        for step in range(args.arith_steps):
            loss, acc = arith_train_step(model, opt, tok, train_rng)
            recent.append(acc); recent[:] = recent[-200:]
            if (step + 1) % args.eval_every == 0:
                seen = arith_eval(model, tok, eval_seen_rng, args.eval_episodes, split="train")
                unseen = arith_eval(model, tok, eval_unseen_rng, args.eval_episodes, split="test")
                print(f"[school0][arith] step={step+1}/{args.arith_steps} "
                      f"train_acc={sum(recent)/len(recent):.3f} held_out_seen_pair_acc={seen:.3f} "
                      f"held_out_UNSEEN_pair_acc={unseen:.3f}", flush=True)

    if args.stage in ("rule", "both"):
        train_rng = random.Random(args.seed + 2)
        eval_rng = random.Random(args.seed + 2 + TEST_SEED_OFFSET)
        recent = []
        for step in range(args.rule_steps):
            loss, acc = rule_train_step(model, opt, tok, train_rng)
            recent.append(acc); recent[:] = recent[-200:]
            if (step + 1) % args.eval_every == 0:
                held_out = rule_eval(model, tok, eval_rng, args.eval_episodes)
                print(f"[school0][rule] step={step+1}/{args.rule_steps} "
                      f"train_acc={sum(recent)/len(recent):.3f} held_out_acc={held_out:.3f} (chance=0.500)", flush=True)


if __name__ == "__main__":
    main()
