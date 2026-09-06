#!/usr/bin/env python3
"""Real, decisive ablation on the Physics comparative-magnitude task's
real plateau (plans/Hatchling world.md Phase 9: 2 seeds x 2500 steps,
held-out acc stuck at ~0.45-0.56, right at the "guess one of the two
named colors" chance floor, not the 4-way floor). User's own proposed
test: does replacing the per-episode-varying COLOR identity with FIXED
symbol tokens (`x`/`y` -- literally CS program-execution's own
identity tokens, which reached 97-100%) make the task solvable? If
so: real evidence the reasoning rule itself was never the bottleneck --
dynamic entity binding/coreference across sentences was. `physics_forward`
is reused UNCHANGED (it is architecture-agnostic to vocabulary); only
the generator and the output label space (2 vs 4) differ.
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
from hatchling_world.language.tokenizer import NurseryTokenizer
from hatchling_world.school.generator import generate_physics_fixed_identity_episode, PHYSICS_IDENTITY_LABELS

TEST_SEED_OFFSET = 10_000_000


def physics_identity_tensors(tok, ep):
    teach_ids = torch.tensor([tok.encode(ep["teach"])])
    scenario_ids = torch.tensor([tok.encode(ep["scenario"])])
    question_ids = torch.tensor([tok.encode(ep["question"])])
    target = torch.tensor([ep["answer_idx"]])
    return teach_ids, scenario_ids, question_ids, target


def train_step(model, opt, tok, rng):
    ep = generate_physics_fixed_identity_episode(rng)
    teach_ids, scenario_ids, question_ids, target = physics_identity_tensors(tok, ep)
    logits = model.physics_forward(teach_ids, scenario_ids, question_ids)
    loss = F.cross_entropy(logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item(), (logits.argmax(-1) == target).float().item()


def eval_acc(model, tok, rng, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_physics_fixed_identity_episode(rng)
            teach_ids, scenario_ids, question_ids, target = physics_identity_tensors(tok, ep)
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
                             n_read_labels=len(PHYSICS_IDENTITY_LABELS))
    print(f"[physics-identity-ablation] vocab_size={tok.vocab_size} "
          f"n_params={sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + TEST_SEED_OFFSET)
    recent = []
    for step in range(args.steps):
        loss, acc = train_step(model, opt, tok, train_rng)
        recent.append(acc); recent[:] = recent[-200:]
        if (step + 1) % args.eval_every == 0:
            held_out = eval_acc(model, tok, eval_rng, args.eval_episodes)
            print(f"[physics-identity-ablation] step={step+1}/{args.steps} "
                  f"train_acc={sum(recent)/len(recent):.3f} held_out_acc={held_out:.3f} "
                  f"(chance=0.500)", flush=True)


if __name__ == "__main__":
    main()
