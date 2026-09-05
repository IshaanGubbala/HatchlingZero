#!/usr/bin/env python3
"""Language Nursery training, plans/Hatchling world.md section 5,
Stages L0 (self-supervised LM) and L1 (grounded nouns/properties).
Real teacher-forced next-token prediction for L0; real behavioral
grounding (select the correct object) for L1, with a held-out
(different seed range) eval split, matching this project's real
train/test discipline throughout.
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
from hatchling_world.language.tokenizer import NurseryTokenizer, NOUNS, COLORS, SIZES, POSITIONS
from hatchling_world.language.nursery_generator import generate_l0_sentence, generate_l1_grounding_episode

TEST_SEED_OFFSET = 10_000_000


def l0_batch(tok: NurseryTokenizer, rng: random.Random, batch: int):
    sentences = [generate_l0_sentence(rng) for _ in range(batch)]
    encoded = [tok.encode(s) for s in sentences]
    max_len = max(len(e) for e in encoded)
    ids = torch.full((batch, max_len), tok.pad_id, dtype=torch.long)
    for i, e in enumerate(encoded):
        ids[i, :len(e)] = torch.tensor(e)
    return ids


def l0_train_step(model, opt, tok, rng, batch_size):
    token_ids = l0_batch(tok, rng, batch_size)
    logits = model.lm_forward(token_ids)  # (B, T-1, V)
    target = token_ids[:, 1:]
    mask = (target != tok.pad_id)
    loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1), reduction="none")
    loss = (loss * mask.reshape(-1).float()).sum() / mask.float().sum().clamp_min(1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        pred = logits.argmax(-1)
        acc = ((pred == target) & mask).float().sum() / mask.float().sum().clamp_min(1)
    return loss.item(), acc.item()


def l1_episode_tensors(tok: NurseryTokenizer, ep: dict):
    instr_ids = torch.tensor([tok.encode(ep["instruction"])])
    type_idx = torch.tensor([[NOUNS.index(o["type"]) for o in ep["objects"]]])
    color_idx = torch.tensor([[COLORS.index(o["color"]) for o in ep["objects"]]])
    size_idx = torch.tensor([[SIZES.index(o["size"]) for o in ep["objects"]]])
    pos_idx = torch.tensor([[POSITIONS.index(o["position"]) for o in ep["objects"]]])
    target = torch.tensor([ep["target_idx"]])
    return instr_ids, type_idx, color_idx, size_idx, pos_idx, target


def l1_train_step(model, opt, tok, rng, n_objects):
    ep = generate_l1_grounding_episode(rng, n_objects=n_objects)
    instr_ids, type_idx, color_idx, size_idx, pos_idx, target = l1_episode_tensors(tok, ep)
    logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
    loss = F.cross_entropy(logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    acc = (logits.argmax(-1) == target).float().item()
    return loss.item(), acc


def l1_eval(model, tok, rng, n_objects, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l1_grounding_episode(rng, n_objects=n_objects)
            instr_ids, type_idx, color_idx, size_idx, pos_idx, target = l1_episode_tensors(tok, ep)
            logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            correct += int((logits.argmax(-1) == target).item())
    return correct / n_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["l0", "l1", "both"], default="both")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l0-steps", type=int, default=3000)
    parser.add_argument("--l0-batch-size", type=int, default=16)
    parser.add_argument("--l1-steps", type=int, default=3000)
    parser.add_argument("--l1-n-objects", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[nursery] vocab_size={tok.vocab_size} n_params={n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.stage in ("l0", "both"):
        train_rng = random.Random(args.seed + 1)
        eval_rng = random.Random(args.seed + TEST_SEED_OFFSET)
        recent_loss, recent_acc = [], []
        for step in range(args.l0_steps):
            loss, acc = l0_train_step(model, opt, tok, train_rng, args.l0_batch_size)
            recent_loss.append(loss); recent_acc.append(acc)
            recent_loss[:] = recent_loss[-200:]; recent_acc[:] = recent_acc[-200:]
            if (step + 1) % args.eval_every == 0:
                held_out_ids = l0_batch(tok, eval_rng, 200)
                with torch.no_grad():
                    logits = model.lm_forward(held_out_ids)
                    target = held_out_ids[:, 1:]
                    mask = (target != tok.pad_id)
                    eval_loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1),
                                                 reduction="none")
                    eval_loss = (eval_loss * mask.reshape(-1).float()).sum() / mask.float().sum().clamp_min(1)
                    eval_ppl = torch.exp(eval_loss).item()
                print(f"[nursery][L0] step={step+1}/{args.l0_steps} train_loss={sum(recent_loss)/len(recent_loss):.4f} "
                      f"train_next_token_acc={sum(recent_acc)/len(recent_acc):.3f} "
                      f"held_out_ppl={eval_ppl:.3f} (chance_ppl~={tok.vocab_size})", flush=True)

    if args.stage in ("l1", "both"):
        train_rng = random.Random(args.seed + 2)
        eval_rng = random.Random(args.seed + 2 + TEST_SEED_OFFSET)
        recent_acc = []
        for step in range(args.l1_steps):
            loss, acc = l1_train_step(model, opt, tok, train_rng, args.l1_n_objects)
            recent_acc.append(acc)
            recent_acc[:] = recent_acc[-200:]
            if (step + 1) % args.eval_every == 0:
                held_out_acc = l1_eval(model, tok, eval_rng, args.l1_n_objects, args.eval_episodes)
                chance = 1.0 / args.l1_n_objects
                print(f"[nursery][L1] step={step+1}/{args.l1_steps} train_acc={sum(recent_acc)/len(recent_acc):.3f} "
                      f"held_out_acc={held_out_acc:.3f} (chance={chance:.3f})", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_checkpoint)
        print(f"[nursery] saved checkpoint to {args.save_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
