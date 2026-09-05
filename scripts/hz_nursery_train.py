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
from hatchling_world.language.tokenizer import NurseryTokenizer, NOUNS, COLORS, SIZES, POSITIONS, NOVEL_LABELS
from hatchling_world.language.nursery_generator import (
    generate_l0_sentence, generate_l1_grounding_episode, generate_l2_verb_episode,
    generate_l3_relation_episode, generate_l4_logic_and_episode, generate_l4_counting_episode,
    generate_l5_qa_episode,
)

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


def l2_episode_tensors(tok: NurseryTokenizer, ep: dict):
    instr_ids = torch.tensor([tok.encode(ep["instruction"])])
    type_idx = torch.tensor([[NOUNS.index(o["type"]) for o in ep["objects"]]])
    color_idx = torch.tensor([[COLORS.index(o["color"]) for o in ep["objects"]]])
    size_idx = torch.tensor([[SIZES.index(o["size"]) for o in ep["objects"]]])
    pos_idx = torch.tensor([[POSITIONS.index(o["position"]) for o in ep["objects"]]])
    held = torch.tensor([[float(o["held"]) for o in ep["objects"]]])
    opened = torch.tensor([[float(o["opened"]) for o in ep["objects"]]])
    target = torch.tensor([ep["target_idx"]])
    consequence_target = torch.tensor([[
        1.0 if ep["position_after"] == "right" else 0.0,
        float(ep["held_after"]),
        float(ep["opened_after"]),
    ]])
    return instr_ids, type_idx, color_idx, size_idx, pos_idx, held, opened, target, consequence_target


def l2_train_step(model, opt, tok, rng, n_objects):
    ep = generate_l2_verb_episode(rng, n_objects=n_objects)
    instr_ids, type_idx, color_idx, size_idx, pos_idx, held, opened, target, cons_target = \
        l2_episode_tensors(tok, ep)
    sel_logits, cons_logits = model.verb_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx, held, opened)
    sel_loss = F.cross_entropy(sel_logits, target)
    cons_loss = F.binary_cross_entropy_with_logits(cons_logits, cons_target)
    loss = sel_loss + cons_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        sel_acc = (sel_logits.argmax(-1) == target).float().item()
        cons_acc = ((cons_logits > 0).float() == cons_target).float().mean().item()
    return loss.item(), sel_acc, cons_acc


def l2_eval(model, tok, rng, n_objects, n_episodes):
    sel_correct, cons_correct, cons_total = 0, 0, 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l2_verb_episode(rng, n_objects=n_objects)
            instr_ids, type_idx, color_idx, size_idx, pos_idx, held, opened, target, cons_target = \
                l2_episode_tensors(tok, ep)
            sel_logits, cons_logits = model.verb_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx, held, opened)
            sel_correct += int((sel_logits.argmax(-1) == target).item())
            cons_correct += int(((cons_logits > 0).float() == cons_target).float().sum().item())
            cons_total += cons_target.numel()
    return sel_correct / n_episodes, cons_correct / cons_total


def l3_train_step(model, opt, tok, rng, n_objects):
    ep = generate_l3_relation_episode(rng, n_objects=n_objects, split="train")
    instr_ids, type_idx, color_idx, size_idx, pos_idx, target = l1_episode_tensors(tok, ep)
    logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
    loss = F.cross_entropy(logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    acc = (logits.argmax(-1) == target).float().item()
    return loss.item(), acc


def l3_eval(model, tok, rng, n_objects, n_episodes, split):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l3_relation_episode(rng, n_objects=n_objects, split=split)
            instr_ids, type_idx, color_idx, size_idx, pos_idx, target = l1_episode_tensors(tok, ep)
            logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            correct += int((logits.argmax(-1) == target).item())
    return correct / n_episodes


def l4_logic_train_step(model, opt, tok, rng, n_objects):
    ep = generate_l4_logic_and_episode(rng, n_objects=n_objects, split="train")
    instr_ids, type_idx, color_idx, size_idx, pos_idx, target = l1_episode_tensors(tok, ep)
    logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
    loss = F.cross_entropy(logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    acc = (logits.argmax(-1) == target).float().item()
    return loss.item(), acc


def l4_logic_eval(model, tok, rng, n_objects, n_episodes, split):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l4_logic_and_episode(rng, n_objects=n_objects, split=split)
            instr_ids, type_idx, color_idx, size_idx, pos_idx, target = l1_episode_tensors(tok, ep)
            logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            correct += int((logits.argmax(-1) == target).item())
    return correct / n_episodes


def l4_counting_tensors(tok: NurseryTokenizer, ep: dict):
    instr_ids = torch.tensor([tok.encode(ep["instruction"])])
    type_idx = torch.tensor([[NOUNS.index(o["type"]) for o in ep["objects"]]])
    color_idx = torch.tensor([[COLORS.index(o["color"]) for o in ep["objects"]]])
    size_idx = torch.tensor([[SIZES.index(o["size"]) for o in ep["objects"]]])
    pos_idx = torch.tensor([[POSITIONS.index(o["position"]) for o in ep["objects"]]])
    label = torch.tensor([float(ep["label"])])
    return instr_ids, type_idx, color_idx, size_idx, pos_idx, label


def l4_counting_train_step(model, opt, tok, rng, n_objects):
    ep = generate_l4_counting_episode(rng, n_objects=n_objects)
    instr_ids, type_idx, color_idx, size_idx, pos_idx, label = l4_counting_tensors(tok, ep)
    logit = model.verify_count_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
    loss = F.binary_cross_entropy_with_logits(logit, label)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    acc = ((logit > 0).float() == label).float().item()
    return loss.item(), acc


def l4_counting_eval(model, tok, rng, n_objects, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l4_counting_episode(rng, n_objects=n_objects)
            instr_ids, type_idx, color_idx, size_idx, pos_idx, label = l4_counting_tensors(tok, ep)
            logit = model.verify_count_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            correct += int(((logit > 0).float() == label).item())
    return correct / n_episodes


def l5_episode_tensors(tok: NurseryTokenizer, ep: dict):
    teach_ids = torch.tensor([tok.encode(ep["teach"])])
    question_ids = torch.tensor([tok.encode(ep["question"])])
    type_idx = torch.tensor([[NOUNS.index(o["type"]) for o in ep["objects"]]])
    color_idx = torch.tensor([[COLORS.index(o["color"]) for o in ep["objects"]]])
    size_idx = torch.tensor([[SIZES.index(o["size"]) for o in ep["objects"]]])
    pos_idx = torch.tensor([[POSITIONS.index(o["position"]) for o in ep["objects"]]])
    label_idx = torch.tensor([ep["label_idx"]])
    return teach_ids, question_ids, type_idx, color_idx, size_idx, pos_idx, label_idx


def l5_train_step(model, opt, tok, rng, n_objects):
    ep = generate_l5_qa_episode(rng, n_objects=n_objects)
    teach_ids, question_ids, type_idx, color_idx, size_idx, pos_idx, label_idx = l5_episode_tensors(tok, ep)
    logits = model.qa_forward(teach_ids, question_ids, type_idx, color_idx, size_idx, pos_idx)
    loss = F.cross_entropy(logits, label_idx)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    acc = (logits.argmax(-1) == label_idx).float().item()
    return loss.item(), acc


def l5_eval(model, tok, rng, n_objects, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l5_qa_episode(rng, n_objects=n_objects)
            teach_ids, question_ids, type_idx, color_idx, size_idx, pos_idx, label_idx = l5_episode_tensors(tok, ep)
            logits = model.qa_forward(teach_ids, question_ids, type_idx, color_idx, size_idx, pos_idx)
            correct += int((logits.argmax(-1) == label_idx).item())
    return correct / n_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["l0", "l1", "l2", "l3", "l4", "l5", "both"], default="both")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l0-steps", type=int, default=3000)
    parser.add_argument("--l0-batch-size", type=int, default=16)
    parser.add_argument("--l1-steps", type=int, default=3000)
    parser.add_argument("--l1-n-objects", type=int, default=4)
    parser.add_argument("--l2-steps", type=int, default=3000)
    parser.add_argument("--l2-n-objects", type=int, default=4)
    parser.add_argument("--l3-steps", type=int, default=3000)
    parser.add_argument("--l3-n-objects", type=int, default=4)
    parser.add_argument("--l4-logic-steps", type=int, default=2500)
    parser.add_argument("--l4-counting-steps", type=int, default=2500)
    parser.add_argument("--l4-n-objects", type=int, default=4)
    parser.add_argument("--l5-steps", type=int, default=3000)
    parser.add_argument("--l5-n-objects", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
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

    if args.stage in ("l2",):
        train_rng = random.Random(args.seed + 3)
        eval_rng = random.Random(args.seed + 3 + TEST_SEED_OFFSET)
        recent_sel_acc, recent_cons_acc = [], []
        for step in range(args.l2_steps):
            loss, sel_acc, cons_acc = l2_train_step(model, opt, tok, train_rng, args.l2_n_objects)
            recent_sel_acc.append(sel_acc); recent_cons_acc.append(cons_acc)
            recent_sel_acc[:] = recent_sel_acc[-200:]; recent_cons_acc[:] = recent_cons_acc[-200:]
            if (step + 1) % args.eval_every == 0:
                held_out_sel_acc, held_out_cons_acc = l2_eval(model, tok, eval_rng, args.l2_n_objects, args.eval_episodes)
                chance_sel = 1.0 / args.l2_n_objects
                print(f"[nursery][L2] step={step+1}/{args.l2_steps} "
                      f"train_sel_acc={sum(recent_sel_acc)/len(recent_sel_acc):.3f} "
                      f"train_cons_acc={sum(recent_cons_acc)/len(recent_cons_acc):.3f} "
                      f"held_out_sel_acc={held_out_sel_acc:.3f} (chance={chance_sel:.3f}) "
                      f"held_out_cons_acc={held_out_cons_acc:.3f} (chance=0.500)", flush=True)

    if args.stage in ("l3",):
        train_rng = random.Random(args.seed + 4)
        eval_seen_rng = random.Random(args.seed + 4 + TEST_SEED_OFFSET)
        eval_unseen_rng = random.Random(args.seed + 4 + 2 * TEST_SEED_OFFSET)
        recent_acc = []
        for step in range(args.l3_steps):
            loss, acc = l3_train_step(model, opt, tok, train_rng, args.l3_n_objects)
            recent_acc.append(acc)
            recent_acc[:] = recent_acc[-200:]
            if (step + 1) % args.eval_every == 0:
                seen_combo_acc = l3_eval(model, tok, eval_seen_rng, args.l3_n_objects, args.eval_episodes, split="train")
                unseen_combo_acc = l3_eval(model, tok, eval_unseen_rng, args.l3_n_objects, args.eval_episodes, split="test")
                chance = 1.0 / args.l3_n_objects
                print(f"[nursery][L3] step={step+1}/{args.l3_steps} train_acc={sum(recent_acc)/len(recent_acc):.3f} "
                      f"held_out_seen_combo_acc={seen_combo_acc:.3f} "
                      f"held_out_UNSEEN_combo_acc={unseen_combo_acc:.3f} (chance={chance:.3f})", flush=True)

    if args.stage in ("l4",):
        train_rng = random.Random(args.seed + 5)
        eval_seen_rng = random.Random(args.seed + 5 + TEST_SEED_OFFSET)
        eval_unseen_rng = random.Random(args.seed + 5 + 2 * TEST_SEED_OFFSET)
        recent_acc = []
        for step in range(args.l4_logic_steps):
            loss, acc = l4_logic_train_step(model, opt, tok, train_rng, args.l4_n_objects)
            recent_acc.append(acc)
            recent_acc[:] = recent_acc[-200:]
            if (step + 1) % args.eval_every == 0:
                seen = l4_logic_eval(model, tok, eval_seen_rng, args.l4_n_objects, args.eval_episodes, split="train")
                unseen = l4_logic_eval(model, tok, eval_unseen_rng, args.l4_n_objects, args.eval_episodes, split="test")
                chance = 1.0 / args.l4_n_objects
                print(f"[nursery][L4-logic] step={step+1}/{args.l4_logic_steps} "
                      f"train_acc={sum(recent_acc)/len(recent_acc):.3f} "
                      f"held_out_seen_combo_acc={seen:.3f} held_out_UNSEEN_combo_acc={unseen:.3f} "
                      f"(chance={chance:.3f})", flush=True)

        train_rng = random.Random(args.seed + 6)
        eval_rng = random.Random(args.seed + 6 + TEST_SEED_OFFSET)
        recent_acc = []
        for step in range(args.l4_counting_steps):
            loss, acc = l4_counting_train_step(model, opt, tok, train_rng, args.l4_n_objects)
            recent_acc.append(acc)
            recent_acc[:] = recent_acc[-200:]
            if (step + 1) % args.eval_every == 0:
                held_out_acc = l4_counting_eval(model, tok, eval_rng, args.l4_n_objects, args.eval_episodes)
                print(f"[nursery][L4-counting] step={step+1}/{args.l4_counting_steps} "
                      f"train_acc={sum(recent_acc)/len(recent_acc):.3f} "
                      f"held_out_acc={held_out_acc:.3f} (chance=0.500)", flush=True)

    if args.stage in ("l5",):
        train_rng = random.Random(args.seed + 7)
        eval_rng = random.Random(args.seed + 7 + TEST_SEED_OFFSET)
        recent_acc = []
        for step in range(args.l5_steps):
            loss, acc = l5_train_step(model, opt, tok, train_rng, args.l5_n_objects)
            recent_acc.append(acc)
            recent_acc[:] = recent_acc[-200:]
            if (step + 1) % args.eval_every == 0:
                held_out_acc = l5_eval(model, tok, eval_rng, args.l5_n_objects, args.eval_episodes)
                chance = 1.0 / len(NOVEL_LABELS)
                print(f"[nursery][L5] step={step+1}/{args.l5_steps} "
                      f"train_acc={sum(recent_acc)/len(recent_acc):.3f} "
                      f"held_out_acc={held_out_acc:.3f} (chance={chance:.3f})", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_checkpoint)
        print(f"[nursery] saved checkpoint to {args.save_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
