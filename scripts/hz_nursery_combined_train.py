#!/usr/bin/env python3
"""Combined multi-signal Language Nursery training, plans/Hatchling
world.md section 5's closing formula (L_LM + L_ground + L_action +
L_world + L_QA) and the Phase 0 checklist's last open item: L0-L6 have
only ever been trained as SEPARATE objectives via hz_nursery_train.py's
--stage flag, one curriculum stage at a time. This script trains one
shared HZLanguageModel on ALL EIGHT real sub-task losses (L0 next-token
LM, L1 grounding, L2 selection+consequence, L3 relation, L4 logic-AND,
L4 counting-verification, L5 QA recall, L6 reading) jointly -- one
combined backward pass per step, summing every loss with equal weight,
no curriculum ordering at all.

This directly answers a real, previously untested question: does
sharing one backbone across all seven task FAMILIES help (positive
transfer -- shared representations get more total gradient signal) or
hurt (negative transfer/interference -- eight different objectives
pulling the same weights in different directions each step) relative
to the sequential per-stage numbers already recorded in the plan? Every
held-out metric here is directly comparable to an existing number:
L0 perplexity (~2.08 sequential), L1 (100%), L2 selection (100%) /
consequence (~94-95%), L3 seen (100%) / unseen (~50-60%, concat_linear
encoder -- this script does NOT use the not-yet-promoted factorized
encoder, to keep this a one-variable-at-a-time comparison), L4-logic
seen/unseen (same band as L3), L4-counting (~65-72%), L5 (100%), L6
(~65-79%).

Uses the SAME architecture, SAME generators, and SAME tensor-packing
helpers as hz_nursery_train.py (imported directly, no duplication) --
only the training LOOP is new.
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
from hatchling_world.language.tokenizer import NurseryTokenizer, NOVEL_LABELS, SIZES  # noqa: E402
from hatchling_world.language.nursery_generator import (  # noqa: E402
    generate_l0_sentence, generate_l1_grounding_episode, generate_l2_verb_episode,
    generate_l3_relation_episode, generate_l4_logic_and_episode, generate_l4_counting_episode,
    generate_l5_qa_episode, generate_l6_reading_episode,
)


def l0_loss_fn(model, tok, rng, batch_size):
    token_ids = nt.l0_batch(tok, rng, batch_size)
    logits = model.lm_forward(token_ids)
    target = token_ids[:, 1:]
    mask = (target != tok.pad_id)
    loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1), reduction="none")
    return (loss * mask.reshape(-1).float()).sum() / mask.float().sum().clamp_min(1)


def combined_train_step(model, opt, tok, rngs, n_objects, n_sentences):
    losses = {}

    losses["l0"] = l0_loss_fn(model, tok, rngs["l0"], batch_size=8)

    ep = generate_l1_grounding_episode(rngs["l1"], n_objects=n_objects)
    instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
    logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
    losses["l1"] = F.cross_entropy(logits, target)

    ep = generate_l2_verb_episode(rngs["l2"], n_objects=n_objects)
    instr_ids, type_idx, color_idx, size_idx, pos_idx, held, opened, target, cons_target = nt.l2_episode_tensors(tok, ep)
    sel_logits, cons_logits = model.verb_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx, held, opened)
    losses["l2"] = F.cross_entropy(sel_logits, target) + F.binary_cross_entropy_with_logits(cons_logits, cons_target)

    ep = generate_l3_relation_episode(rngs["l3"], n_objects=n_objects, split="train")
    instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
    logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
    losses["l3"] = F.cross_entropy(logits, target)

    ep = generate_l4_logic_and_episode(rngs["l4logic"], n_objects=n_objects, split="train")
    instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
    logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
    losses["l4logic"] = F.cross_entropy(logits, target)

    ep = generate_l4_counting_episode(rngs["l4count"], n_objects=n_objects)
    instr_ids, type_idx, color_idx, size_idx, pos_idx, label = nt.l4_counting_tensors(tok, ep)
    logit = model.verify_count_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
    losses["l4count"] = F.binary_cross_entropy_with_logits(logit, label)

    ep = generate_l5_qa_episode(rngs["l5"], n_objects=n_objects)
    teach_ids, question_ids, type_idx, color_idx, size_idx, pos_idx, label_idx = nt.l5_episode_tensors(tok, ep)
    logits = model.qa_forward(teach_ids, question_ids, type_idx, color_idx, size_idx, pos_idx)
    losses["l5"] = F.cross_entropy(logits, label_idx)

    ep = generate_l6_reading_episode(rngs["l6"], n_sentences=n_sentences)
    sentence_ids_list, question_ids, answer_idx = nt.l6_episode_tensors(tok, ep)
    logits = model.read_forward(sentence_ids_list, question_ids)
    losses["l6"] = F.cross_entropy(logits, answer_idx)

    total = sum(losses.values())
    opt.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {k: v.item() for k, v in losses.items()}


def run_all_evals(model, tok, eval_rngs, n_objects, n_sentences, n_episodes):
    """Every held-out metric already established in the plan for the
    sequential per-stage runs, computed here under the combined model
    so they're directly comparable."""
    held_out_ids = nt.l0_batch(tok, eval_rngs["l0"], 200)
    with torch.no_grad():
        logits = model.lm_forward(held_out_ids)
        target = held_out_ids[:, 1:]
        mask = (target != tok.pad_id)
        loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1), reduction="none")
        loss = (loss * mask.reshape(-1).float()).sum() / mask.float().sum().clamp_min(1)
        l0_ppl = torch.exp(loss).item()

    l2_sel_acc, l2_cons_acc = nt.l2_eval(model, tok, eval_rngs["l2"], n_objects, n_episodes)
    return {
        "l0_held_out_ppl": l0_ppl,
        "l1_held_out_acc": nt.l1_eval(model, tok, eval_rngs["l1"], n_objects, n_episodes),
        "l2_held_out_sel_acc": l2_sel_acc,
        "l2_held_out_cons_acc": l2_cons_acc,
        "l3_held_out_seen_combo_acc": nt.l3_eval(model, tok, eval_rngs["l3s"], n_objects, n_episodes, split="train"),
        "l3_held_out_unseen_combo_acc": nt.l3_eval(model, tok, eval_rngs["l3u"], n_objects, n_episodes, split="test"),
        "l4logic_held_out_seen_combo_acc": nt.l4_logic_eval(model, tok, eval_rngs["l4logics"], n_objects, n_episodes, split="train"),
        "l4logic_held_out_unseen_combo_acc": nt.l4_logic_eval(model, tok, eval_rngs["l4logicu"], n_objects, n_episodes, split="test"),
        "l4count_held_out_acc": nt.l4_counting_eval(model, tok, eval_rngs["l4count"], n_objects, n_episodes),
        "l5_held_out_acc": nt.l5_eval(model, tok, eval_rngs["l5"], n_objects, n_episodes),
        "l6_held_out_acc": nt.l6_eval(model, tok, eval_rngs["l6"], n_sentences, n_episodes)[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--n-objects", type=int, default=4)
    parser.add_argument("--n-sentences", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_combined_train.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS), n_read_labels=len(SIZES))
    print(f"[combined] n_params={sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    task_names = ["l0", "l1", "l2", "l3", "l4logic", "l4count", "l5", "l6"]
    train_rngs = {name: random.Random(args.seed + i + 1) for i, name in enumerate(task_names)}
    eval_names = ["l0", "l1", "l2", "l3s", "l3u", "l4logics", "l4logicu", "l4count", "l5", "l6"]
    eval_rngs = {name: random.Random(args.seed + i + 1 + nt.TEST_SEED_OFFSET) for i, name in enumerate(eval_names)}

    history = []
    recent = {name: [] for name in task_names}
    for step in range(args.steps):
        losses = combined_train_step(model, opt, tok, train_rngs, args.n_objects, args.n_sentences)
        for name in task_names:
            recent[name].append(losses[name])
            recent[name][:] = recent[name][-200:]

        if (step + 1) % args.eval_every == 0:
            evals = run_all_evals(model, tok, eval_rngs, args.n_objects, args.n_sentences, args.eval_episodes)
            train_loss_str = " ".join(f"{k}={sum(v)/len(v):.3f}" for k, v in recent.items())
            print(f"[combined] step={step+1}/{args.steps} train_losses[{train_loss_str}]", flush=True)
            print(f"[combined] step={step+1}/{args.steps} held_out: " +
                  " ".join(f"{k}={v:.3f}" for k, v in evals.items()), flush=True)
            history.append({"step": step + 1, "train_losses": {k: sum(v) / len(v) for k, v in recent.items()},
                             "held_out": evals})

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[combined] wrote {args.results_file}", flush=True)

    print("\n[combined] === FINAL HELD-OUT vs SEQUENTIAL BASELINES (plan) ===")
    final = history[-1]["held_out"]
    baselines = {
        "l0_held_out_ppl": 2.08, "l1_held_out_acc": 1.00, "l2_held_out_sel_acc": 1.00,
        "l2_held_out_cons_acc": 0.945, "l3_held_out_seen_combo_acc": 1.00,
        "l3_held_out_unseen_combo_acc": 0.55, "l4logic_held_out_seen_combo_acc": 1.00,
        "l4logic_held_out_unseen_combo_acc": 0.50, "l4count_held_out_acc": 0.69,
        "l5_held_out_acc": 1.00, "l6_held_out_acc": 0.75,
    }
    print(f"{'metric':<32} {'combined':>10} {'sequential':>12}")
    for k, v in final.items():
        print(f"{k:<32} {v:>10.3f} {baselines.get(k, float('nan')):>12.3f}")


if __name__ == "__main__":
    main()
