#!/usr/bin/env python3
"""Phase 5 (depth), plans/Hatchling world.md's next unstarted phase.
Every real experiment this whole session fixed R (n_rounds_l1, the
reasoning workspace's tied-operator round count) at 8 -- never swept
it. This tests R in {1, 2, 4, 8, 16} directly, per section 2's KEEP
rule: sweeping the EXISTING recurrence's round-count hyperparameter is
not a "new recurrence experiment" (that rule is about not redesigning
the tied operator itself); it's the standard experiment type already
used in the FSM/PAPER work.

"Horizon buckets" (the original room-navigation framing) is reinterpreted
here as task DIFFICULTY buckets, matching this session's own real
findings -- four tasks spanning the full spectrum already characterized:
  - L1 (easy, saturates ~100% with R=8)
  - L3 unseen-combo (hard, generalization gap, noisy 30-92% with R=8)
  - L4-counting (hard, real capacity ceiling, ~65-72% with R=8, ruled
    out as a readout problem by two separate ablations)
  - L5-stress n_facts=3 (hardest, real memory-write/gate problem,
    ~24.5-35.4% with R=8, ruled out as a slot-diversity/routing problem
    by three separate ablations)

Real question: does simply giving H more reasoning rounds help ANY of
the three currently-open problems (L3, L4-counting, L5-stress), none
of which were ever diagnosed as a reasoning-depth issue? If R doesn't
move any of them, that's a real, disclosable negative result narrowing
the search further (not insufficient depth); if it does, that's a
genuinely actionable lever nothing in this whole session tried.

"Action efficiency vs R" (real wall-clock cost) is also measured
directly -- ws.run() loops R times, so cost should scale with R; worth
confirming rather than assuming, and worth knowing the real cost of
whatever R is eventually recommended.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hz_nursery_train as nt  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.tokenizer import NurseryTokenizer, NOVEL_LABELS  # noqa: E402
from hatchling_world.language.nursery_generator import (  # noqa: E402
    generate_l1_grounding_episode, generate_l3_relation_episode, generate_l4_counting_episode,
    generate_l5_stress_episode,
)


def train_l1(model, opt, tok, rng, n_objects, steps):
    for _ in range(steps):
        ep = generate_l1_grounding_episode(rng, n_objects=n_objects)
        instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
        logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def eval_l1(model, tok, rng, n_objects, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l1_grounding_episode(rng, n_objects=n_objects)
            instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
            logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            correct += int((logits.argmax(-1) == target).item())
    return correct / n_episodes


def train_l3(model, opt, tok, rng, n_objects, steps):
    for _ in range(steps):
        ep = generate_l3_relation_episode(rng, n_objects=n_objects, split="train")
        instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
        logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def eval_l3_unseen(model, tok, rng, n_objects, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l3_relation_episode(rng, n_objects=n_objects, split="test")
            instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
            logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            correct += int((logits.argmax(-1) == target).item())
    return correct / n_episodes


def train_l4_counting(model, opt, tok, rng, n_objects, steps):
    for _ in range(steps):
        ep = generate_l4_counting_episode(rng, n_objects=n_objects)
        instr_ids, type_idx, color_idx, size_idx, pos_idx, label = nt.l4_counting_tensors(tok, ep)
        logit = model.verify_count_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
        loss = F.binary_cross_entropy_with_logits(logit, label)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def eval_l4_counting(model, tok, rng, n_objects, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l4_counting_episode(rng, n_objects=n_objects)
            instr_ids, type_idx, color_idx, size_idx, pos_idx, label = nt.l4_counting_tensors(tok, ep)
            logit = model.verify_count_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            correct += int(((logit > 0).float() == label).item())
    return correct / n_episodes


def train_l5_stress(model, opt, tok, rng, steps):
    for _ in range(steps):
        ep = generate_l5_stress_episode(rng, n_facts=3, n_distractors=0)
        sequence_ids_list = [torch.tensor([tok.encode(s)]) for s in ep["sequence"]]
        question_ids = torch.tensor([tok.encode(ep["question"])])
        answer_idx = torch.tensor([ep["answer_idx"]])
        logits = model.stress_recall_forward(sequence_ids_list, question_ids)
        loss = F.cross_entropy(logits, answer_idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def eval_l5_stress(model, tok, rng, n_episodes):
    correct = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l5_stress_episode(rng, n_facts=3, n_distractors=0)
            sequence_ids_list = [torch.tensor([tok.encode(s)]) for s in ep["sequence"]]
            question_ids = torch.tensor([tok.encode(ep["question"])])
            answer_idx = torch.tensor([ep["answer_idx"]])
            logits = model.stress_recall_forward(sequence_ids_list, question_ids)
            correct += int((logits.argmax(-1).item() == ep["answer_idx"]))
    return correct / n_episodes


TASKS = {
    "L1_easy": {"train": train_l1, "eval": eval_l1, "chance": None},  # chance filled per n_objects below
    "L3_unseen_combo": {"train": train_l3, "eval": eval_l3_unseen, "chance": None},
    "L4_counting": {"train": train_l4_counting, "eval": eval_l4_counting, "chance": 0.5},
    "L5_stress_n3": {"train": train_l5_stress, "eval": eval_l5_stress, "chance": 1.0 / len(NOVEL_LABELS)},
}


def run_one(task_name, R, tok, args):
    torch.manual_seed(args.seed)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=8,
                             workspace_slots=args.workspace_slots, n_rounds_l1=R,
                             n_qa_labels=len(NOVEL_LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + nt.TEST_SEED_OFFSET)
    task = TASKS[task_name]

    start = time.perf_counter()
    if task_name == "L5_stress_n3":
        task["train"](model, opt, tok, train_rng, args.steps)
    else:
        task["train"](model, opt, tok, train_rng, args.n_objects, args.steps)
    elapsed = time.perf_counter() - start
    steps_per_sec = args.steps / elapsed

    if task_name == "L5_stress_n3":
        acc = task["eval"](model, tok, eval_rng, args.eval_episodes)
    else:
        acc = task["eval"](model, tok, eval_rng, args.n_objects, args.eval_episodes)

    chance = task["chance"] if task["chance"] is not None else 1.0 / args.n_objects
    return acc, chance, steps_per_sec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-objects", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--r-values", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_r_sweep.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()
    results = {}
    for task_name in TASKS:
        results[task_name] = {}
        for R in args.r_values:
            print(f"\n[r-sweep] ==== task={task_name} R={R} ====", flush=True)
            acc, chance, steps_per_sec = run_one(task_name, R, tok, args)
            results[task_name][R] = {"acc": acc, "chance": chance, "steps_per_sec": steps_per_sec}
            print(f"[r-sweep] task={task_name} R={R} acc={acc:.3f} (chance={chance:.3f}) "
                  f"steps_per_sec={steps_per_sec:.1f}", flush=True)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[r-sweep] wrote {args.results_file}", flush=True)

    print("\n[r-sweep] === SUMMARY (acc by R, chance in parens) ===")
    for task_name, by_r in results.items():
        chance = next(iter(by_r.values()))["chance"]
        row = "  ".join(f"R={R}:{v['acc']:.3f}" for R, v in by_r.items())
        print(f"{task_name:<20} (chance={chance:.3f})  {row}")

    print("\n[r-sweep] === SPEED (steps/sec by R) ===")
    for task_name, by_r in results.items():
        row = "  ".join(f"R={R}:{v['steps_per_sec']:.1f}" for R, v in by_r.items())
        print(f"{task_name:<20} {row}")


if __name__ == "__main__":
    main()
