#!/usr/bin/env python3
"""L5 memory stress test, explicit user request 2026-09-05: L5's
original episode taught exactly one fact with zero interference. This
characterizes the REAL capacity/interference properties of S -- teach
2-4 novel facts, interleave distractor sentences, ask about one fact
later -- across a sweep of (n_facts, n_distractors) configurations, not
just a single pass/fail number.

For each config, trains ONE fresh model end-to-end on
generate_l5_stress_episode + HZLanguageModel.stress_recall_forward,
then reports held-out accuracy broken down by:
  - query_idx: which TAUGHT-order fact was asked about (0=taught first)
  - fact_position: where that teaching actually landed in the full
    interleaved passage (0=passage start)
so "forgetting because taught long ago" and "forgetting because buried
under later turns" can be told apart.
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
from hatchling_world.language.tokenizer import NurseryTokenizer, NOVEL_LABELS  # noqa: E402
from hatchling_world.language.nursery_generator import generate_l5_stress_episode  # noqa: E402


def episode_tensors(tok, ep):
    sequence_ids_list = [torch.tensor([tok.encode(s)]) for s in ep["sequence"]]
    question_ids = torch.tensor([tok.encode(ep["question"])])
    answer_idx = torch.tensor([ep["answer_idx"]])
    return sequence_ids_list, question_ids, answer_idx


def train_step(model, opt, tok, rng, n_facts, n_distractors):
    ep = generate_l5_stress_episode(rng, n_facts=n_facts, n_distractors=n_distractors)
    sequence_ids_list, question_ids, answer_idx = episode_tensors(tok, ep)
    logits = model.stress_recall_forward(sequence_ids_list, question_ids)
    loss = F.cross_entropy(logits, answer_idx)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item(), (logits.argmax(-1) == answer_idx).float().item()


def evaluate(model, tok, rng, n_facts, n_distractors, n_episodes):
    correct = 0
    by_query_idx, by_fact_position = {}, {}
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = generate_l5_stress_episode(rng, n_facts=n_facts, n_distractors=n_distractors)
            sequence_ids_list, question_ids, answer_idx = episode_tensors(tok, ep)
            logits = model.stress_recall_forward(sequence_ids_list, question_ids)
            is_correct = int((logits.argmax(-1) == answer_idx).item())
            correct += is_correct
            by_query_idx.setdefault(ep["query_idx"], []).append(is_correct)
            by_fact_position.setdefault(ep["fact_position"], []).append(is_correct)
    acc = correct / n_episodes
    by_query_idx = {k: sum(v) / len(v) for k, v in by_query_idx.items()}
    by_fact_position = {k: sum(v) / len(v) for k, v in by_fact_position.items()}
    return acc, by_query_idx, by_fact_position


def run_config(n_facts, n_distractors, tok, args):
    torch.manual_seed(args.seed)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + nt.TEST_SEED_OFFSET)

    recent = []
    for step in range(args.steps):
        loss, acc = train_step(model, opt, tok, train_rng, n_facts, n_distractors)
        recent.append(acc)
        recent[:] = recent[-200:]
        if (step + 1) % args.eval_every == 0:
            held_out_acc, by_qi, by_fp = evaluate(model, tok, eval_rng, n_facts, n_distractors, args.eval_episodes)
            print(f"[stress][facts={n_facts} distractors={n_distractors}] step={step+1}/{args.steps} "
                  f"train_acc={sum(recent)/len(recent):.3f} held_out_acc={held_out_acc:.3f} "
                  f"by_query_idx={by_qi} by_fact_position={by_fp}", flush=True)

    final_acc, final_by_qi, final_by_fp = evaluate(model, tok, eval_rng, n_facts, n_distractors, args.eval_episodes * 2)
    chance = 1.0 / len(NOVEL_LABELS)
    return {"n_facts": n_facts, "n_distractors": n_distractors, "held_out_acc": final_acc,
            "by_query_idx": final_by_qi, "by_fact_position": final_by_fp, "chance": chance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_l5_memory_stress.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()
    configs = [(2, 0), (3, 0), (4, 0), (2, 2), (3, 2), (4, 2), (3, 4)]
    results = []
    for n_facts, n_distractors in configs:
        print(f"[stress] ==== n_facts={n_facts} n_distractors={n_distractors} ====", flush=True)
        results.append(run_config(n_facts, n_distractors, tok, args))

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[stress] wrote {args.results_file}", flush=True)

    print("\n[stress] === SUMMARY ===")
    print(f"{'facts':>6} {'distractors':>12} {'held_out_acc':>14} {'chance':>8}")
    for r in results:
        print(f"{r['n_facts']:>6} {r['n_distractors']:>12} {r['held_out_acc']:>14.3f} {r['chance']:>8.3f}")


if __name__ == "__main__":
    main()
