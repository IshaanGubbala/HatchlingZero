#!/usr/bin/env python3
"""First real attempted FIX for the L5 memory cliff, explicit user
request 2026-09-05 ("try"), after two diagnostics fully localized the
problem (plans/Hatchling world.md):
  - the cliff is a real storage failure in S, not an H-retrieval issue
    (a direct linear probe on S can't recover even the first-taught
    fact after 3 facts are written);
  - the gate is content-blind: it writes ~46% new content into S
    whether the incoming sentence is a word-for-word repeat or a
    genuinely new fact (difference +0.0041, noise).

These two findings connect: S's M_S=8 slots were ALSO found to be
nearly collapsed to one effective dimension (participation ratio
~1.1-1.2 out of 8) even when recall works perfectly at n_facts=1. If
every slot's query vector is nearly identical, every slot attends to
roughly the same content and gets roughly the same delta_S -- so of
course the gate looks content-blind in aggregate: there's no room for
a per-slot content-sensitive decision when the slots aren't
differentiated from each other in the first place.

The fix tried here: an auxiliary DIVERSITY LOSS on S's slot vectors
(mean squared pairwise cosine similarity, pushed toward 0 -- i.e.
toward orthogonal slots) added to the task loss during training. This
does NOT modify HZCQPersistentMemory or HZCQReasoningWorkspace at all
(the "no new recurrence experiments" KEEP rule, and no changes to
mem.update's own code) -- it replicates the exact forward computation
externally (same pattern as hz_nursery_l5_gate_diagnostic.py's
update_with_gate_capture), this time keeping gradients, and adds one
extra loss term. If this works, it's a training-loss fix, not an
architecture change.

Real test, not assumed: does turning on this loss (a) actually break
the slot collapse (does participation ratio go up), and (b) does that
translate into a real held-out multi-fact recall improvement? Both are
measured directly, at several lambda values including 0.0 (the
existing, already-diagnosed baseline).
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


def stress_recall_forward_with_diversity(model, sequence_ids_list, question_ids):
    """Exact replica of HZLanguageModel.stress_recall_forward's own
    computation (same submodule calls, same order), but WITH gradients
    (no torch.no_grad anywhere the original didn't have it either --
    stress_recall_forward is already a plain differentiable method) and
    ALSO returns the diversity loss computed on S right after all
    sequence turns are ingested, before the question turn -- the exact
    moment all taught facts have just been written."""
    B = question_ids.shape[0]
    S = model.mem.init_state(B, device=question_ids.device)
    for sentence_ids in sequence_ids_list:
        for t in range(sentence_ids.shape[1]):
            S = model.mem.update(S, model.token_embed(sentence_ids[:, t]).unsqueeze(1))

    S0 = S[0]  # (M_S, D) -- diversity measured per-example; B=1 throughout this project's episodes
    normed = F.normalize(S0, dim=-1)
    sim = normed @ normed.T
    off_diag_mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    diversity_loss = (sim[off_diag_mask] ** 2).mean()

    for t in range(question_ids.shape[1]):
        S = model.mem.update(S, model.token_embed(question_ids[:, t]).unsqueeze(1))

    x_null = model.read_null_x.expand(B, 1, model.D)
    H = model.ws.run(B, S, x_null, n_rounds=model.n_rounds_l1)
    q = model.qa_rq(H).mean(dim=1, keepdim=True)
    scores = torch.matmul(q, model.qa_rk(H).transpose(-1, -2)) / (model.D ** 0.5)
    read = torch.matmul(F.softmax(scores, dim=-1), H).mean(dim=1)
    logits = model.qa_head(read)
    return logits, diversity_loss


def participation_ratio(S: torch.Tensor) -> float:
    sv = torch.linalg.svdvals(S)
    return float((sv.sum() ** 2) / (sv ** 2).sum())


def train_one(lam, tok, args):
    torch.manual_seed(args.seed)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=8,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + nt.TEST_SEED_OFFSET)

    for step in range(args.steps):
        ep = generate_l5_stress_episode(train_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
        sequence_ids_list = [torch.tensor([tok.encode(s)]) for s in ep["sequence"]]
        question_ids = torch.tensor([tok.encode(ep["question"])])
        answer_idx = torch.tensor([ep["answer_idx"]])
        logits, div_loss = stress_recall_forward_with_diversity(model, sequence_ids_list, question_ids)
        task_loss = F.cross_entropy(logits, answer_idx)
        loss = task_loss + lam * div_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.eval_every == 0:
            correct = 0
            ranks = []
            with torch.no_grad():
                for _ in range(args.eval_episodes):
                    e = generate_l5_stress_episode(eval_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
                    seq_ids = [torch.tensor([tok.encode(s)]) for s in e["sequence"]]
                    q_ids = torch.tensor([tok.encode(e["question"])])
                    a_idx = torch.tensor([e["answer_idx"]])
                    lg, _ = stress_recall_forward_with_diversity(model, seq_ids, q_ids)
                    correct += int((lg.argmax(-1) == a_idx).item())
                    S = model.mem.init_state(1)
                    for sentence in e["sequence"]:
                        ids = torch.tensor([tok.encode(sentence)])
                        for t in range(ids.shape[1]):
                            S = model.mem.update(S, model.token_embed(ids[:, t]).unsqueeze(1))
                    ranks.append(participation_ratio(S[0]))
            acc = correct / args.eval_episodes
            mean_rank = sum(ranks) / len(ranks)
            print(f"[div-fix][lambda={lam}] step={step+1}/{args.steps} held_out_acc={acc:.3f} "
                  f"mean_participation_ratio={mean_rank:.3f} (max=8)", flush=True)

    correct = 0
    ranks = []
    with torch.no_grad():
        for _ in range(args.eval_episodes * 2):
            e = generate_l5_stress_episode(eval_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
            seq_ids = [torch.tensor([tok.encode(s)]) for s in e["sequence"]]
            q_ids = torch.tensor([tok.encode(e["question"])])
            a_idx = torch.tensor([e["answer_idx"]])
            lg, _ = stress_recall_forward_with_diversity(model, seq_ids, q_ids)
            correct += int((lg.argmax(-1) == a_idx).item())
            S = model.mem.init_state(1)
            for sentence in e["sequence"]:
                ids = torch.tensor([tok.encode(sentence)])
                for t in range(ids.shape[1]):
                    S = model.mem.update(S, model.token_embed(ids[:, t]).unsqueeze(1))
            ranks.append(participation_ratio(S[0]))
    final_acc = correct / (args.eval_episodes * 2)
    final_rank = sum(ranks) / len(ranks)
    return final_acc, final_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-facts", type=int, default=3)
    parser.add_argument("--n-distractors", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_l5_diversity_loss_fix.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()
    results = {}
    for lam in args.lambdas:
        print(f"\n[div-fix] ==== lambda={lam} ====", flush=True)
        acc, rank = train_one(lam, tok, args)
        results[lam] = {"held_out_acc": acc, "mean_participation_ratio": rank}
        print(f"[div-fix] lambda={lam} FINAL held_out_acc={acc:.3f} mean_participation_ratio={rank:.3f}", flush=True)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[div-fix] wrote {args.results_file}", flush=True)

    print("\n[div-fix] === SUMMARY ===")
    print(f"{'lambda':>8} {'held_out_acc':>14} {'participation_ratio':>22}")
    for lam, r in results.items():
        print(f"{lam:>8} {r['held_out_acc']:>14.3f} {r['mean_participation_ratio']:>22.3f}")


if __name__ == "__main__":
    main()
