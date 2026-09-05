#!/usr/bin/env python3
"""Fourth attempted memory-cliff fix, and the first one motivated by a
hard mathematical fact rather than a trained-behavior hypothesis.

Every earlier fix in this thread (diversity loss on S, top-k routing
on the gate logit) failed to move held-out recall at all. Before
trying a third loss-shaping idea, inspecting the actual math behind
mem.update() reveals why: every fact sentence in this codebase is
ingested ONE TOKEN AT A TIME (`S = mem.update(S, token_embed(tok)
.unsqueeze(1))` in a loop), so demo_hidden always has T_demo=1. Softmax
over a length-1 vector is mathematically ALWAYS exactly 1.0, regardless
of the scores -- so `read = attn @ V` reduces to exactly V for EVERY
slot, meaning delta_S is IDENTICAL across all M_S slots on every
single-token update, regardless of Q. The cross-attention mechanism
has literally nothing to discriminate over when there is only one
token to attend to. Neither prior fix could have touched this: slot
diversity in S and gate competition both operate downstream of a
delta_S that was already forced identical across slots by construction.

The fix tried here: ingest each WHOLE FACT SENTENCE as one multi-token
chunk (T_demo = sentence length) instead of looping token-by-token.
This is not a change to HZCQPersistentMemory's code at all -- `update`
already accepts demo_hidden of any T_demo -- it is a change in HOW the
language model CALLS it, using mem.update once per sentence instead of
once per token. With T_demo>1, cross-attention finally has multiple
real positions to discriminate over, so different slots' queries CAN
(if anything makes them want to) prefer different tokens and receive
genuinely different delta_S values for the first time in this whole
diagnostic thread.

Compares: token-by-token (the existing behavior, reproducing every
earlier result) vs whole-sentence-per-update. Same n_facts=3 config,
same held-out accuracy + participation ratio + fact-decoding probe
methodology as every earlier script in this thread.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hz_nursery_train as nt  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.tokenizer import NurseryTokenizer, NOVEL_LABELS  # noqa: E402
from hatchling_world.language.nursery_generator import generate_l5_stress_episode  # noqa: E402


def stress_recall_forward_variant(model, tok, sequence, question, whole_sentence: bool):
    """whole_sentence=False: exact replica of the existing
    stress_recall_forward (one mem.update() call per TOKEN).
    whole_sentence=True: one mem.update() call per SENTENCE, with all
    of that sentence's token embeddings passed as one (1, T, D) chunk
    -- the only change under test."""
    B = 1
    S = model.mem.init_state(B)
    for sentence in sequence:
        ids = torch.tensor([tok.encode(sentence)])
        if whole_sentence:
            hidden = model.token_embed(ids)  # (1, T, D) -- the whole sentence at once
            S = model.mem.update(S, hidden)
        else:
            for t in range(ids.shape[1]):
                S = model.mem.update(S, model.token_embed(ids[:, t]).unsqueeze(1))

    q_ids = torch.tensor([tok.encode(question)])
    if whole_sentence:
        S = model.mem.update(S, model.token_embed(q_ids))
    else:
        for t in range(q_ids.shape[1]):
            S = model.mem.update(S, model.token_embed(q_ids[:, t]).unsqueeze(1))

    x_null = model.read_null_x.expand(B, 1, model.D)
    H = model.ws.run(B, S, x_null, n_rounds=model.n_rounds_l1)
    q = model.qa_rq(H).mean(dim=1, keepdim=True)
    scores = torch.matmul(q, model.qa_rk(H).transpose(-1, -2)) / (model.D ** 0.5)
    read = torch.matmul(F.softmax(scores, dim=-1), H).mean(dim=1)
    logits = model.qa_head(read)
    return logits, S


def participation_ratio(S: torch.Tensor) -> float:
    sv = torch.linalg.svdvals(S)
    return float((sv.sum() ** 2) / (sv ** 2).sum())


class FactProbe(nn.Module):
    def __init__(self, d_model: int, n_labels: int):
        super().__init__()
        self.head = nn.Linear(d_model, n_labels)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        return self.head(S.mean(dim=1))


def train_one(whole_sentence, tok, args):
    torch.manual_seed(args.seed)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=8,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + nt.TEST_SEED_OFFSET)

    for step in range(args.steps):
        ep = generate_l5_stress_episode(train_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
        answer_idx = torch.tensor([ep["answer_idx"]])
        logits, _ = stress_recall_forward_variant(model, tok, ep["sequence"], ep["question"], whole_sentence)
        loss = F.cross_entropy(logits, answer_idx)
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
                    a_idx = torch.tensor([e["answer_idx"]])
                    lg, S = stress_recall_forward_variant(model, tok, e["sequence"], e["question"], whole_sentence)
                    correct += int((lg.argmax(-1) == a_idx).item())
                    ranks.append(participation_ratio(S[0]))
            acc = correct / args.eval_episodes
            mean_rank = sum(ranks) / len(ranks)
            print(f"[whole-sentence={whole_sentence}] step={step+1}/{args.steps} held_out_acc={acc:.3f} "
                  f"mean_participation_ratio={mean_rank:.3f}", flush=True)

    correct = 0
    ranks = []
    with torch.no_grad():
        for _ in range(args.eval_episodes * 2):
            e = generate_l5_stress_episode(eval_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
            a_idx = torch.tensor([e["answer_idx"]])
            lg, S = stress_recall_forward_variant(model, tok, e["sequence"], e["question"], whole_sentence)
            correct += int((lg.argmax(-1) == a_idx).item())
            ranks.append(participation_ratio(S[0]))
    final_acc = correct / (args.eval_episodes * 2)
    final_rank = sum(ranks) / len(ranks)

    for p in model.parameters():
        p.requires_grad_(False)
    probe_results = {}
    for query_idx in range(args.n_facts):
        probe = FactProbe(args.d_model, len(NOVEL_LABELS))
        probe_opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        probe_train_rng = random.Random(args.seed + 900 + query_idx)
        probe_eval_rng = random.Random(args.seed + 900 + query_idx + nt.TEST_SEED_OFFSET)

        def build_S_and_label(rng):
            ep = generate_l5_stress_episode(rng, n_facts=args.n_facts, n_distractors=0)
            with torch.no_grad():
                _, S = stress_recall_forward_variant(model, tok, ep["sequence"], ep["question"], whole_sentence)
            label_idx = torch.tensor([NOVEL_LABELS.index(ep["labels"][query_idx])])
            return S, label_idx

        for _ in range(args.probe_steps):
            S, label_idx = build_S_and_label(probe_train_rng)
            logits = probe(S)
            ploss = F.cross_entropy(logits, label_idx)
            probe_opt.zero_grad(set_to_none=True)
            ploss.backward()
            probe_opt.step()

        pcorrect = 0
        with torch.no_grad():
            for _ in range(args.probe_eval_episodes):
                S, label_idx = build_S_and_label(probe_eval_rng)
                logits = probe(S)
                pcorrect += int((logits.argmax(-1) == label_idx).item())
        probe_results[query_idx] = pcorrect / args.probe_eval_episodes
    for p in model.parameters():
        p.requires_grad_(True)

    return final_acc, final_rank, probe_results


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
    parser.add_argument("--probe-steps", type=int, default=1500)
    parser.add_argument("--probe-eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_l5_whole_sentence_write_fix.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()
    results = {}
    for name, whole_sentence in [("token_by_token (existing)", False), ("whole_sentence (fix)", True)]:
        print(f"\n[ws-fix] ==== {name} ====", flush=True)
        acc, rank, probe = train_one(whole_sentence, tok, args)
        results[name] = {"held_out_acc": acc, "mean_participation_ratio": rank, "fact_probe_by_query_idx": probe}
        print(f"[ws-fix] {name} FINAL held_out_acc={acc:.3f} mean_participation_ratio={rank:.3f} probe={probe}",
              flush=True)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ws-fix] wrote {args.results_file}", flush=True)

    print("\n[ws-fix] === SUMMARY ===")
    for name, r in results.items():
        print(f"{name}: held_out_acc={r['held_out_acc']:.3f} "
              f"participation_ratio={r['mean_participation_ratio']:.3f} probe={r['fact_probe_by_query_idx']}")


if __name__ == "__main__":
    main()
