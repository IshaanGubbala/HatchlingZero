#!/usr/bin/env python3
"""Combining fixes, explicit user request 2026-09-05 ("yes combine"),
after whole-sentence ingestion became the first attempted fix in this
thread to move held-out recall at all (0.245 -> 0.333). The earlier
diversity-loss and routing fixes failed for a specific, now-fixed
reason: with T_demo=1 (token-by-token ingestion), delta_S was
IDENTICAL across every slot by construction (softmax over a length-1
dimension is always exactly 1.0), so neither intervention ever had
real per-slot signal to act on. With whole-sentence ingestion
(T_demo=sentence length), that's no longer true -- cross-attention
finally produces genuinely different attn/delta_S per slot. This
re-tests both earlier ideas ON TOP OF whole-sentence ingestion, plus
the attention-diversity loss that was abandoned as mathematically dead
under token-by-token ingestion (it's real now).

Four conditions, same n_facts=3 config, same instrumentation
(held-out accuracy, S participation ratio, per-slot fact-decoding
probe, slot-choice logging for the routing variants) as every earlier
script in this thread:
  1. whole_sentence            -- the working fix alone (control, reproduces 0.333)
  2. whole_sentence+attn_div   -- + attention-diversity loss (mean squared pairwise
                                   cosine similarity across per-slot attention
                                   distributions, now meaningful with T_demo>1)
  3. whole_sentence+top1       -- + top-1 routing on the gate logit
  4. whole_sentence+top2       -- + top-2 routing on the gate logit

No changes to HZCQPersistentMemory's source -- same non-invasive
external replication of update()'s math used throughout this thread.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hz_nursery_train as nt  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from reference.hz0h_bdh_hzcq_v1_persistent_memory_torch import _rms  # noqa: E402
from hatchling_world.language.tokenizer import NurseryTokenizer, NOVEL_LABELS  # noqa: E402
from hatchling_world.language.nursery_generator import generate_l5_stress_episode  # noqa: E402


def gate_logit(mem, S_prev, delta_S):
    q = torch.cat([
        _rms(S_prev), _rms(delta_S),
        F.cosine_similarity(S_prev, delta_S, dim=-1).unsqueeze(-1),
        _rms(S_prev - delta_S),
    ], dim=-1)
    hid = F.silu(q @ mem.gate_w1 + mem.gate_b1)
    return hid @ mem.gate_w2 + mem.gate_b2


def attn_diversity_loss(attn: torch.Tensor) -> torch.Tensor:
    """attn: (B, M_S, T_demo). Meaningful now that T_demo can be >1
    (whole-sentence chunks) -- degenerate (always exactly 1.0) under
    the old token-by-token (T_demo=1) ingestion."""
    a = attn.squeeze(0)  # (M_S, T_demo)
    normed = F.normalize(a, dim=-1)
    sim = normed @ normed.T
    off_diag = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return (sim[off_diag] ** 2).mean()


def update_combined(mem, S_prev, demo_hidden, routing_k, tau):
    Q = mem.q_proj(S_prev)
    K = mem.k_proj(demo_hidden)
    V = mem.v_proj(demo_hidden)
    scale = 1.0 / math.sqrt(Q.size(-1))
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    attn = F.softmax(scores, dim=-1)  # (B, M_S, T_demo)
    read = torch.matmul(attn, V)
    delta_S = mem.ln_read(mem.write_proj(read))

    logits = gate_logit(mem, S_prev, delta_S)
    if routing_k is None:
        g = torch.sigmoid(logits)
    else:
        flat = logits.squeeze(-1)
        topk_vals, topk_idx = flat.topk(routing_k, dim=-1)
        mask = torch.full_like(flat, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        p = F.softmax(mask / tau, dim=-1)
        g = p.unsqueeze(-1)

    S_new = mem.ln_state(S_prev + g * delta_S)
    chosen_slot = int(logits.squeeze(-1).argmax(dim=-1)[0].item())
    return S_new, chosen_slot, attn


def stress_recall_forward_combined(model, tok, sequence, question, routing_k, tau, want_attn_div):
    B = 1
    S = model.mem.init_state(B)
    chosen_slots = []
    div_losses = []
    for sentence in sequence:
        ids = torch.tensor([tok.encode(sentence)])
        hidden = model.token_embed(ids)  # (1, T, D) -- whole sentence, one chunk
        S, slot, attn = update_combined(model.mem, S, hidden, routing_k, tau)
        chosen_slots.append(slot)
        if want_attn_div:
            div_losses.append(attn_diversity_loss(attn))

    q_ids = torch.tensor([tok.encode(question)])
    q_hidden = model.token_embed(q_ids)
    S, _, _ = update_combined(model.mem, S, q_hidden, routing_k, tau)

    x_null = model.read_null_x.expand(B, 1, model.D)
    H = model.ws.run(B, S, x_null, n_rounds=model.n_rounds_l1)
    q = model.qa_rq(H).mean(dim=1, keepdim=True)
    scores = torch.matmul(q, model.qa_rk(H).transpose(-1, -2)) / (model.D ** 0.5)
    read = torch.matmul(F.softmax(scores, dim=-1), H).mean(dim=1)
    logits = model.qa_head(read)
    mean_div_loss = torch.stack(div_losses).mean() if div_losses else None
    return logits, chosen_slots, S, mean_div_loss


def participation_ratio(S: torch.Tensor) -> float:
    sv = torch.linalg.svdvals(S)
    return float((sv.sum() ** 2) / (sv ** 2).sum())


class FactProbe(nn.Module):
    def __init__(self, d_model: int, n_labels: int):
        super().__init__()
        self.head = nn.Linear(d_model, n_labels)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        return self.head(S.mean(dim=1))


def train_one(name, routing_k, attn_div_lambda, tau, tok, args):
    torch.manual_seed(args.seed)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=8,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + nt.TEST_SEED_OFFSET)
    want_attn_div = attn_div_lambda > 0

    for step in range(args.steps):
        ep = generate_l5_stress_episode(train_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
        answer_idx = torch.tensor([ep["answer_idx"]])
        logits, _, _, div_loss = stress_recall_forward_combined(
            model, tok, ep["sequence"], ep["question"], routing_k, tau, want_attn_div)
        loss = F.cross_entropy(logits, answer_idx)
        if want_attn_div:
            loss = loss + attn_div_lambda * div_loss
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
                    lg, _, S, _ = stress_recall_forward_combined(
                        model, tok, e["sequence"], e["question"], routing_k, tau, False)
                    correct += int((lg.argmax(-1) == a_idx).item())
                    ranks.append(participation_ratio(S[0]))
            acc = correct / args.eval_episodes
            mean_rank = sum(ranks) / len(ranks)
            print(f"[combined][{name}] step={step+1}/{args.steps} held_out_acc={acc:.3f} "
                  f"mean_participation_ratio={mean_rank:.3f}", flush=True)

    correct = 0
    ranks = []
    slot_choice_by_fact = {1: [], 2: [], 3: []}
    with torch.no_grad():
        for _ in range(args.eval_episodes * 2):
            e = generate_l5_stress_episode(eval_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
            a_idx = torch.tensor([e["answer_idx"]])
            lg, chosen_slots, S, _ = stress_recall_forward_combined(
                model, tok, e["sequence"], e["question"], routing_k, tau, False)
            correct += int((lg.argmax(-1) == a_idx).item())
            ranks.append(participation_ratio(S[0]))
            for fact_num, slot in enumerate(chosen_slots, start=1):
                slot_choice_by_fact[fact_num].append(slot)
    final_acc = correct / (args.eval_episodes * 2)
    final_rank = sum(ranks) / len(ranks)
    slot_dist = {k: dict(Counter(v)) for k, v in slot_choice_by_fact.items()}

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
                _, _, S, _ = stress_recall_forward_combined(
                    model, tok, ep["sequence"], ep["question"], routing_k, tau, False)
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

    return final_acc, final_rank, probe_results, slot_dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-facts", type=int, default=3)
    parser.add_argument("--n-distractors", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--attn-div-lambda", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--probe-steps", type=int, default=1500)
    parser.add_argument("--probe-eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_l5_combined_fix.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()
    configs = [
        ("whole_sentence", None, 0.0),
        ("whole_sentence+attn_div", None, args.attn_div_lambda),
        ("whole_sentence+top1", 1, 0.0),
        ("whole_sentence+top2", 2, 0.0),
    ]
    results = {}
    for name, routing_k, attn_div_lambda in configs:
        print(f"\n[combined] ==== {name} ====", flush=True)
        acc, rank, probe, slot_dist = train_one(name, routing_k, attn_div_lambda, args.tau, tok, args)
        results[name] = {"held_out_acc": acc, "mean_participation_ratio": rank,
                          "fact_probe_by_query_idx": probe, "slot_distribution_by_fact": slot_dist}
        print(f"[combined] {name} FINAL held_out_acc={acc:.3f} mean_participation_ratio={rank:.3f} probe={probe}",
              flush=True)
        print(f"[combined] {name} slot choices by fact: {slot_dist}", flush=True)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[combined] wrote {args.results_file}", flush=True)

    print("\n[combined] === SUMMARY ===")
    for name, r in results.items():
        print(f"{name}: held_out_acc={r['held_out_acc']:.3f} "
              f"participation_ratio={r['mean_participation_ratio']:.3f} probe={r['fact_probe_by_query_idx']}")


if __name__ == "__main__":
    main()
