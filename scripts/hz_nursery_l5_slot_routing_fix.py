#!/usr/bin/env python3
"""Second attempted memory-cliff fix: SLOT ROUTING, explicit user
proposal 2026-09-05, after the diversity-loss fix's clean negative
result (participation ratio 1.1 -> 7.6-8.0, held-out recall unchanged
at chance regardless). That result separated slot GEOMETRY from
selective-write CAPABILITY -- making S's basis diverse gave the write
mechanism no new pressure to actually use that diversity, because the
existing gate applies an INDEPENDENT sigmoid per slot with no
competition between slots at all. This targets that directly.

The mechanism: HZCQPersistentMemory._gate already computes a per-slot
gate LOGIT (g_logit, pre-sigmoid) from real per-slot features
(RMS(S_prev), RMS(delta_S), cos(S_prev,delta_S), RMS(S_prev-delta_S)).
The existing behavior applies sigmoid(g_logit) to EACH slot
independently -- slots never compete for who gets to store new
content. This script replaces that with a softmax over the SAME
logits, restricted to the top-k highest-scoring slots (k in {None (=
baseline, unmodified), 1, 2}) -- exactly the user's own proposed
p_j = softmax(z_j / tau) with low-temperature/top-k routing, using
z_j = g_logit_j directly rather than a new routing network (no new
parameters -- the routing SIGNAL was already there, only the
COMPETITION between slots was missing).

No changes to HZCQPersistentMemory's source: this replicates _gate's
own math externally (same non-invasive pattern as every diagnostic
before it) up to the pre-sigmoid logit, then branches into either the
original independent-sigmoid behavior (baseline) or the new top-k
softmax behavior.

Instrumentation, per the user's own request: which slot each of the 3
facts actually gets routed to (do genuinely different facts land in
different slots, or does routing collapse onto the same slot anyway?),
plus a per-slot fact-decoding probe (same methodology as
hz_nursery_l5_memory_cliff_diagnostic.py's Part 3) to check whether
facts become independently recoverable from S once routing is added.
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


def gate_logit(mem, S_prev: torch.Tensor, delta_S: torch.Tensor) -> torch.Tensor:
    """Exact replica of HZCQPersistentMemory._gate up to the PRE-SIGMOID
    logit -- the only thing routing needs that _gate doesn't expose."""
    q = torch.cat([
        _rms(S_prev), _rms(delta_S),
        F.cosine_similarity(S_prev, delta_S, dim=-1).unsqueeze(-1),
        _rms(S_prev - delta_S),
    ], dim=-1)
    hid = F.silu(q @ mem.gate_w1 + mem.gate_b1)
    return hid @ mem.gate_w2 + mem.gate_b2  # (B, M_S, 1)


def update_with_routing(mem, S_prev: torch.Tensor, demo_hidden: torch.Tensor, k: int | None, tau: float):
    """Exact replica of HZCQPersistentMemory.update, except the gate is
    either the original independent sigmoid (k=None, the unmodified
    baseline) or a top-k softmax over the same logits (k=1 or 2),
    forcing slots to COMPETE for who gets to store this content instead
    of each deciding independently. Returns (S_new, chosen_slot_idx)
    where chosen_slot_idx is the argmax slot (for instrumentation only,
    meaningful for k=1)."""
    Q = mem.q_proj(S_prev)
    K = mem.k_proj(demo_hidden)
    V = mem.v_proj(demo_hidden)
    scale = 1.0 / math.sqrt(Q.size(-1))
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    attn = F.softmax(scores, dim=-1)
    read = torch.matmul(attn, V)
    delta_S = mem.ln_read(mem.write_proj(read))

    logits = gate_logit(mem, S_prev, delta_S)  # (B, M_S, 1)
    if k is None:
        g = torch.sigmoid(logits)
    else:
        flat = logits.squeeze(-1)  # (B, M_S)
        topk_vals, topk_idx = flat.topk(k, dim=-1)
        mask = torch.full_like(flat, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        p = F.softmax(mask / tau, dim=-1)  # zero outside top-k, competes within it
        g = p.unsqueeze(-1)

    S_new = mem.ln_state(S_prev + g * delta_S)
    chosen_slot = int(logits.squeeze(-1).argmax(dim=-1)[0].item())
    return S_new, chosen_slot


def stress_recall_forward_routed(model, sequence_ids_list, question_ids, k, tau):
    B = question_ids.shape[0]
    S = model.mem.init_state(B, device=question_ids.device)
    chosen_slots = []
    for sentence_ids in sequence_ids_list:
        for t in range(sentence_ids.shape[1]):
            x = model.token_embed(sentence_ids[:, t]).unsqueeze(1)
            S, slot = update_with_routing(model.mem, S, x, k, tau)
        chosen_slots.append(slot)  # slot chosen by the LAST token of each fact sentence
    for t in range(question_ids.shape[1]):
        x = model.token_embed(question_ids[:, t]).unsqueeze(1)
        S, _ = update_with_routing(model.mem, S, x, k, tau)

    x_null = model.read_null_x.expand(B, 1, model.D)
    H = model.ws.run(B, S, x_null, n_rounds=model.n_rounds_l1)
    q = model.qa_rq(H).mean(dim=1, keepdim=True)
    scores = torch.matmul(q, model.qa_rk(H).transpose(-1, -2)) / (model.D ** 0.5)
    read = torch.matmul(F.softmax(scores, dim=-1), H).mean(dim=1)
    logits = model.qa_head(read)
    return logits, chosen_slots, S


class FactProbe(nn.Module):
    def __init__(self, d_model: int, n_labels: int):
        super().__init__()
        self.head = nn.Linear(d_model, n_labels)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        return self.head(S.mean(dim=1))


def train_one(k, tau, tok, args):
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
        logits, _, _ = stress_recall_forward_routed(model, sequence_ids_list, question_ids, k, tau)
        loss = F.cross_entropy(logits, answer_idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.eval_every == 0:
            correct = 0
            with torch.no_grad():
                for _ in range(args.eval_episodes):
                    e = generate_l5_stress_episode(eval_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
                    seq_ids = [torch.tensor([tok.encode(s)]) for s in e["sequence"]]
                    q_ids = torch.tensor([tok.encode(e["question"])])
                    a_idx = torch.tensor([e["answer_idx"]])
                    lg, _, _ = stress_recall_forward_routed(model, seq_ids, q_ids, k, tau)
                    correct += int((lg.argmax(-1) == a_idx).item())
            acc = correct / args.eval_episodes
            print(f"[routing][k={k}] step={step+1}/{args.steps} held_out_acc={acc:.3f}", flush=True)

    # Final held-out accuracy + slot-choice instrumentation
    correct = 0
    slot_choice_by_fact = {1: [], 2: [], 3: []}
    with torch.no_grad():
        for _ in range(args.eval_episodes * 2):
            e = generate_l5_stress_episode(eval_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
            seq_ids = [torch.tensor([tok.encode(s)]) for s in e["sequence"]]
            q_ids = torch.tensor([tok.encode(e["question"])])
            a_idx = torch.tensor([e["answer_idx"]])
            lg, chosen_slots, _ = stress_recall_forward_routed(model, seq_ids, q_ids, k, tau)
            correct += int((lg.argmax(-1) == a_idx).item())
            for fact_num, slot in enumerate(chosen_slots, start=1):
                slot_choice_by_fact[fact_num].append(slot)
    final_acc = correct / (args.eval_episodes * 2)

    slot_distributions = {
        fact_num: dict(Counter(slots)) for fact_num, slots in slot_choice_by_fact.items()
    }

    # Fact-decoding probe (same methodology as the cliff diagnostic's Part 3)
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
                _, _, S = stress_recall_forward_routed(
                    model, [torch.tensor([tok.encode(s)]) for s in ep["sequence"]],
                    torch.tensor([tok.encode(ep["question"])]), k, tau)
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

    return final_acc, slot_distributions, probe_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-facts", type=int, default=3)
    parser.add_argument("--n-distractors", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--probe-steps", type=int, default=1500)
    parser.add_argument("--probe-eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_l5_slot_routing_fix.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()
    configs = [("baseline (soft, unmodified)", None), ("top-1", 1), ("top-2", 2)]
    results = {}
    for name, k in configs:
        print(f"\n[routing] ==== {name} (k={k}) ====", flush=True)
        acc, slot_dist, probe = train_one(k, args.tau, tok, args)
        results[name] = {"held_out_acc": acc, "slot_distribution_by_fact": slot_dist, "fact_probe_by_query_idx": probe}
        print(f"[routing] {name} FINAL held_out_acc={acc:.3f}", flush=True)
        print(f"[routing] {name} slot choices by fact: {slot_dist}", flush=True)
        print(f"[routing] {name} fact-decoding probe: {probe}", flush=True)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[routing] wrote {args.results_file}", flush=True)

    print("\n[routing] === SUMMARY ===")
    for name, r in results.items():
        print(f"{name}: held_out_acc={r['held_out_acc']:.3f} probe={r['fact_probe_by_query_idx']}")


if __name__ == "__main__":
    main()
