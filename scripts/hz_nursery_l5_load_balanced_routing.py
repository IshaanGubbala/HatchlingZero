#!/usr/bin/env python3
"""ONE decisive evaluation, explicit user directive 2026-09-05: does
load-balanced top-k routing beat the current best (whole-sentence
ingestion + attention-diversity loss, 0.345 held-out accuracy on
n_facts=3) -- and if not, KILL the routing idea and move to CUDA
dispatch/systems work regardless of outcome. Not a license to keep
tuning; this is the last routing experiment in this thread unless it
clears the bar.

Naive top-k routing (hz_nursery_l5_slot_routing_fix.py,
hz_nursery_l5_combined_fix.py) failed with a specific, falsifiable
pathology: top-1 routed every fact, every episode, to slot 0 -- a
rich-get-richer collapse where whichever slot starts with a marginally
higher logit gets ALL further gradient (since it's the only slot ever
selected), confirmed to be a real load-imbalance problem (the same one
documented in real mixture-of-experts literature) rather than an
artifact of insufficient content signal. The standard MoE fix is an
auxiliary LOAD-BALANCING loss (Switch Transformer style): maintain a
running (EMA) estimate of how often each slot has recently been
selected, and penalize assigning high routing probability to
already-overused slots. Adapted here to this project's one-episode-
at-a-time training (not large batched MoE dispatch): an EMA buffer
over ALL write events across ALL training steps estimates recent
per-slot usage; each write event's full (unmasked) softmax probability
is penalized proportionally to how overused that slot has been.

Pre-committed success criterion (explicit, not renegotiated after
seeing results): 3-fact held-out recall > 0.345 (the current best),
REPRODUCIBLY across 3 seeds, AND the direct-S fact-decoding probe
improves. "Routing became prettier/more uniform" is NOT sufficient on
its own -- the diversity-loss fix already proved pretty internal
geometry doesn't matter by itself.

Conditions compared, 3 seeds each, n_facts=3:
  1. whole_sentence+attn_div          -- current best, RE-RUN across
                                          seeds (only ever run once,
                                          seed 0 -- "reproducibly
                                          across seeds" requires this
                                          baseline be re-verified too)
  2. whole_sentence+attn_div+lb_top1  -- + load-balanced top-1 routing
  3. whole_sentence+attn_div+lb_top2  -- + load-balanced top-2 routing

Instrumentation per the user's explicit request: held-out accuracy,
mean participation ratio, per-fact fact-decoding probe, routing
entropy (mean entropy of the FULL softmax routing distribution --
higher = more spread across slots), slot-choice overlap between facts
(does fact A's preferred slot collide with fact B's).

If any load-balanced condition's mean accuracy clears 0.345
reproducibly, this script also runs a 1/2/3/4-fact capacity sweep on
that winning configuration to characterize the fix properly before
calling it done.
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
    a = attn.squeeze(0)
    normed = F.normalize(a, dim=-1)
    sim = normed @ normed.T
    off_diag = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return (sim[off_diag] ** 2).mean()


class EMAUsageTracker:
    """Running estimate of per-slot selection frequency across ALL
    write events, ALL episodes, ALL training steps -- the "recent load"
    signal a real batched MoE would get for free from within-batch
    statistics, adapted to this project's one-episode-at-a-time
    training loop."""

    def __init__(self, n_slots: int, decay: float = 0.99):
        self.usage = torch.full((n_slots,), 1.0 / n_slots)
        self.decay = decay

    def penalty_for(self, full_probs: torch.Tensor) -> torch.Tensor:
        """full_probs: (M_S,) this write event's full softmax routing
        distribution. Returns the load-balance loss CONTRIBUTION for
        this single write event: dot(recent_usage.detach(), full_probs)
        -- penalizes assigning probability mass to slots that have
        recently been overused, standard Switch Transformer style."""
        return (self.usage.detach() * full_probs).sum() * self.usage.numel()

    def update(self, chosen_slot: int) -> None:
        one_hot = torch.zeros_like(self.usage)
        one_hot[chosen_slot] = 1.0
        self.usage = self.decay * self.usage + (1 - self.decay) * one_hot


def update_lb_routed(mem, S_prev, demo_hidden, routing_k, tau, ema: EMAUsageTracker | None):
    """update_combined's mechanics, plus: full (unmasked) softmax
    probs returned for the load-balance loss, and the EMA usage buffer
    updated with the actual chosen slot (only relevant when ema is not
    None, i.e. a load-balanced condition)."""
    Q = mem.q_proj(S_prev)
    K = mem.k_proj(demo_hidden)
    V = mem.v_proj(demo_hidden)
    scale = 1.0 / math.sqrt(Q.size(-1))
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    attn = F.softmax(scores, dim=-1)
    read = torch.matmul(attn, V)
    delta_S = mem.ln_read(mem.write_proj(read))

    logits = gate_logit(mem, S_prev, delta_S)  # (B, M_S, 1)
    flat = logits.squeeze(-1)  # (B, M_S)
    full_probs = F.softmax(flat, dim=-1)[0]  # (M_S,) -- for load-balance loss + entropy tracking

    if routing_k is None:
        g = torch.sigmoid(logits)
    else:
        topk_vals, topk_idx = flat.topk(routing_k, dim=-1)
        mask = torch.full_like(flat, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        p = F.softmax(mask / tau, dim=-1)
        g = p.unsqueeze(-1)

    S_new = mem.ln_state(S_prev + g * delta_S)
    chosen_slot = int(flat.argmax(dim=-1)[0].item())

    lb_loss = None
    if ema is not None:
        lb_loss = ema.penalty_for(full_probs)
        ema.update(chosen_slot)

    entropy = float(-(full_probs.clamp_min(1e-9) * full_probs.clamp_min(1e-9).log()).sum().item())
    return S_new, chosen_slot, attn, lb_loss, entropy


def stress_recall_forward_lb(model, tok, sequence, question, routing_k, tau, attn_div_lambda, lb_lambda, ema):
    B = 1
    S = model.mem.init_state(B)
    chosen_slots, div_losses, lb_losses, entropies = [], [], [], []
    for sentence in sequence:
        ids = torch.tensor([tok.encode(sentence)])
        hidden = model.token_embed(ids)
        S, slot, attn, lb_loss, ent = update_lb_routed(model.mem, S, hidden, routing_k, tau, ema)
        chosen_slots.append(slot)
        entropies.append(ent)
        if attn_div_lambda > 0:
            div_losses.append(attn_diversity_loss(attn))
        if lb_loss is not None:
            lb_losses.append(lb_loss)

    q_ids = torch.tensor([tok.encode(question)])
    q_hidden = model.token_embed(q_ids)
    S, _, _, _, _ = update_lb_routed(model.mem, S, q_hidden, routing_k, tau, ema)

    x_null = model.read_null_x.expand(B, 1, model.D)
    H = model.ws.run(B, S, x_null, n_rounds=model.n_rounds_l1)
    q = model.qa_rq(H).mean(dim=1, keepdim=True)
    scores = torch.matmul(q, model.qa_rk(H).transpose(-1, -2)) / (model.D ** 0.5)
    read = torch.matmul(F.softmax(scores, dim=-1), H).mean(dim=1)
    logits = model.qa_head(read)

    mean_div = torch.stack(div_losses).mean() if div_losses else None
    mean_lb = torch.stack(lb_losses).mean() if lb_losses else None
    mean_entropy = sum(entropies) / len(entropies)
    return logits, chosen_slots, S, mean_div, mean_lb, mean_entropy


def participation_ratio(S: torch.Tensor) -> float:
    sv = torch.linalg.svdvals(S)
    return float((sv.sum() ** 2) / (sv ** 2).sum())


class FactProbe(nn.Module):
    def __init__(self, d_model: int, n_labels: int):
        super().__init__()
        self.head = nn.Linear(d_model, n_labels)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        return self.head(S.mean(dim=1))


def train_one(seed, routing_k, use_lb, attn_div_lambda, tau, tok, args, n_facts=None):
    n_facts = n_facts or args.n_facts
    torch.manual_seed(seed)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=8,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = random.Random(seed + 1)
    eval_rng = random.Random(seed + 1 + nt.TEST_SEED_OFFSET)
    ema = EMAUsageTracker(8) if use_lb else None
    lb_lambda = args.lb_lambda if use_lb else 0.0

    for step in range(args.steps):
        ep = generate_l5_stress_episode(train_rng, n_facts=n_facts, n_distractors=args.n_distractors)
        answer_idx = torch.tensor([ep["answer_idx"]])
        logits, _, _, div_loss, lb_loss, _ = stress_recall_forward_lb(
            model, tok, ep["sequence"], ep["question"], routing_k, tau, attn_div_lambda, lb_lambda, ema)
        loss = F.cross_entropy(logits, answer_idx)
        if div_loss is not None:
            loss = loss + attn_div_lambda * div_loss
        if lb_loss is not None:
            loss = loss + lb_lambda * lb_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.eval_every == 0:
            correct = 0
            with torch.no_grad():
                for _ in range(args.eval_episodes):
                    e = generate_l5_stress_episode(eval_rng, n_facts=n_facts, n_distractors=args.n_distractors)
                    a_idx = torch.tensor([e["answer_idx"]])
                    lg, _, _, _, _, _ = stress_recall_forward_lb(
                        model, tok, e["sequence"], e["question"], routing_k, tau, 0.0, 0.0, None)
                    correct += int((lg.argmax(-1) == a_idx).item())
            print(f"[lb-routing][seed={seed}] step={step+1}/{args.steps} held_out_acc={correct/args.eval_episodes:.3f}",
                  flush=True)

    correct = 0
    ranks, entropies = [], []
    slot_choice_by_fact = {i: [] for i in range(1, n_facts + 1)}
    with torch.no_grad():
        for _ in range(args.eval_episodes * 2):
            e = generate_l5_stress_episode(eval_rng, n_facts=n_facts, n_distractors=args.n_distractors)
            a_idx = torch.tensor([e["answer_idx"]])
            lg, chosen_slots, S, _, _, ent = stress_recall_forward_lb(
                model, tok, e["sequence"], e["question"], routing_k, tau, 0.0, 0.0, None)
            correct += int((lg.argmax(-1) == a_idx).item())
            ranks.append(participation_ratio(S[0]))
            entropies.append(ent)
            for fact_num, slot in enumerate(chosen_slots, start=1):
                slot_choice_by_fact[fact_num].append(slot)
    final_acc = correct / (args.eval_episodes * 2)
    final_rank = sum(ranks) / len(ranks)
    mean_entropy = sum(entropies) / len(entropies)
    slot_dist = {k: dict(Counter(v)) for k, v in slot_choice_by_fact.items()}

    for p in model.parameters():
        p.requires_grad_(False)
    probe_results = {}
    for query_idx in range(n_facts):
        probe = FactProbe(args.d_model, len(NOVEL_LABELS))
        probe_opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        probe_train_rng = random.Random(seed + 900 + query_idx)
        probe_eval_rng = random.Random(seed + 900 + query_idx + nt.TEST_SEED_OFFSET)

        def build_S_and_label(rng):
            ep = generate_l5_stress_episode(rng, n_facts=n_facts, n_distractors=0)
            with torch.no_grad():
                _, _, S, _, _, _ = stress_recall_forward_lb(
                    model, tok, ep["sequence"], ep["question"], routing_k, tau, 0.0, 0.0, None)
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

    return {"held_out_acc": final_acc, "mean_participation_ratio": final_rank, "mean_routing_entropy": mean_entropy,
            "fact_probe_by_query_idx": probe_results, "slot_distribution_by_fact": slot_dist}


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
    parser.add_argument("--lb-lambda", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--probe-steps", type=int, default=1500)
    parser.add_argument("--probe-eval-episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--success-threshold", type=float, default=0.345)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_l5_load_balanced_routing.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()
    conditions = [
        ("whole_sentence+attn_div (current best, re-verify)", None, False),
        ("whole_sentence+attn_div+lb_top1", 1, True),
        ("whole_sentence+attn_div+lb_top2", 2, True),
    ]

    results = {}
    for name, routing_k, use_lb in conditions:
        print(f"\n[lb-routing] ==== {name} ====", flush=True)
        seed_results = []
        for seed in args.seeds:
            print(f"[lb-routing] -- seed {seed} --", flush=True)
            r = train_one(seed, routing_k, use_lb, args.attn_div_lambda, args.tau, tok, args)
            seed_results.append(r)
            print(f"[lb-routing] {name} seed={seed} held_out_acc={r['held_out_acc']:.3f} "
                  f"participation_ratio={r['mean_participation_ratio']:.3f} "
                  f"routing_entropy={r['mean_routing_entropy']:.3f} probe={r['fact_probe_by_query_idx']}", flush=True)
        accs = [r["held_out_acc"] for r in seed_results]
        results[name] = {"per_seed": seed_results, "mean_acc": sum(accs) / len(accs),
                          "min_acc": min(accs), "max_acc": max(accs)}
        print(f"[lb-routing] {name} ACROSS SEEDS: mean={results[name]['mean_acc']:.3f} "
              f"min={results[name]['min_acc']:.3f} max={results[name]['max_acc']:.3f}", flush=True)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[lb-routing] wrote {args.results_file}", flush=True)

    print("\n[lb-routing] === DECISION ===")
    baseline_mean = results["whole_sentence+attn_div (current best, re-verify)"]["mean_acc"]
    print(f"baseline (re-verified) mean acc = {baseline_mean:.3f}")
    winner = None
    for name in ["whole_sentence+attn_div+lb_top1", "whole_sentence+attn_div+lb_top2"]:
        r = results[name]
        cleared = r["min_acc"] > args.success_threshold
        print(f"{name}: mean={r['mean_acc']:.3f} min={r['min_acc']:.3f} max={r['max_acc']:.3f} "
              f"-- {'CLEARS' if cleared else 'does NOT clear'} success threshold ({args.success_threshold}) "
              f"reproducibly (min-across-seeds basis)")
        if cleared and (winner is None or r["mean_acc"] > results[winner]["mean_acc"]):
            winner = name

    if winner is None:
        print("\n[lb-routing] VERDICT: load-balanced routing does NOT clear the pre-committed bar. "
              "Per instruction: kill it, do not tune further. Moving to CUDA dispatch/systems work next.")
    else:
        print(f"\n[lb-routing] VERDICT: {winner} clears the bar reproducibly. "
              f"Running the 1/2/3/4-fact capacity sweep on this winning config...")
        routing_k = 1 if "top1" in winner else 2
        sweep = {}
        for n_facts in [1, 2, 3, 4]:
            r = train_one(args.seeds[0], routing_k, True, args.attn_div_lambda, args.tau, tok, args, n_facts=n_facts)
            sweep[n_facts] = r
            print(f"[lb-routing][sweep] n_facts={n_facts} held_out_acc={r['held_out_acc']:.3f}", flush=True)
        with open(str(args.results_file).replace(".json", "_sweep.json"), "w") as f:
            json.dump(sweep, f, indent=2)
        print("\n[lb-routing][sweep] === 1/2/3/4-fact capacity curve ===")
        for n_facts, r in sweep.items():
            print(f"n_facts={n_facts}: held_out_acc={r['held_out_acc']:.3f}")


if __name__ == "__main__":
    main()
