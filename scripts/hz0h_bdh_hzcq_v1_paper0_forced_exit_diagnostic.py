#!/usr/bin/env python3
"""PAPER-0 — forced-exit trajectory diagnostics (paper-derived queue,
plan section "Paper-Derived Reasoning Upgrades -- Controlled Queue",
first item in the strict experiment order). Source: "Adaptive Depth in
Looped Transformers: Diagnosing Learned Halting Gates and Trajectory
Readouts" (2026, arXiv:2607.20519).

Real, no-architecture-change instrumentation of a SINGLE recurrent
trajectory: run one real forward pass to R=max_rounds on an already-
trained checkpoint (default: the confirmed M_H=32 Pareto-point
checkpoint from the capacity ablation), and decode/read out the
classifier at EVERY intermediate H_r, not just H_R. This is deliberately
different from the depth x R sweeps done earlier this project (those
rerun the model with different total R -- separate trajectories); this
diagnostic inspects intermediate states from the SAME trajectory, per
PAPER-0's explicit "do not infer trajectory quality only by rerunning
with different total R" instruction.

No training. Loads a checkpoint, runs eval-only forward passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_hzcq_v1_persistent_memory_torch import HZCQPersistentMemory, HZCQPersistentMemoryConfig
from reference.hz0h_bdh_hzcq_v1_reasoning_workspace_torch import HZCQReasoningWorkspace, HZCQReasoningWorkspaceConfig


def sample_episode(K: int, A: int, n_demos: int, depth: int, batch: int, rng: torch.Generator):
    demo_state = torch.zeros(batch, n_demos, dtype=torch.long)
    demo_symbol = torch.zeros(batch, n_demos, dtype=torch.long)
    demo_next = torch.zeros(batch, n_demos, dtype=torch.long)
    q_start = torch.zeros(batch, dtype=torch.long)
    q_seq = torch.zeros(batch, depth, dtype=torch.long)
    q_final = torch.zeros(batch, dtype=torch.long)
    all_pairs = torch.cartesian_prod(torch.arange(K), torch.arange(A))
    for b in range(batch):
        table = torch.randint(0, K, (K, A), generator=rng)
        perm = torch.randperm(K * A, generator=rng)
        pairs = all_pairs[perm][:n_demos]
        demo_state[b] = pairs[:, 0]
        demo_symbol[b] = pairs[:, 1]
        demo_next[b] = table[pairs[:, 0], pairs[:, 1]]
        start = torch.randint(0, K, (1,), generator=rng).item()
        seq = torch.randint(0, A, (depth,), generator=rng)
        q_start[b] = start
        q_seq[b] = seq
        state = start
        for t in range(depth):
            state = table[state, seq[t].item()].item()
        q_final[b] = state
    return demo_state, demo_symbol, demo_next, q_start, q_seq, q_final


def build_model(D: int, K: int, A: int, workspace_slots: int, gate_hidden: int, allow_ablation: bool):
    state_embed = nn.Embedding(K, D)
    symbol_embed = nn.Embedding(A, D)
    mem = HZCQPersistentMemory(HZCQPersistentMemoryConfig(n_embd=D, memory_slots=8, gate_hidden=gate_hidden))
    ws = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(
        n_embd=D, workspace_slots=workspace_slots, gate_hidden=gate_hidden,
        allow_ablation_slots=allow_ablation))
    demo_encoder = nn.Linear(3 * D, D, bias=False)
    query_encoder = nn.Linear(2 * D, D, bias=False)
    rq, rk, rv = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
    classifier = nn.Linear(D, K, bias=False)
    return state_embed, symbol_embed, mem, ws, demo_encoder, query_encoder, rq, rk, rv, classifier


def kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) where P is this round's distribution, Q is the
    previous round's -- 'how much did the predictive distribution move
    this round' read as divergence FROM the old belief TO the new one."""
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    return (p.exp() * (p - q)).sum(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                         default=Path("results/local/hz0h_bdh_hzcq_v1_fsm_mh32_checkpoint.pt"),
                         help="default: the confirmed M_H=32 Pareto-point checkpoint")
    parser.add_argument("--d-model", type=int, default=80)
    parser.add_argument("--k-states", type=int, default=5)
    parser.add_argument("--a-symbols", type=int, default=4)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--depth", type=int, default=16, help="query sequence depth -- the deepest, most sequential task")
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    D, K, A = args.d_model, args.k_states, args.a_symbols
    n_demos = K * A
    allow_ablation = args.workspace_slots > 8
    model = build_model(D, K, A, args.workspace_slots, args.gate_hidden, allow_ablation)
    state_embed, symbol_embed, mem, ws, demo_encoder, query_encoder, rq, rk, rv, classifier = model

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_embed.load_state_dict(ckpt["state_embed"]); symbol_embed.load_state_dict(ckpt["symbol_embed"])
    mem.load_state_dict(ckpt["mem"]); ws.load_state_dict(ckpt["ws"])
    demo_encoder.load_state_dict(ckpt["demo_encoder"]); query_encoder.load_state_dict(ckpt["query_encoder"])
    rq.load_state_dict(ckpt["rq"]); rk.load_state_dict(ckpt["rk"]); rv.load_state_dict(ckpt["rv"])
    classifier.load_state_dict(ckpt["classifier"])
    for m in model:
        m.eval()
    print(f"[paper0] loaded checkpoint {args.checkpoint}", flush=True)

    rng = torch.Generator().manual_seed(args.seed)
    demo_state, demo_symbol, demo_next, q_start, q_seq, q_final = sample_episode(
        K, A, n_demos, args.depth, args.eval_episodes, rng)

    with torch.no_grad():
        d_s, d_a, d_n = state_embed(demo_state), symbol_embed(demo_symbol), state_embed(demo_next)
        demo_hidden_all = demo_encoder(torch.cat([d_s, d_a, d_n], dim=-1))
        demo_hiddens = [demo_hidden_all[:, i:i + 1, :] for i in range(n_demos)]
        S = mem.update_sequence(args.eval_episodes, demo_hiddens)

        start_rep = state_embed(q_start).unsqueeze(1).expand(-1, args.depth, -1)
        q_tokens = query_encoder(torch.cat([start_rep, symbol_embed(q_seq)], dim=-1))
        final_q = query_encoder(torch.cat([state_embed(q_start), symbol_embed(q_seq[:, -1])], dim=-1)).unsqueeze(1)

        # ONE real trajectory, run to max_rounds, decoding at every round.
        H = ws.init_state(args.eval_episodes)
        H_prev_for_metrics = H
        per_round = []
        for r in range(1, args.max_rounds + 1):
            read_from_s = ws.read_s(H, S)
            read_from_x = ws.read_x(H, q_tokens)
            delta_H = ws.ln_read(ws.write_proj(torch.cat([read_from_s, read_from_x], dim=-1)))
            g = ws._gate(H, delta_H, S)
            H_new = ws.ln_state(H + g * delta_H)

            scores = torch.matmul(rq(final_q), rk(H_new).transpose(-1, -2)) / (D ** 0.5)
            read = torch.matmul(F.softmax(scores, dim=-1), rv(H_new)).squeeze(1)
            logits = classifier(read)  # (B, K)

            pred = logits.argmax(-1)
            accuracy = (pred == q_final).float().mean().item()
            correct_logit = logits.gather(1, q_final.unsqueeze(1)).squeeze(1)
            other_logits = logits.masked_fill(F.one_hot(q_final, K).bool(), float("-inf"))
            margin = (correct_logit - other_logits.max(dim=-1).values).mean().item()
            probs = F.softmax(logits, dim=-1)
            entropy = (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()
            cos_h = F.cosine_similarity(H_new, H, dim=-1).mean().item()
            norm_h = (H_new - H).norm(dim=-1).mean().item()

            round_metrics = {
                "round": r,
                "accuracy": accuracy,
                "correct_logit_margin": margin,
                "entropy": entropy,
                "cos_H_vs_prev": cos_h,
                "norm_H_change": norm_h,
                "gate_magnitude": g.mean().item(),
            }
            if r > 1:
                round_metrics["kl_vs_prev_round"] = kl_div(logits, per_round[-1]["_logits"]).mean().item()
            round_metrics["_logits"] = logits  # kept for next round's KL, stripped before writing out
            per_round.append(round_metrics)
            H = H_new

    for rm in per_round:
        print(f"[paper0] round={rm['round']:2d} acc={rm['accuracy']:.4f} "
              f"margin={rm['correct_logit_margin']:+.4f} entropy={rm['entropy']:.4f} "
              f"cos(H,H_prev)={rm['cos_H_vs_prev']:.4f} ||dH||={rm['norm_H_change']:.4f} "
              f"gate={rm['gate_magnitude']:.4f}"
              + (f" KL={rm['kl_vs_prev_round']:.5f}" if "kl_vs_prev_round" in rm else ""), flush=True)

    clean = [{k: v for k, v in rm.items() if k != "_logits"} for rm in per_round]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "per_round": clean,
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
