#!/usr/bin/env python3
"""PAPER-1 — attractor/convergence diagnostic (paper-derived queue,
second item in the strict experiment order). Source: "Equilibrium
Reasoners: Learning Attractors Enables Scalable Reasoning" (2026,
arXiv:2605.21488).

Real, no-architecture-change instrumentation, direct follow-up to
PAPER-0's finding (readout predictions freeze by round ~5-6 while raw
H keeps moving substantially, ||dH||~1.93, not shrinking). Question
here: is there a real attractor in the READOUT's effective output
space even though raw H-space keeps drifting? Tests this by running
several small perturbations of H_0 through the SAME trajectory (same
S, same query, same weights) and tracking whether perturbed
trajectories converge toward each other in H-space and/or in
prediction space, and whether that convergence differs between
episodes the (unperturbed) model gets right vs wrong.

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


def run_trajectory(ws, rq, rk, rv, classifier, H0, S, q_tokens, final_q, max_rounds, D):
    """One real trajectory from a given H0. Returns H_per_round (list of
    (B,M_H,D)) and logits_per_round (list of (B,K))."""
    H = H0
    Hs, logits_seq = [], []
    for _ in range(max_rounds):
        read_from_s = ws.read_s(H, S)
        read_from_x = ws.read_x(H, q_tokens)
        delta_H = ws.ln_read(ws.write_proj(torch.cat([read_from_s, read_from_x], dim=-1)))
        g = ws._gate(H, delta_H, S)
        H = ws.ln_state(H + g * delta_H)
        scores = torch.matmul(rq(final_q), rk(H).transpose(-1, -2)) / (D ** 0.5)
        read = torch.matmul(F.softmax(scores, dim=-1), rv(H)).squeeze(1)
        logits_seq.append(classifier(read))
        Hs.append(H)
    return Hs, logits_seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                         default=Path("results/local/hz0h_bdh_hzcq_v1_fsm_mh32_checkpoint.pt"))
    parser.add_argument("--d-model", type=int, default=80)
    parser.add_argument("--k-states", type=int, default=5)
    parser.add_argument("--a-symbols", type=int, default=4)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--n-episodes", type=int, default=60)
    parser.add_argument("--n-perturbations", type=int, default=6)
    parser.add_argument("--perturbation-scale", type=float, default=0.1,
                         help="std of the Gaussian noise added to H_0, as a fraction of H_init's own std")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                         help="real bug fixed 2026-09-03 (same class as the FSM harness's): defaults to "
                              "cuda when available, though this diagnostic is cheap enough that CPU is "
                              "usually fine.")
    args = parser.parse_args()

    device = torch.device(args.device)
    D, K, A = args.d_model, args.k_states, args.a_symbols
    n_demos = K * A
    allow_ablation = args.workspace_slots > 8
    model = build_model(D, K, A, args.workspace_slots, args.gate_hidden, allow_ablation)
    model = tuple(m.to(device) for m in model)
    state_embed, symbol_embed, mem, ws, demo_encoder, query_encoder, rq, rk, rv, classifier = model

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_embed.load_state_dict(ckpt["state_embed"]); symbol_embed.load_state_dict(ckpt["symbol_embed"])
    mem.load_state_dict(ckpt["mem"]); ws.load_state_dict(ckpt["ws"])
    demo_encoder.load_state_dict(ckpt["demo_encoder"]); query_encoder.load_state_dict(ckpt["query_encoder"])
    rq.load_state_dict(ckpt["rq"]); rk.load_state_dict(ckpt["rk"]); rv.load_state_dict(ckpt["rv"])
    classifier.load_state_dict(ckpt["classifier"])
    for m in model:
        m.eval()
    print(f"[paper1] loaded checkpoint {args.checkpoint} device={device}", flush=True)

    rng = torch.Generator().manual_seed(args.seed)
    P = args.n_perturbations
    noise_std = args.perturbation_scale * ws.H_init.detach().std().item()
    print(f"[paper1] perturbation noise std={noise_std:.5f} (H_init std={ws.H_init.detach().std().item():.5f})", flush=True)

    per_round_h_dist = [[] for _ in range(args.max_rounds)]
    per_round_pred_agreement = [[] for _ in range(args.max_rounds)]
    per_episode_correct = []

    with torch.no_grad():
        for ep in range(args.n_episodes):
            demo_state, demo_symbol, demo_next, q_start, q_seq, q_final = sample_episode(
                K, A, n_demos, args.depth, 1, rng)
            demo_state, demo_symbol, demo_next, q_start, q_seq, q_final = (
                t.to(device) for t in (demo_state, demo_symbol, demo_next, q_start, q_seq, q_final))
            d_s, d_a, d_n = state_embed(demo_state), symbol_embed(demo_symbol), state_embed(demo_next)
            demo_hidden_all = demo_encoder(torch.cat([d_s, d_a, d_n], dim=-1))
            demo_hiddens = [demo_hidden_all[:, i:i + 1, :] for i in range(n_demos)]
            S = mem.update_sequence(1, demo_hiddens)
            start_rep = state_embed(q_start).unsqueeze(1).expand(-1, args.depth, -1)
            q_tokens = query_encoder(torch.cat([start_rep, symbol_embed(q_seq)], dim=-1))
            final_q = query_encoder(torch.cat([state_embed(q_start), symbol_embed(q_seq[:, -1])], dim=-1)).unsqueeze(1)

            # broadcast this one episode's S/q_tokens/final_q across P perturbed starts
            S_b = S.expand(P, -1, -1)
            q_tokens_b = q_tokens.expand(P, -1, -1)
            final_q_b = final_q.expand(P, -1, -1)
            H0_base = ws.init_state(1).expand(P, -1, -1)
            noise = (torch.randn(P, *H0_base.shape[1:], generator=rng) * noise_std).to(device)
            noise[0].zero_()  # perturbation 0 = the real unperturbed start, kept as reference
            H0 = H0_base + noise

            Hs, logits_seq = run_trajectory(ws, rq, rk, rv, classifier, H0, S_b, q_tokens_b, final_q_b, args.max_rounds, D)

            unperturbed_pred = logits_seq[-1][0].argmax(-1).item()
            per_episode_correct.append(unperturbed_pred == q_final.item())

            for r in range(args.max_rounds):
                H_r = Hs[r]  # (P, M_H, D)
                # mean pairwise L2 distance across all P*(P-1)/2 perturbation pairs
                diffs = H_r.unsqueeze(0) - H_r.unsqueeze(1)  # (P,P,M_H,D)
                pair_dist = diffs.norm(dim=-1).mean(dim=-1)  # (P,P) mean over slots
                iu = torch.triu_indices(P, P, offset=1, device=pair_dist.device)
                mean_pair_dist = pair_dist[iu[0], iu[1]].mean().item()
                per_round_h_dist[r].append(mean_pair_dist)

                preds_r = logits_seq[r].argmax(-1)  # (P,)
                agreement = (preds_r == preds_r[0]).float().mean().item()  # fraction agreeing with perturbation 0
                per_round_pred_agreement[r].append(agreement)

    def mean(xs):
        return sum(xs) / len(xs)

    overall = []
    for r in range(args.max_rounds):
        overall.append({
            "round": r + 1,
            "mean_pairwise_H_distance": mean(per_round_h_dist[r]),
            "mean_prediction_agreement": mean(per_round_pred_agreement[r]),
        })
        print(f"[paper1] round={r+1:2d} mean_pairwise_H_dist={overall[-1]['mean_pairwise_H_distance']:.4f} "
              f"pred_agreement={overall[-1]['mean_prediction_agreement']:.4f}", flush=True)

    n_correct = sum(per_episode_correct)
    correct_idx = [i for i, c in enumerate(per_episode_correct) if c]
    wrong_idx = [i for i, c in enumerate(per_episode_correct) if not c]

    def group_stats(idxs, label):
        if not idxs:
            print(f"[paper1] {label}: no episodes", flush=True)
            return None
        g = []
        for r in range(args.max_rounds):
            vals_h = [per_round_h_dist[r][i] for i in idxs]
            vals_p = [per_round_pred_agreement[r][i] for i in idxs]
            g.append({"round": r + 1, "mean_pairwise_H_distance": mean(vals_h), "mean_prediction_agreement": mean(vals_p)})
        print(f"[paper1] {label} (n={len(idxs)}): round1 H_dist={g[0]['mean_pairwise_H_distance']:.4f} "
              f"-> round{args.max_rounds} H_dist={g[-1]['mean_pairwise_H_distance']:.4f}, "
              f"round1 agreement={g[0]['mean_prediction_agreement']:.4f} -> "
              f"round{args.max_rounds} agreement={g[-1]['mean_prediction_agreement']:.4f}", flush=True)
        return g

    print(f"[paper1] unperturbed accuracy over {args.n_episodes} episodes: {n_correct/args.n_episodes:.4f}", flush=True)
    correct_group = group_stats(correct_idx, "CORRECT episodes")
    wrong_group = group_stats(wrong_idx, "WRONG episodes")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "overall_per_round": overall,
        "unperturbed_accuracy": n_correct / args.n_episodes,
        "correct_group_per_round": correct_group,
        "wrong_group_per_round": wrong_group,
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
