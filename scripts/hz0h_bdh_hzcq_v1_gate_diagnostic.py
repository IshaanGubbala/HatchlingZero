#!/usr/bin/env python3
"""HZ-CQ-v1 real per-round gate-magnitude instrumentation, canonical
version of the ad-hoc script that produced the real, committed finding
in plans/newnewplan.md (2026-09-02): on the composed-permutation task
the adaptive gate g_r collapses sharply across rounds (0.823 -> 0.404
-> 0.103 -> 0.079 -> ~0.01-0.04 by round 5+), the opposite mechanistic
signature from the FSM task's gate staying near-fully-open at every
round (see hz0h_bdh_hzcq_v1_fsm_depth_r_experiment.py's built-in gate
instrumentation).

Trains a real K-way composed-PERMUTATION ICL task (not the FSM's
sequential-transition task -- permutation composition is the harder,
symbolic-lookup-style task where the gate-collapse signature was
first found), then manually replicates HZCQReasoningWorkspace.step's
internals round-by-round on a real trained model to record g_r
directly -- `run()` itself only returns the final H, so this diagnostic
intentionally does NOT call `run()`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_hzcq_v1_persistent_memory_torch import HZCQPersistentMemory, HZCQPersistentMemoryConfig
from reference.hz0h_bdh_hzcq_v1_reasoning_workspace_torch import HZCQReasoningWorkspace, HZCQReasoningWorkspaceConfig

TRAIN_DEPTHS = [1, 2, 4, 8, 16]
TRAIN_R_VALUES = [2, 4, 6, 8, 12, 16, 24]


def compose_perms(K: int, depth: int, batch: int, rng: torch.Generator):
    perms = [torch.stack([torch.randperm(K, generator=rng) for _ in range(batch)]) for _ in range(depth)]

    def apply(x: torch.Tensor) -> torch.Tensor:
        for p in perms:
            x = torch.gather(p, 1, x)
        return x

    return apply


def sample_episode(K: int, t_query: int, n_demos: int, depth: int, batch: int, rng: torch.Generator):
    apply = compose_perms(K, depth, batch, rng)
    demo_in = torch.randint(0, K, (batch, n_demos), generator=rng)
    demo_out = apply(demo_in)
    q_in = torch.randint(0, K, (batch, t_query), generator=rng)
    q_out = apply(q_in)
    return demo_in, demo_out, q_in, q_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--d-model", type=int, default=80)
    parser.add_argument("--k-symbols", type=int, default=5)
    parser.add_argument("--t-query", type=int, default=4)
    parser.add_argument("--n-demos", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=8)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=130000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--diagnostic-depth", type=int, default=16)
    parser.add_argument("--diagnostic-rounds", type=int, default=16)
    parser.add_argument("--diagnostic-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                         help="real bug fixed 2026-09-03 (same class as the FSM harness's): this script "
                              "never moved anything to CUDA, so a GPU dispatch would silently run on CPU. "
                              "Defaults to cuda when available.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    D, K = args.d_model, args.k_symbols
    symbol_embed = nn.Embedding(K, D)
    mem = HZCQPersistentMemory(HZCQPersistentMemoryConfig(n_embd=D, memory_slots=8, gate_hidden=args.gate_hidden))
    ws = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(n_embd=D, workspace_slots=args.workspace_slots, gate_hidden=args.gate_hidden))
    demo_encoder = nn.Linear(2 * D, D, bias=False)
    rq, rk, rv = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
    classifier = nn.Linear(D, K, bias=False)
    symbol_embed, mem, ws, demo_encoder, rq, rk, rv, classifier = (
        m.to(device) for m in (symbol_embed, mem, ws, demo_encoder, rq, rk, rv, classifier))
    print(f"[gate_diag] device={device}", flush=True)
    params = (list(symbol_embed.parameters()) + list(mem.parameters()) + list(ws.parameters())
              + list(demo_encoder.parameters()) + list(rq.parameters()) + list(rk.parameters())
              + list(rv.parameters()) + list(classifier.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr)

    def fwd(demo_in, demo_out, q_in, n_rounds):
        batch, n_demos = demo_in.shape
        xs, ys = symbol_embed(demo_in), symbol_embed(demo_out)
        demo_hidden_all = demo_encoder(torch.cat([xs, ys], dim=-1))
        demo_hiddens = [demo_hidden_all[:, i:i + 1, :] for i in range(n_demos)]
        S = mem.update_sequence(batch, demo_hiddens)
        x_q_embed = symbol_embed(q_in)
        H = ws.run(batch, S, x_q_embed, n_rounds=n_rounds)
        scores = torch.matmul(rq(x_q_embed), rk(H).transpose(-1, -2)) / (D ** 0.5)
        read = torch.matmul(F.softmax(scores, dim=-1), rv(H))
        return classifier(read)

    train_rng = torch.Generator().manual_seed(args.seed + 1)
    accs = []
    started = time.perf_counter()
    for step in range(args.train_steps):
        depth = TRAIN_DEPTHS[torch.randint(0, len(TRAIN_DEPTHS), (1,), generator=train_rng).item()]
        n_rounds = TRAIN_R_VALUES[torch.randint(0, len(TRAIN_R_VALUES), (1,), generator=train_rng).item()]
        demo_in, demo_out, q_in, q_out = sample_episode(K, args.t_query, args.n_demos, depth, args.batch_size, train_rng)
        demo_in, demo_out, q_in, q_out = (t.to(device) for t in (demo_in, demo_out, q_in, q_out))
        opt.zero_grad(set_to_none=True)
        logits = fwd(demo_in, demo_out, q_in, n_rounds)
        loss = F.cross_entropy(logits.reshape(-1, K), q_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        accs.append((logits.argmax(-1) == q_out).float().mean().item())
        if (step + 1) % 20000 == 0:
            recent = sum(accs[-500:]) / min(500, len(accs))
            print(f"[gate_diag] step {step+1}/{args.train_steps} recent_acc={recent:.4f} "
                  f"elapsed={time.perf_counter()-started:.0f}s", flush=True)

    print(f"[gate_diag] TRAIN DONE {time.perf_counter()-started:.0f}s", flush=True)

    # Real diagnostic: manually replicate `step`'s internals round-by-round
    # on a trained model, capturing g_r directly -- `run()` only returns
    # the final H, so it cannot be used here.
    ws.eval(); mem.eval()
    eval_rng = torch.Generator().manual_seed(999)
    demo_in, demo_out, q_in, q_out = sample_episode(K, args.t_query, args.n_demos, args.diagnostic_depth, args.diagnostic_batch, eval_rng)
    demo_in, demo_out, q_in, q_out = (t.to(device) for t in (demo_in, demo_out, q_in, q_out))
    with torch.no_grad():
        xs, ys = symbol_embed(demo_in), symbol_embed(demo_out)
        demo_hidden_all = demo_encoder(torch.cat([xs, ys], dim=-1))
        demo_hiddens = [demo_hidden_all[:, i:i + 1, :] for i in range(args.n_demos)]
        S = mem.update_sequence(args.diagnostic_batch, demo_hiddens)
        x_q_embed = symbol_embed(q_in)
        H = ws.init_state(args.diagnostic_batch)
        gate_magnitudes, h_change_norms = [], []
        for _ in range(args.diagnostic_rounds):
            H_prev = H.clone()
            read_from_s = ws.read_s(H, S)
            read_from_x = ws.read_x(H, x_q_embed)
            delta_H = ws.ln_read(ws.write_proj(torch.cat([read_from_s, read_from_x], dim=-1)))
            g = ws._gate(H, delta_H, S)
            H = ws.ln_state(H + g * delta_H)
            gate_magnitudes.append(g.mean().item())
            h_change_norms.append((H - H_prev).norm(dim=-1).mean().item())
        print(f"[gate_diag] real per-round gate magnitude (mean g, R=1..{args.diagnostic_rounds}): "
              f"{[round(x, 4) for x in gate_magnitudes]}", flush=True)
        print(f"[gate_diag] real per-round H change norm: {[round(x, 4) for x in h_change_norms]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "gate_magnitudes_per_round": gate_magnitudes,
        "h_change_norms_per_round": h_change_norms,
        "n_steps_trained": args.train_steps,
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
