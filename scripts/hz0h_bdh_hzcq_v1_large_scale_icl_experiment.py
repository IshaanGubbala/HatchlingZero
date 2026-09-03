#!/usr/bin/env python3
"""HZ-CQ-v1 real, large-scale ICL comparison: S+H vs a matched standard
Transformer baseline on the identical composed-permutation task,
canonical version of the pair of ad-hoc scripts that produced the real,
committed headline result in plans/newnewplan.md (2026-09-02): at
K=3, N_DEMOS=6, 150K steps, S+H reaches 99.90% eval accuracy with
71,762 params while a matched Transformer (151,488 params, more than
2x the parameter count) stays at exact chance (32.85% vs 33.33% chance)
for the entire run -- a real, clean quality-per-parameter win, not an
artifact of an unfairly hard task (an earlier K=5/K=6 version of this
task was information-theoretically underdetermined by its own demo
count and produced an uninterpretable near-chance result for BOTH
architectures; K=3 was the real calibration fix).

`--model sh` trains HZCQPersistentMemory + HZCQReasoningWorkspace.
`--model transformer` trains a plain nn.TransformerEncoder of matched
depth as the baseline. Run both with the same --k-symbols/--n-demos to
reproduce the comparison.
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


def to_device(*tensors, device):
    return tuple(t.to(device) for t in tensors)


def sample_episode(K: int, t_query: int, n_demos: int, batch: int, rng: torch.Generator):
    perm = torch.stack([torch.randperm(K, generator=rng) for _ in range(batch)])
    demo_in = torch.randint(0, K, (batch, n_demos), generator=rng)
    demo_out = torch.gather(perm, 1, demo_in)
    q_in = torch.randint(0, K, (batch, t_query), generator=rng)
    q_out = torch.gather(perm, 1, q_in)
    return demo_in, demo_out, q_in, q_out


class TinyICLTransformer(nn.Module):
    def __init__(self, K: int, n_demos: int, t_query: int, d_model: int, n_head: int, n_layer: int):
        super().__init__()
        self.K, self.n_demos, self.t_query = K, n_demos, t_query
        seq_len = 2 * n_demos + t_query
        self.symbol_embed = nn.Embedding(K, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
                                            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.classifier = nn.Linear(d_model, K, bias=False)

    def forward(self, demo_in, demo_out, q_in):
        batch = demo_in.shape[0]
        tokens, types = [], []
        for i in range(self.n_demos):
            tokens.append(demo_in[:, i]); types.append(0)
            tokens.append(demo_out[:, i]); types.append(1)
        for i in range(self.t_query):
            tokens.append(q_in[:, i]); types.append(0)
        tokens = torch.stack(tokens, dim=1)
        type_ids = torch.tensor(types, device=tokens.device).unsqueeze(0).expand(batch, -1)
        x = self.symbol_embed(tokens) + self.type_embed(type_ids) + self.pos_embed.unsqueeze(0)
        x = self.encoder(x)
        return self.classifier(x[:, 2 * self.n_demos:, :])


def run_sh(args, K: int, device):
    D = args.d_model
    symbol_embed = nn.Embedding(K, D)
    mem = HZCQPersistentMemory(HZCQPersistentMemoryConfig(n_embd=D, memory_slots=8, gate_hidden=16))
    ws = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(n_embd=D, workspace_slots=8, gate_hidden=16))
    demo_encoder = nn.Linear(2 * D, D, bias=False)
    rq, rk, rv = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
    classifier = nn.Linear(D, K, bias=False)
    symbol_embed, mem, ws, demo_encoder, rq, rk, rv, classifier = to_device(
        symbol_embed, mem, ws, demo_encoder, rq, rk, rv, classifier, device=device)
    params = (list(symbol_embed.parameters()) + list(mem.parameters()) + list(ws.parameters())
              + list(demo_encoder.parameters()) + list(rq.parameters()) + list(rk.parameters())
              + list(rv.parameters()) + list(classifier.parameters()))
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr)

    def fwd(demo_in, demo_out, q_in, n_rounds=8):
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

    rng = torch.Generator().manual_seed(args.seed + 1)
    accs, history = [], []
    started = time.perf_counter()
    for step in range(args.train_steps):
        demo_in, demo_out, q_in, q_out = sample_episode(K, args.t_query, args.n_demos, args.batch_size, rng)
        demo_in, demo_out, q_in, q_out = to_device(demo_in, demo_out, q_in, q_out, device=device)
        opt.zero_grad(set_to_none=True)
        logits = fwd(demo_in, demo_out, q_in)
        loss = F.cross_entropy(logits.reshape(-1, K), q_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        accs.append((logits.argmax(-1) == q_out).float().mean().item())
        if (step + 1) % 10000 == 0:
            recent = sum(accs[-2000:]) / min(2000, len(accs))
            elapsed = time.perf_counter() - started
            history.append({"step": step + 1, "acc": recent, "elapsed_s": elapsed})
            print(f"[S+H] step {step+1}/{args.train_steps} recent_acc={recent:.4f} elapsed={elapsed:.0f}s", flush=True)

    eval_rng = torch.Generator().manual_seed(999)
    with torch.no_grad():
        demo_in, demo_out, q_in, q_out = sample_episode(K, args.t_query, args.n_demos, args.eval_episodes, eval_rng)
        demo_in, demo_out, q_in, q_out = to_device(demo_in, demo_out, q_in, q_out, device=device)
        logits = fwd(demo_in, demo_out, q_in)
        eval_acc = (logits.argmax(-1) == q_out).float().mean().item()
    return eval_acc, history, n_params


def run_transformer(args, K: int, device):
    model = TinyICLTransformer(K, args.n_demos, args.t_query, args.d_model, args.n_head, args.n_layer)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed + 1)
    accs, history = [], []
    started = time.perf_counter()
    for step in range(args.train_steps):
        demo_in, demo_out, q_in, q_out = sample_episode(K, args.t_query, args.n_demos, args.batch_size, rng)
        demo_in, demo_out, q_in, q_out = to_device(demo_in, demo_out, q_in, q_out, device=device)
        opt.zero_grad(set_to_none=True)
        logits = model(demo_in, demo_out, q_in)
        loss = F.cross_entropy(logits.reshape(-1, K), q_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        accs.append((logits.argmax(-1) == q_out).float().mean().item())
        if (step + 1) % 10000 == 0:
            recent = sum(accs[-2000:]) / min(2000, len(accs))
            elapsed = time.perf_counter() - started
            history.append({"step": step + 1, "acc": recent, "elapsed_s": elapsed})
            print(f"[Transformer] step {step+1}/{args.train_steps} recent_acc={recent:.4f} elapsed={elapsed:.0f}s", flush=True)

    eval_rng = torch.Generator().manual_seed(999)
    with torch.no_grad():
        demo_in, demo_out, q_in, q_out = sample_episode(K, args.t_query, args.n_demos, args.eval_episodes, eval_rng)
        demo_in, demo_out, q_in, q_out = to_device(demo_in, demo_out, q_in, q_out, device=device)
        logits = model(demo_in, demo_out, q_in)
        eval_acc = (logits.argmax(-1) == q_out).float().mean().item()
    return eval_acc, history, n_params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("sh", "transformer"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k-symbols", type=int, default=3)
    parser.add_argument("--t-query", type=int, default=4)
    parser.add_argument("--n-demos", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4, help="transformer only")
    parser.add_argument("--n-layer", type=int, default=3, help="transformer only")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                         help="real bug fixed 2026-09-03 (same class as the FSM harness's): this script "
                              "never moved anything to CUDA, so a GPU dispatch would silently run on CPU. "
                              "Defaults to cuda when available.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    K = args.k_symbols
    print(f"=== {args.model}, {args.train_steps} steps, K={K}, N_DEMOS={args.n_demos} device={device} ===", flush=True)
    if args.model == "sh":
        eval_acc, history, n_params = run_sh(args, K, device)
        tag = "S+H"
    else:
        eval_acc, history, n_params = run_transformer(args, K, device)
        tag = "Transformer"
    print(f"[RESULT] {tag} final eval accuracy={eval_acc:.4f} (chance={1/K:.4f}) n_params={n_params}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "eval_acc": eval_acc, "chance": 1 / K, "history": history,
        "n_steps": args.train_steps, "n_params": n_params,
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
