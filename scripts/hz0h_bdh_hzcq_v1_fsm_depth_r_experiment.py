#!/usr/bin/env python3
"""HZ-CQ-v1 real experiment harness: a genuinely-sequential finite-
state-machine task, real paired depth x R evaluation, real per-round
gate-magnitude instrumentation, and a real, opt-in M_H capacity
ablation -- the canonical, reusable version of the ad-hoc scripts used
to reach the real, committed findings in plans/newnewplan.md
(2026-09-02/03): R does not improve accuracy on this task at any
tested M_H, but M_H itself does (confirmed +3.04pp, M_H=32 vs M_H=8,
n=2000/cell, mainline plan section 8.5).

Real task: demos show the FULL transition table of a random K-state,
A-symbol finite state machine (N_DEMOS = K*A, deterministic full
coverage -- an earlier random-with-replacement version left ~40% of
the table undemonstrated and produced an uninterpretable result, see
plans/newnewplan.md's "confounded by demo coverage" section). The
query gives a start state and a real symbol SEQUENCE of length
`depth`; the true answer is the state after sequentially applying every
transition -- genuinely incremental, not a precomputable single lookup
the way the composed-permutation task
(hz0h_bdh_hzcq_v1_composition_depth_experiment.py) turned out to be.
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
EVAL_DEPTHS = [1, 2, 4, 8, 16]
EVAL_R_VALUES = [1, 2, 4, 6, 8, 12, 16, 24]


def build_random_fsm(K: int, A: int, rng: torch.Generator) -> torch.Tensor:
    return torch.randint(0, K, (K, A), generator=rng)


def sample_episode(K: int, A: int, n_demos: int, depth: int, batch: int, rng: torch.Generator):
    demo_state = torch.zeros(batch, n_demos, dtype=torch.long)
    demo_symbol = torch.zeros(batch, n_demos, dtype=torch.long)
    demo_next = torch.zeros(batch, n_demos, dtype=torch.long)
    q_start = torch.zeros(batch, dtype=torch.long)
    q_seq = torch.zeros(batch, depth, dtype=torch.long)
    q_final = torch.zeros(batch, dtype=torch.long)
    all_pairs = torch.cartesian_prod(torch.arange(K), torch.arange(A))
    for b in range(batch):
        table = build_random_fsm(K, A, rng)
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


def to_device(*tensors, device):
    return tuple(t.to(device) for t in tensors)


def build_model(D: int, K: int, A: int, workspace_slots: int, gate_hidden: int, allow_ablation: bool,
                 identity_biased: bool = False, layerscale_init: float = 0.1,
                 bounded_residual: bool = False, bound_scale: float = 1.0,
                 bounded_accumulating: bool = False, beta: float = 0.1):
    state_embed = nn.Embedding(K, D)
    symbol_embed = nn.Embedding(A, D)
    mem = HZCQPersistentMemory(HZCQPersistentMemoryConfig(n_embd=D, memory_slots=8, gate_hidden=gate_hidden))
    ws = HZCQReasoningWorkspace(HZCQReasoningWorkspaceConfig(
        n_embd=D, workspace_slots=workspace_slots, gate_hidden=gate_hidden,
        allow_ablation_slots=allow_ablation, identity_biased=identity_biased, layerscale_init=layerscale_init,
        bounded_residual=bounded_residual, bound_scale=bound_scale,
        bounded_accumulating=bounded_accumulating, beta=beta))
    demo_encoder = nn.Linear(3 * D, D, bias=False)
    query_encoder = nn.Linear(2 * D, D, bias=False)
    rq, rk, rv = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
    classifier = nn.Linear(D, K, bias=False)
    return state_embed, symbol_embed, mem, ws, demo_encoder, query_encoder, rq, rk, rv, classifier


def forward_episode(model, demo_state, demo_symbol, demo_next, q_start, q_seq, n_rounds, D, n_demos):
    state_embed, symbol_embed, mem, ws, demo_encoder, query_encoder, rq, rk, rv, classifier = model
    batch = demo_state.shape[0]
    d_s, d_a, d_n = state_embed(demo_state), symbol_embed(demo_symbol), state_embed(demo_next)
    demo_hidden_all = demo_encoder(torch.cat([d_s, d_a, d_n], dim=-1))
    demo_hiddens = [demo_hidden_all[:, i:i + 1, :] for i in range(n_demos)]
    S = mem.update_sequence(batch, demo_hiddens)
    depth = q_seq.shape[1]
    start_rep = state_embed(q_start).unsqueeze(1).expand(-1, depth, -1)
    seq_embed = symbol_embed(q_seq)
    q_tokens = query_encoder(torch.cat([start_rep, seq_embed], dim=-1))
    H = ws.run(batch, S, q_tokens, n_rounds=n_rounds)
    final_q = query_encoder(torch.cat([state_embed(q_start), symbol_embed(q_seq[:, -1])], dim=-1)).unsqueeze(1)
    scores = torch.matmul(rq(final_q), rk(H).transpose(-1, -2)) / (D ** 0.5)
    read = torch.matmul(F.softmax(scores, dim=-1), rv(H)).squeeze(1)
    return classifier(read)


def gate_magnitude_by_depth(model, K, A, n_demos, D, depths, n_rounds, batch=8, seed=555, device=None):
    state_embed, symbol_embed, mem, ws, demo_encoder, query_encoder, rq, rk, rv, classifier = model
    ws.eval(); mem.eval()
    rng = torch.Generator().manual_seed(seed)
    out = {}
    with torch.no_grad():
        for depth in depths:
            demo_state, demo_symbol, demo_next, q_start, q_seq, _ = sample_episode(K, A, n_demos, depth, batch, rng)
            if device is not None:
                demo_state, demo_symbol, demo_next, q_start, q_seq = (
                    demo_state.to(device), demo_symbol.to(device), demo_next.to(device),
                    q_start.to(device), q_seq.to(device))
            d_s, d_a, d_n = state_embed(demo_state), symbol_embed(demo_symbol), state_embed(demo_next)
            demo_hidden_all = demo_encoder(torch.cat([d_s, d_a, d_n], dim=-1))
            demo_hiddens = [demo_hidden_all[:, i:i + 1, :] for i in range(n_demos)]
            S = mem.update_sequence(batch, demo_hiddens)
            start_rep = state_embed(q_start).unsqueeze(1).expand(-1, depth, -1)
            q_tokens = query_encoder(torch.cat([start_rep, symbol_embed(q_seq)], dim=-1))
            H = ws.init_state(batch)
            mags = []
            for _ in range(n_rounds):
                read_s = ws.read_s(H, S)
                read_x = ws.read_x(H, q_tokens)
                delta_H = ws.ln_read(ws.write_proj(torch.cat([read_s, read_x], dim=-1)))
                g = ws._gate(H, delta_H, S)
                # Real bug fixed 2026-09-03: this used to hardcode
                # LN(H+g*delta_H) regardless of config, so identity_biased/
                # bounded_residual models had their gate evaluated on an
                # H-trajectory that didn't match what they were actually
                # trained on. Route through the real _apply_update instead.
                H_base = ws._compute_h_base(batch, S=S, device=H.device, dtype=H.dtype) \
                    if ws.config.bounded_residual else None
                H = ws._apply_update(H, g, delta_H, H_base)
                mags.append(g.mean().item())
            out[depth] = mags
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--d-model", type=int, default=80)
    parser.add_argument("--k-states", type=int, default=5)
    parser.add_argument("--a-symbols", type=int, default=4)
    parser.add_argument("--workspace-slots", type=int, default=8,
                         help="M_H. Plan section 6.2 locks this to 4 or 8; pass a "
                              "larger power of 2 with --allow-ablation-slots for the "
                              "real capacity ablation (section 8.5's confirmed +3.04pp finding).")
    parser.add_argument("--allow-ablation-slots", action="store_true")
    parser.add_argument("--identity-biased", action="store_true",
                         help="PAPER-2 ablation: H_{r+1}=H_r+alpha*g_r*DeltaH_r (no post-update "
                              "LayerNorm) instead of the default H_{r+1}=LN(H_r+g_r*DeltaH_r). "
                              "Everything else (read_s/read_x/gate/M_H/readout) is unchanged.")
    parser.add_argument("--layerscale-init", type=float, default=0.1,
                         help="identity_biased only: initial value of the learned alpha scalar")
    parser.add_argument("--bounded-residual", action="store_true",
                         help="PAPER-3 ablation: H_{r+1}=H_base+g_r*bound_scale*tanh(DeltaH_r), where "
                              "H_base is a FIXED evidence-conditioned anchor (H_init cross-attended once "
                              "against S) instead of the previous round's H_r -- re-anchors every round "
                              "and hard-caps the correction magnitude, unlike PAPER-2's unbounded alpha*DeltaH. "
                              "Mutually exclusive with --identity-biased.")
    parser.add_argument("--bound-scale", type=float, default=1.0,
                         help="bounded_residual only: hard cap on tanh-squashed correction magnitude")
    parser.add_argument("--bounded-accumulating", action="store_true",
                         help="PAPER-3b ablation: H_{r+1}=H_r+beta*tanh(g_r*DeltaH_r) -- real "
                              "accumulation off H_r (like PAPER-2, unlike PAPER-3) PLUS a hard tanh "
                              "bound (like PAPER-3, unlike PAPER-2), beta FIXED not learned. Mutually "
                              "exclusive with --identity-biased and --bounded-residual.")
    parser.add_argument("--beta", type=float, default=0.1,
                         help="bounded_accumulating only: fixed (not learned) update-scale hyperparameter")
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=2000,
                         help="Real, important: 300 was ambiguous (noise floor "
                              "~2.7pp, same order as several observed effects). "
                              "2000 was what actually resolved every real ambiguity "
                              "this session hit -- do not go below it without a "
                              "real reason.")
    parser.add_argument("--continue-from-checkpoint", type=Path, default=None)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                         help="real bug fixed 2026-09-03: this script previously never moved anything "
                              "to CUDA, so a RunPod GPU dispatch silently ran on CPU the whole time "
                              "(0% GPU utilization observed). Defaults to cuda when available.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    D, K, A = args.d_model, args.k_states, args.a_symbols
    n_demos = K * A
    model = build_model(D, K, A, args.workspace_slots, args.gate_hidden, args.allow_ablation_slots,
                         identity_biased=args.identity_biased, layerscale_init=args.layerscale_init,
                         bounded_residual=args.bounded_residual, bound_scale=args.bound_scale,
                         bounded_accumulating=args.bounded_accumulating, beta=args.beta)
    model = tuple(m.to(device) for m in model)
    state_embed, symbol_embed, mem, ws, demo_encoder, query_encoder, rq, rk, rv, classifier = model
    print(f"[fsm_v1] device={device}", flush=True)

    if args.continue_from_checkpoint is not None:
        ckpt = torch.load(args.continue_from_checkpoint, map_location=device)
        state_embed.load_state_dict(ckpt["state_embed"]); symbol_embed.load_state_dict(ckpt["symbol_embed"])
        mem.load_state_dict(ckpt["mem"]); ws.load_state_dict(ckpt["ws"])
        demo_encoder.load_state_dict(ckpt["demo_encoder"]); query_encoder.load_state_dict(ckpt["query_encoder"])
        rq.load_state_dict(ckpt["rq"]); rk.load_state_dict(ckpt["rk"]); rv.load_state_dict(ckpt["rv"])
        classifier.load_state_dict(ckpt["classifier"])
        print(f"[fsm_v1] loaded checkpoint {args.continue_from_checkpoint}", flush=True)

    params = (list(state_embed.parameters()) + list(symbol_embed.parameters()) + list(mem.parameters())
              + list(ws.parameters()) + list(demo_encoder.parameters()) + list(query_encoder.parameters())
              + list(rq.parameters()) + list(rk.parameters()) + list(rv.parameters()) + list(classifier.parameters()))
    n_params = sum(p.numel() for p in params)
    print(f"[fsm_v1] params={n_params} K={K} A={A} M_H={args.workspace_slots} n_demos={n_demos} "
          f"(full coverage) identity_biased={args.identity_biased} "
          f"layerscale_init={args.layerscale_init if args.identity_biased else 'n/a'} "
          f"bounded_residual={args.bounded_residual} "
          f"bound_scale={args.bound_scale if args.bounded_residual else 'n/a'} "
          f"bounded_accumulating={args.bounded_accumulating} "
          f"beta={args.beta if args.bounded_accumulating else 'n/a'}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr)

    train_rng = torch.Generator().manual_seed(args.seed + 1)
    accs_by_depth = {d: [] for d in TRAIN_DEPTHS}
    started = time.perf_counter()
    for step in range(args.train_steps):
        depth = TRAIN_DEPTHS[torch.randint(0, len(TRAIN_DEPTHS), (1,), generator=train_rng).item()]
        n_rounds = TRAIN_R_VALUES[torch.randint(0, len(TRAIN_R_VALUES), (1,), generator=train_rng).item()]
        demo_state, demo_symbol, demo_next, q_start, q_seq, q_final = sample_episode(K, A, n_demos, depth, args.batch_size, train_rng)
        demo_state, demo_symbol, demo_next, q_start, q_seq, q_final = to_device(
            demo_state, demo_symbol, demo_next, q_start, q_seq, q_final, device=device)
        opt.zero_grad(set_to_none=True)
        logits = forward_episode(model, demo_state, demo_symbol, demo_next, q_start, q_seq, n_rounds, D, n_demos)
        loss = F.cross_entropy(logits, q_final)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        accs_by_depth[depth].append((logits.argmax(-1) == q_final).float().mean().item())
        if (step + 1) % 15000 == 0:
            summary = {d: round(sum(v[-300:]) / len(v[-300:]), 3) if v else None for d, v in accs_by_depth.items()}
            print(f"[fsm_v1] step {step+1}/{args.train_steps} elapsed={time.perf_counter()-started:.0f}s acc_by_depth={summary}", flush=True)

    print(f"[fsm_v1] TRAIN DONE {time.perf_counter()-started:.0f}s", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_embed": state_embed.state_dict(), "symbol_embed": symbol_embed.state_dict(),
            "mem": mem.state_dict(), "ws": ws.state_dict(),
            "demo_encoder": demo_encoder.state_dict(), "query_encoder": query_encoder.state_dict(),
            "rq": rq.state_dict(), "rk": rk.state_dict(), "rv": rv.state_dict(), "classifier": classifier.state_dict(),
        }, args.save_checkpoint)
        print(f"[fsm_v1] saved checkpoint to {args.save_checkpoint}", flush=True)

    eval_rng = torch.Generator().manual_seed(999)
    results = {}
    with torch.no_grad():
        for depth in EVAL_DEPTHS:
            results[depth] = {}
            for r in EVAL_R_VALUES:
                demo_state, demo_symbol, demo_next, q_start, q_seq, q_final = sample_episode(
                    K, A, n_demos, depth, args.eval_episodes, eval_rng)
                demo_state, demo_symbol, demo_next, q_start, q_seq, q_final = to_device(
                    demo_state, demo_symbol, demo_next, q_start, q_seq, q_final, device=device)
                logits = forward_episode(model, demo_state, demo_symbol, demo_next, q_start, q_seq, r, D, n_demos)
                acc = (logits.argmax(-1) == q_final).float().mean().item()
                results[depth][r] = acc
                print(f"[fsm_v1] eval depth={depth} R={r} accuracy={acc:.4f}", flush=True)

    gate_by_depth = gate_magnitude_by_depth(model, K, A, n_demos, D, EVAL_DEPTHS, n_rounds=16, device=device)
    for depth, mags in gate_by_depth.items():
        print(f"[fsm_v1_gate] depth={depth} gate_magnitude_per_round: {[round(x,4) for x in mags]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "results": {str(d): {str(r): v for r, v in rd.items()} for d, rd in results.items()},
        "gate_by_depth": {str(d): v for d, v in gate_by_depth.items()},
        "n_params": n_params,
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
