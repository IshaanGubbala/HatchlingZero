#!/usr/bin/env python3
"""Real "clean" variable-depth answer-only training, replacing Phase 3's
disqualified per-round state supervision: L = L_answer(h_R, y) only.
No state targets, no per-round heads (one SHARED answer head applied
only at the sampled terminal round R), no instruction about what
intermediate representations should contain. R sampled from {1,2,4,8}
per real training example; if recurrence genuinely helps solve harder
(more-hop) problems, A_1 < A_2 < A_4 < A_8 should emerge naturally on
the shortcut-resistant chain task (scripts/hz0h_bdh_shortcut_resistant_chain_task.py).

Real, deliberate choice per the redesigned plan: this trains PURELY on
the synthetic reasoning task (no interleaved ordinary-corpus LM loss)
-- this is a focused, dedicated probe of whether the recurrent
operator can learn to use extra depth productively, not a candidate
replacement production checkpoint. Warmstarts decoder_up/decoder_down
from the same real trained checkpoint every other experiment this
project uses, for a consistent, comparable starting point.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context
from scripts.hz0h_bdh_shortcut_resistant_chain_task import LOCATIONS, generate_chain_example
from scripts.hz0h_bdh_vb_subspace_decoder_quality_check import svd_warmstart_decoder
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize


def add_answer_head(model: BDHVBSubspaceDecoder, n_classes: int) -> None:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.answer_head = nn.Parameter(torch.zeros((model.config.n_embd, n_classes), device=device, dtype=dtype).normal_(std=0.02))


def _iteration(x: torch.Tensor, model: BDHVBSubspaceDecoder, B: int, T: int, D: int, nh: int, N: int):
    x_latent = x @ model.encoder
    x_sparse = F.relu(x_latent)
    v_bottleneck = x @ model.P
    yKV_bottleneck = model.attn(Q=x_sparse, K=x_sparse, V=v_bottleneck)
    yKV = yKV_bottleneck @ model.O
    yKV = model.ln(yKV)
    y_latent = yKV @ model.encoder_v
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse
    xy_sparse = model.drop(xy_sparse)
    alpha = torch.matmul(xy_sparse, model.decoder_up.view(nh, N, -1)).sum(dim=1, keepdim=True)
    yMLP = alpha @ model.decoder_down
    y = model.ln(yMLP)
    x = model.ln(x + y)
    return x


def forward_answer_at_round(model: BDHVBSubspaceDecoder, idx: torch.Tensor, n_rounds: int,
                             answer_label: torch.Tensor | None = None):
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)
    for _r in range(n_rounds):
        x = torch.utils.checkpoint.checkpoint(_iteration, x, model, B, T, D, nh, N, use_reentrant=False)

    last_token = x[:, :, -1, :].reshape(B, D)
    answer_logits = last_token @ model.answer_head
    loss = None
    if answer_label is not None:
        loss = F.cross_entropy(answer_logits, answer_label)
    return answer_logits, loss


def train(config, args, device):
    torch.manual_seed(args.seed)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    if args.init_checkpoint is not None:
        svd_warmstart_decoder(model, args.init_checkpoint, config.subspace_rank, device)
    add_answer_head(model, n_classes=len(LOCATIONS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))

    rng = random.Random(args.seed)
    hop_pool = [1, 2, 3, 4, 6, 8]
    r_pool = [1, 2, 4, 8]
    started = time.perf_counter()
    for step in range(args.n_examples):
        n_hops = rng.choice(hop_pool)
        n_rounds = rng.choice(r_pool)
        text, correct_idx, _shortcut_idx = generate_chain_example(rng, n_hops)
        idx = torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=device)
        label = torch.tensor([correct_idx], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args, device):
            _, loss = forward_answer_at_round(model, idx, n_rounds, label)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if args.log_every and (step + 1) % args.log_every == 0:
            now = time.perf_counter()
            rate = (step + 1) / (now - started)
            eta = (args.n_examples - step - 1) / max(rate, 1e-6)
            print(f"[var_depth] example {step+1}/{args.n_examples} hops={n_hops} R={n_rounds} "
                  f"loss={float(loss):.4f} {rate:.1f} ex/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[var_depth] DONE {args.n_examples} examples in {elapsed:.0f}s", flush=True)
    model.eval()
    return model, elapsed


@torch.no_grad()
def evaluate_matrix(model, args, device):
    rng = random.Random(args.seed + 999)
    hop_pool = [1, 2, 3, 4, 6, 8]
    r_pool = [1, 2, 4, 8, 12, 16]
    matrix = {}
    for n_hops in hop_pool:
        matrix[n_hops] = {}
        for n_rounds in r_pool:
            correct = 0
            shortcut_matches = 0
            total = args.eval_n
            for _ in range(total):
                text, correct_idx, shortcut_idx = generate_chain_example(rng, n_hops)
                idx = torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=device)
                with autocast_context(args, device):
                    logits, _ = forward_answer_at_round(model, idx, n_rounds)
                pred = int(logits.argmax(dim=-1).item())
                if pred == correct_idx:
                    correct += 1
                elif pred == shortcut_idx:
                    shortcut_matches += 1
            matrix[n_hops][n_rounds] = {"accuracy": correct / total, "shortcut_rate": shortcut_matches / total}
            print(f"  hops={n_hops} R={n_rounds}: accuracy={correct/total:.3f} "
                  f"shortcut_rate={shortcut_matches/total:.3f}", flush=True)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, default=Path("results/local/hz0h_bdh_checkpoint_for_ablation.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-examples", type=int, default=20_000)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    args = parser.parse_args()

    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )
    model, elapsed = train(config, args, device)
    print("=== R x hop-count evaluation matrix ===", flush=True)
    matrix = evaluate_matrix(model, args, device)

    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "training_seconds": elapsed,
        "matrix": matrix,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("answer_head")}
        torch.save({
            "state_dict": state_dict,
            "config": {"n_layer": config.n_layer, "n_embd": config.n_embd, "n_head": config.n_head,
                       "mlp_internal_dim_multiplier": config.mlp_internal_dim_multiplier, "vocab_size": config.vocab_size,
                       "dropout": config.dropout, "d_state": config.d_state, "subspace_rank": config.subspace_rank},
            "seed": args.seed, "elapsed_seconds": elapsed, "has_round_embed": False,
        }, args.save_checkpoint)
        print(f"[done] wrote real checkpoint to {args.save_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
