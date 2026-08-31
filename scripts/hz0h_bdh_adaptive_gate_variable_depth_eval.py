#!/usr/bin/env python3
"""R-stability test, priority-sequence step 2 (plans/newnewplan.md):
does the real state-dependent adaptive gate (val_loss=1.3879, locked in
as the new 8/8 baseline) fix the old late-depth collapse, or is it "just"
an LM-loss win with the same R=12/16 breakdown as the plain compound
model showed?

Direct methodological parallel to scripts/hz0h_bdh_variable_depth_answer_train.py
(the original R-scaling result this compares against: accuracy peaked
around R=2-4 and DECLINED by R=8, with R=12/16 collapsing toward
chance) -- same shortcut-resistant chain task, same R pool during
training ({1,2,4,8}), same eval matrix ({1,2,4,8,12,16}), same
whole-model real gradient descent (not a frozen probe -- this tests
whether the RECURRENT OPERATOR, trained on this problem, learns to use
extra depth, a stronger test than probing a frozen representation).

The one real difference: this loads the LOCKED adaptive-gate checkpoint
(reference/hz0h_bdh_adaptive_gate_torch.py's forward, gate params
included) instead of doing a fresh SVD-only warmstart from the plain
compound checkpoint, and every round uses the adaptive gate's
controlled residual write instead of the plain unconditional add.
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

from reference.hz0h_bdh_adaptive_gate_torch import add_adaptive_gate, _refresh_iteration
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context
from scripts.hz0h_bdh_shortcut_resistant_chain_task import LOCATIONS, generate_chain_example
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize


def add_answer_head(model: BDHVBSubspaceDecoder, n_classes: int) -> None:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.answer_head = nn.Parameter(torch.zeros((model.config.n_embd, n_classes), device=device, dtype=dtype).normal_(std=0.02))


def load_adaptive_gate_checkpoint(config: BDHVBSubspaceDecoderConfig, checkpoint_path: Path, gate_hidden: int, device) -> BDHVBSubspaceDecoder:
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    add_adaptive_gate(model, hidden=gate_hidden, g_init=0.58, state_independent=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    real_missing = [k for k in missing if not k.startswith("answer_head")]
    assert not real_missing, f"real missing keys loading adaptive-gate checkpoint: {real_missing}"
    assert not unexpected, f"unexpected keys loading adaptive-gate checkpoint: {unexpected}"
    print(f"[load] loaded {checkpoint_path}, missing (expected, no answer_head yet)={missing}, unexpected={unexpected}", flush=True)
    return model


def forward_answer_at_round(model: BDHVBSubspaceDecoder, idx: torch.Tensor, n_rounds: int,
                             answer_label: torch.Tensor | None = None):
    """Every round is a real exact-address refresh (matches how this
    checkpoint was trained: n_refresh==n_iterations, always fresh) --
    _refresh_iteration already does checkpoint()-wrapped gradient
    checkpointing internally via bdh_adaptive_gate_forward_checkpointed's
    convention; reused directly here rather than reimplemented."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)
    h_prev = x
    for _r in range(n_rounds):
        x_new, _e, _g = torch.utils.checkpoint.checkpoint(_refresh_iteration, x, h_prev, model, B, T, D, nh, N, use_reentrant=False)
        h_prev = x
        x = x_new

    last_token = x[:, :, -1, :].reshape(B, D)
    answer_logits = last_token @ model.answer_head
    loss = None
    if answer_label is not None:
        loss = F.cross_entropy(answer_logits, answer_label)
    return answer_logits, loss


def train(config, args, device):
    torch.manual_seed(args.seed)
    model = load_adaptive_gate_checkpoint(config, args.init_checkpoint, args.gate_hidden, device)
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
            print(f"[var_depth_gate] example {step+1}/{args.n_examples} hops={n_hops} R={n_rounds} "
                  f"loss={float(loss):.4f} {rate:.1f} ex/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[var_depth_gate] DONE {args.n_examples} examples in {elapsed:.0f}s", flush=True)
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
    parser.add_argument("--init-checkpoint", type=Path, default=Path("results/local/hz0h_bdh_adaptive_gate_retrain_checkpoint.pt"))
    parser.add_argument("--out", type=Path, required=True)
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
    parser.add_argument("--gate-hidden", type=int, default=16)
    args = parser.parse_args()

    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )
    model, elapsed = train(config, args, device)
    print("=== R x hop-count evaluation matrix (adaptive-gate backbone) ===", flush=True)
    matrix = evaluate_matrix(model, args, device)

    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "training_seconds": elapsed,
        "matrix": matrix,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
