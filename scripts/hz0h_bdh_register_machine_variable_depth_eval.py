#!/usr/bin/env python3
"""Shared R x step-count eval matrix for the progressive-latentization
falsification experiment (plans/newnewplan.md, 2026-08-31) -- applied
identically to Arms A/B/C/D's resulting checkpoints (and the plain
locked baseline) so all four are compared on the exact same instrument.
Uses the SHORTCUT-RESISTANT compact register-machine task (decoy
variables, shuffled clause order) -- deliberately harder/more
adversarial than any arm's own training data, since none of A/B/C/D
train on this exact format; this measures real transfer, not
memorization of the eval format itself.

Same protocol as hz0h_bdh_adaptive_gate_variable_depth_eval.py (which
this mirrors for the entity-chain task): fit a fresh answer_head via
real whole-model gradient descent (R sampled from {1,2,4,8} during
training), then evaluate the full R x step-count matrix at
R in {1,2,4,8,12,16}.
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
from scripts.hz0h_bdh_register_machine_task import generate_register_machine_example
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize


def add_answer_head(model: BDHVBSubspaceDecoder, n_classes: int = 10) -> None:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.answer_head = nn.Parameter(torch.zeros((model.config.n_embd, n_classes), device=device, dtype=dtype).normal_(std=0.02))


def load_adaptive_gate_checkpoint(config: BDHVBSubspaceDecoderConfig, checkpoint_path: Path, gate_hidden: int, device) -> BDHVBSubspaceDecoder:
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    add_adaptive_gate(model, hidden=gate_hidden, g_init=0.58, state_independent=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    real_missing = [k for k in missing if not k.startswith("answer_head")]
    assert not real_missing, f"real missing keys: {real_missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    print(f"[load] loaded {checkpoint_path}, missing={missing}", flush=True)
    return model


def forward_answer_at_round(model: BDHVBSubspaceDecoder, idx: torch.Tensor, n_rounds: int,
                             answer_label: torch.Tensor | None = None):
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


def train_probe(model, args, device):
    add_answer_head(model, n_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
    rng = random.Random(args.seed)
    step_pool = [1, 2, 3, 4, 6, 8]
    r_pool = [1, 2, 4, 8]
    started = time.perf_counter()
    for step in range(args.n_examples):
        n_steps = rng.choice(step_pool)
        n_rounds = rng.choice(r_pool)
        text, _step_targets, correct, _decoy = generate_register_machine_example(rng, n_steps)
        idx = torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=device)
        label = torch.tensor([correct], dtype=torch.long, device=device)
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
            print(f"[reg_eval] example {step+1}/{args.n_examples} steps={n_steps} R={n_rounds} "
                  f"loss={float(loss):.4f} {rate:.1f} ex/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[reg_eval] probe DONE {args.n_examples} examples in {elapsed:.0f}s", flush=True)
    model.eval()
    return elapsed


@torch.no_grad()
def evaluate_matrix(model, args, device):
    rng = random.Random(args.seed + 999)
    step_pool = [1, 2, 3, 4, 6, 8]
    r_pool = [1, 2, 4, 8, 12, 16]
    matrix = {}
    for n_steps in step_pool:
        matrix[n_steps] = {}
        for n_rounds in r_pool:
            correct = 0
            shortcut_matches = 0
            total = args.eval_n
            for _ in range(total):
                text, _st, correct_val, decoy_val = generate_register_machine_example(rng, n_steps)
                idx = torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=device)
                with autocast_context(args, device):
                    logits, _ = forward_answer_at_round(model, idx, n_rounds)
                pred = int(logits.argmax(dim=-1).item())
                if pred == correct_val:
                    correct += 1
                elif pred == decoy_val:
                    shortcut_matches += 1
            matrix[n_steps][n_rounds] = {"accuracy": correct / total, "shortcut_rate": shortcut_matches / total}
            print(f"  steps={n_steps} R={n_rounds}: accuracy={correct/total:.3f} shortcut_rate={shortcut_matches/total:.3f}", flush=True)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
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
    parser.add_argument("--arm-label", default="unknown")
    args = parser.parse_args()

    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )
    model = load_adaptive_gate_checkpoint(config, args.init_checkpoint, args.gate_hidden, device)
    elapsed = train_probe(model, args, device)
    print(f"=== R x step-count evaluation matrix (arm={args.arm_label}) ===", flush=True)
    matrix = evaluate_matrix(model, args, device)

    report = {"arm_label": args.arm_label, "init_checkpoint": str(args.init_checkpoint), "training_seconds": elapsed, "matrix": matrix}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
