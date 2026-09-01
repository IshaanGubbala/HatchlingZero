#!/usr/bin/env python3
"""Real byte-level LM pretrain at the new 150M-param HZ-CQ config
(plans/newnewplan.md section 33), run locally (MPS) per explicit user
request -- not a GPU dispatch.

Mirrors scripts/hz0h_bdh_adaptive_gate_quality_check.py exactly (same
model class, same forward function, same corpus, same optimizer/
curriculum-free full-refresh recipe that produced the 1.3879 champion)
at the new dims: n_embd=2128, d_state=532, subspace_rank=64, mult=16,
n_head=8 -> 150,577,280 real params (verified by direct instantiation,
see plans/newnewplan.md section 33's groundwork note).

Real, disclosed limitation: NO SVD warmstart for decoder_up/decoder_down.
Every prior run in this project's history warmstarted that piece from
results/local/hz0h_bdh_checkpoint_for_ablation.pt's dense decoder matrix
-- but that checkpoint is a DENSE (non-subspace) BDH at the OLD 206.47M
dims, and its decoder shape (nh*N, D) does not match this config's, so
it cannot be SVD-factored into this model's rank-64 decoder_up/down.
Building a real dense-BDH-at-150M source to SVD from would mean
repeating this project's entire original bootstrap chain (train dense
BDH -> validate exact baseline -> THEN warmstart the subspace decoder
from it) before this script could even start -- out of scope for one
pretrain run. decoder_up/decoder_down therefore start from the same
random init (std=0.02) used for every other new parameter in this
model; this project's own prior finding (docstring of
scripts/hz0h_bdh_subspace_decoder_warmstart_quality_check.py) is that
this specific piece would likely benefit from SVD warmstart if a
source existed -- a real, known, disclosed gap, not hidden.

The resulting checkpoint (all of embed/encoder/encoder_v/lm_head/attn/
gate/decoder_up/decoder_down, real weights from real training) becomes
the FULL warmstart source for HZ-CQ's later ARC fine-tuning -- solving
the actual problem this pretrain exists to solve (no full-model 150M
checkpoint existed at all before this).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_adaptive_gate_torch import add_adaptive_gate, bdh_adaptive_gate_forward_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, make_optimizer
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import lr_at, read_batch


def train(config, args, device):
    torch.manual_seed(args.seed)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    add_adaptive_gate(model, hidden=args.gate_hidden, g_init=args.g_init)
    optimizer = make_optimizer(model.parameters(), args, device)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    tokens = 0
    started = time.perf_counter()
    with args.data.open() as handle:
        epochs = [0]
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                _, loss = bdh_adaptive_gate_forward_checkpointed(model, idx, args.n_layer, args.n_refresh, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = tokens / (now - started)
                eta = (steps - step - 1) / max(step + 1, 1) * (now - started)
                print(f"[hzcq_150m_pretrain] step {step+1}/{steps} loss={float(loss):.4f} "
                      f"{rate:.0f} tok/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[hzcq_150m_pretrain] DONE {tokens} tokens in {elapsed:.0f}s final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model, elapsed


def evaluate_loss(model, args, device):
    epochs = [0]
    losses = []
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            _, loss = bdh_adaptive_gate_forward_checkpointed(model, idx, args.n_layer, args.n_refresh, target)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=25_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    # 150M-config defaults (plans/newnewplan.md section 33's real groundwork
    # note: 150,577,280 params, verified by direct instantiation).
    parser.add_argument("--n-embd", type=int, default=2128)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=532)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--n-refresh", type=int, default=8, help="full refresh, matching the validated 1.3879 champion mechanism")
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--g-init", type=float, default=0.58)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )
    model, elapsed = train(config, args, device)
    val_loss = evaluate_loss(model, args, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[hzcq_150m_pretrain] validation_loss={val_loss} params={params/1e6:.2f}M elapsed={elapsed:.0f}s "
          f"n_refresh={args.n_refresh}/{args.n_layer}", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(config), "state_dict": model.state_dict()}, args.save_checkpoint)
        print(f"[hzcq_150m_pretrain] saved checkpoint to {args.save_checkpoint}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "validation_loss": val_loss, "params": params, "elapsed_s": elapsed,
        "n_layer": args.n_layer, "n_refresh": args.n_refresh, "n_embd": args.n_embd,
        "target_tokens": args.target_tokens, "no_svd_warmstart": True,
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
