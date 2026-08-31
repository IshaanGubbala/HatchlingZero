#!/usr/bin/env python3
"""Arms A and B of the progressive-latentization falsification experiment
(plans/newnewplan.md, "Progressive Latentization Training" proposal,
2026-08-31). Both continue training the LOCKED adaptive-gate checkpoint
(val_loss=1.3879) on ordinary LM data (hz0h_bytes_25m) for a matched
extra token budget, differing only in recurrence depth:

  Arm A (--random-r not set): fixed R=8 the whole run -- pure "more of
    the same" continuation, the control every other arm's improvement
    gets measured against.
  Arm B (--random-r): R sampled per step from a distribution centered
    on 6-8 with a real tail out to 12/16 (per Huginn's own recipe: the
    model should see R=12/16 often enough during training that those
    depths aren't exotic, not so often that easy examples waste
    compute) -- same weights across all R (weight tying already this
    project's own validated principle), so the model has an actual
    training-time reason to expect "more R" to mean "more computation
    available," unlike every checkpoint trained so far (always R=8).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_adaptive_gate_torch import add_adaptive_gate, bdh_adaptive_gate_forward_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, make_optimizer
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import lr_at, read_batch

# Huginn-style: mostly 6-8, real tail to 12/16, weighted so easy depths
# dominate wall-clock but deep ones aren't exotic to the model.
R_POOL = [6, 6, 6, 7, 7, 8, 8, 8, 8, 12, 12, 16]


def load_adaptive_gate_checkpoint(config: BDHVBSubspaceDecoderConfig, checkpoint_path: Path, gate_hidden: int, device) -> BDHVBSubspaceDecoder:
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    add_adaptive_gate(model, hidden=gate_hidden, g_init=0.58, state_independent=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    real_missing = [k for k in missing if not k.startswith("answer_head")]
    assert not real_missing, f"real missing keys: {real_missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    print(f"[load] loaded {checkpoint_path}", flush=True)
    return model


def train(config, args, device):
    torch.manual_seed(args.seed)
    model = load_adaptive_gate_checkpoint(config, args.init_checkpoint, args.gate_hidden, device)
    optimizer = make_optimizer(model.parameters(), args, device)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    rng = random.Random(args.seed)
    tokens = 0
    started = time.perf_counter()
    with args.data.open() as handle:
        epochs = [0]
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            n_refresh = rng.choice(R_POOL) if args.random_r else args.n_layer
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                _, loss = bdh_adaptive_gate_forward_checkpointed(model, idx, n_refresh, n_refresh, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = tokens / (now - started)
                eta = (steps - step - 1) / max(step + 1, 1) * (now - started)
                print(f"[continued_lm] arm={'B' if args.random_r else 'A'} step {step+1}/{steps} R={n_refresh} "
                      f"loss={float(loss):.4f} {rate:.0f} tok/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[continued_lm] DONE {tokens} tokens in {elapsed:.0f}s final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model, elapsed


def evaluate_loss(model, args, device):
    epochs = [0]
    losses = []
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            _, loss = bdh_adaptive_gate_forward_checkpointed(model, idx, args.n_layer, args.n_layer, target)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("results/local/hz0h_bdh_adaptive_gate_retrain_checkpoint.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--random-r", action="store_true", help="Arm B: sample R from R_POOL per step. Unset = Arm A: fixed R=n_layer.")
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
    print(f"[continued_lm] arm={'B' if args.random_r else 'A'} validation_loss={val_loss} "
          f"params={params/1e6:.2f}M elapsed={elapsed:.0f}s", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(config), "state_dict": model.state_dict()}, args.save_checkpoint)
        print(f"[continued_lm] saved checkpoint to {args.save_checkpoint}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "arm": "B" if args.random_r else "A", "validation_loss": val_loss, "params": params, "elapsed_s": elapsed,
        "random_r": args.random_r,
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
