#!/usr/bin/env python3
"""Phase 3 of plans/HatchlingZero_Internal_Computation_Phase_2026-08-29.md:
train with a real state-supervision auxiliary loss on synthetic
object-location examples, interleaved with ordinary LM training on the
real corpus. Every `--synthetic-every` steps, one step trains ONLY on
a real synthetic example (state loss on all n_layer rounds' per-round
probe heads, no LM loss -- there's no natural continuation target for
"Where is X?" text); every other step trains ONLY on the ordinary
corpus (plain LM loss, no state loss -- state probe heads get zero
gradient on these batches automatically, same "hard selection ->
automatic freeze" mechanism used elsewhere this project).

Real, conservative lambda sweep per the plan (MTP's 2026-08-28 failure
is the standing warning that a reasonable-looking auxiliary loss can
damage BDH): {0.01, 0.03, 0.1}, run via --lambda-state.

Promotion requires ordinary LM validation loss to stay intact relative
to the plain baseline (1.4142, results/local/hz0h_vb_subspace_decoder_25m_plain_baseline.json)
-- this script reports val_loss the same way every other 25M-token arm
this project has, for direct comparison. A held-out round-state-probe
comparison (reusing hz0h_bdh_round_state_probe_diagnostic.py on the
resulting checkpoint) is the separate, real test of whether state
supervision helped shape z_r beyond what emerged unsupervised in
Phase 1/2.
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

from reference.hz0h_bdh_vb_subspace_decoder_checkpointed_torch import bdh_vb_subspace_decoder_forward_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_state_supervised_torch import (
    add_state_probe_heads,
    bdh_vb_subspace_decoder_forward_state_supervised_checkpointed,
)
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, curriculum_stages
from scripts.hz0h_bdh_round_state_probe_diagnostic import LOCATIONS, generate_example
from scripts.hz0h_bdh_vb_subspace_decoder_quality_check import svd_warmstart_decoder
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, read_batch


def train(config, args, device):
    torch.manual_seed(args.seed)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    if args.init_checkpoint is not None:
        svd_warmstart_decoder(model, args.init_checkpoint, config.subspace_rank, device)
    add_state_probe_heads(model, n_classes=len(LOCATIONS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))

    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    stages = curriculum_stages(args.target_tokens, config.n_layer)
    rng = random.Random(args.seed)
    tokens = 0
    started = time.perf_counter()
    epochs = [0]
    with args.data.open() as handle:
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            depth = depth_at(tokens, stages)
            optimizer.zero_grad(set_to_none=True)

            if (step + 1) % args.synthetic_every == 0:
                n_hops = rng.choice([1, 2, 3, 4])
                text, answer_idx = generate_example(rng, n_hops)
                idx = torch.tensor([list(text.encode("utf-8"))], dtype=torch.long, device=device)
                state_labels = torch.tensor([answer_idx], dtype=torch.long, device=device)
                with autocast_context(args, device):
                    _, loss = bdh_vb_subspace_decoder_forward_state_supervised_checkpointed(
                        model, idx, depth, targets=None, state_labels=state_labels, lambda_state=args.lambda_state)
                tokens += len(text.encode("utf-8"))
            else:
                data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
                idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
                with autocast_context(args, device):
                    _, loss = bdh_vb_subspace_decoder_forward_state_supervised_checkpointed(
                        model, idx, depth, targets=target, state_labels=None, lambda_state=args.lambda_state)
                tokens += args.batch_size * args.sequence_length

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = tokens / (now - started)
                eta = (steps - step - 1) / max(step + 1, 1) * (now - started)
                kind = "synthetic" if (step + 1) % args.synthetic_every == 0 else "lm"
                print(f"[state_supervised] step {step+1}/{steps} depth={depth} kind={kind} loss={float(loss):.4f} "
                      f"{rate:.0f} tok/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[state_supervised] DONE {tokens} tokens in {elapsed:.0f}s final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model, elapsed


def evaluate_loss(model, args, device):
    epochs = [0]
    losses = []
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            _, loss = bdh_vb_subspace_decoder_forward_checkpointed(model, idx, model.config.n_layer, target)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("results/local/hz0h_bdh_checkpoint_for_ablation.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=25_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--lambda-state", type=float, default=0.03)
    parser.add_argument("--synthetic-every", type=int, default=10,
                         help="1 in N steps trains on a synthetic state-labeled example instead of the "
                              "ordinary corpus.")
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
    print(f"[state_supervised] validation_loss={val_loss} params={params/1e6:.2f}M", flush=True)

    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "results": {"vb_subspace_decoder": {"validation_loss": val_loss, "parameter_count": params, "training_seconds": elapsed}},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        # state_probe_heads are real, "temporary" (per the plan's own framing) --
        # not needed at inference/eval, and stripping them keeps this checkpoint
        # loadable by hz0h_bdh_round_state_probe_diagnostic.py's plain loader
        # (which trains fresh frozen probes anyway, matching Phase 1/2's own
        # methodology) without needing to teach it a third checkpoint variant.
        state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("state_probe_heads")}
        torch.save({
            "state_dict": state_dict,
            "config": {"n_layer": config.n_layer, "n_embd": config.n_embd, "n_head": config.n_head,
                       "mlp_internal_dim_multiplier": config.mlp_internal_dim_multiplier, "vocab_size": config.vocab_size,
                       "dropout": config.dropout, "d_state": config.d_state, "subspace_rank": config.subspace_rank},
            "seed": args.seed, "target_tokens": args.target_tokens,
            "elapsed_seconds": elapsed, "validation_loss": val_loss,
            "has_round_embed": False,
        }, args.save_checkpoint)
        print(f"[done] wrote real checkpoint to {args.save_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
