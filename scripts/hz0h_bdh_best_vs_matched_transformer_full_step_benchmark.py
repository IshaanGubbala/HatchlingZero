#!/usr/bin/env python3
"""Real, current max-speed BDH (packed encoder + symmetric attention
backward, the two independently-validated wins from this session combined
in 981d8e1) vs a parameter-matched Transformer, at the established
production comparison config for this project (n_layer=8, n_head=8,
BDH n_embd=2496/mult=16 ~300.32M params, Transformer d_model=1536
~302.57M params -- ratio 1.0075). "Parameter-matched" is on total param
count and n_layer/n_head, not d_model/n_embd equality: BDH's own hidden
width (n_embd=2496, internal latent N=4992) is structurally different
from a Transformer's d_model, that's the architecture, not a knob being
tuned to cheat the comparison.

Full production training step: forward, backward, grad clipping, one
AdamW step. Fresh process per arm, real-shape batch, repeated trials for
variance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_packed_encoder_symmetric_torch import bdh_packed_encoder_symmetric_forward_checkpointed
from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH
from reference.hz0h_bdh_torch import BDHConfig

ARMS = ("bdh_best", "matched_transformer")


def make_bdh(args, device):
    torch.manual_seed(args.seed)
    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.bdh_n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.bdh_multiplier, vocab_size=args.vocab_size, dropout=0.0,
    )
    model = PackedEncoderBDH(config).to(device=device, dtype=torch.float32)
    return model


def make_transformer(args, device):
    torch.manual_seed(args.seed)
    config = MatchedTransformerConfig({
        "vocab_size": args.vocab_size, "d_model": args.transformer_d_model, "num_layers": args.n_layer,
        "num_heads": args.n_head, "head_dim": args.transformer_d_model // args.n_head,
        "d_ff": args.transformer_d_model * 4, "use_rope": True,
    })
    model = MatchedTransformerLM(config).to(device=device, dtype=torch.float32)
    return model


def make_batch(args, device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    idx = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device, generator=generator)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device, generator=generator)
    return idx, targets


def one_step(arm, model, idx, targets, optimizer, args):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if arm == "bdh_best":
            _, loss = bdh_packed_encoder_symmetric_forward_checkpointed(
                model, idx, model.config.n_layer, targets, checkpoint_segment_size=args.checkpoint_segment_size,
            )
        else:
            logits = model(idx)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return loss


def run_child(args):
    device = torch.device("cuda")
    model = make_bdh(args, device) if args.arm == "bdh_best" else make_transformer(args, device)
    idx, targets = make_batch(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    param_count = sum(p.numel() for p in model.parameters())

    for _ in range(args.warmup):
        one_step(args.arm, model, idx, targets, optimizer, args)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    seconds_per_step_trials = []
    loss_value = None
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.steps):
            loss_value = float(one_step(args.arm, model, idx, targets, optimizer, args))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        seconds_per_step_trials.append(elapsed / args.steps)

    mean = sum(seconds_per_step_trials) / len(seconds_per_step_trials)
    variance = sum((t - mean) ** 2 for t in seconds_per_step_trials) / len(seconds_per_step_trials)
    result = {
        "arm": args.arm,
        "param_count": param_count,
        "seconds_per_step_trials": seconds_per_step_trials,
        "seconds_per_step_mean": mean,
        "seconds_per_step_stdev": variance ** 0.5,
        "tokens_per_second_mean": args.batch_size * args.sequence_length / mean,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "final_loss": loss_value,
        "finite": bool(torch.isfinite(torch.tensor(loss_value))),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


def child_command(args, arm, out):
    return [
        sys.executable, str(Path(__file__).resolve()),
        "--child", "--arm", arm, "--out", str(out),
        "--batch-size", str(args.batch_size),
        "--sequence-length", str(args.sequence_length),
        "--bdh-n-embd", str(args.bdh_n_embd),
        "--bdh-multiplier", str(args.bdh_multiplier),
        "--transformer-d-model", str(args.transformer_d_model),
        "--n-head", str(args.n_head),
        "--n-layer", str(args.n_layer),
        "--vocab-size", str(args.vocab_size),
        "--checkpoint-segment-size", str(args.checkpoint_segment_size),
        "--grad-clip", str(args.grad_clip),
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
    ]


def run_parent(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    results = {}
    with tempfile.TemporaryDirectory(prefix="hz0h_bdh_vs_transformer_") as directory:
        for arm in ARMS:
            child_out = Path(directory) / f"{arm}.json"
            subprocess.run(child_command(args, arm, child_out), check=True)
            results[arm] = json.loads(child_out.read_text(encoding="utf-8"))
    bdh = results["bdh_best"]
    transformer = results["matched_transformer"]
    report = {
        "experiment_id": "bdh_best_vs_matched_transformer_full_step_v1",
        "scope": "complete production training step: forward, backward, grad clipping, one AdamW step",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "config": {
            "batch_size": args.batch_size, "sequence_length": args.sequence_length,
            "n_layer": args.n_layer, "n_head": args.n_head,
            "bdh_n_embd": args.bdh_n_embd, "bdh_multiplier": args.bdh_multiplier,
            "transformer_d_model": args.transformer_d_model,
            "vocab_size": args.vocab_size, "checkpoint_segment_size": args.checkpoint_segment_size,
            "grad_clip": args.grad_clip,
        },
        "bdh_forward": "bdh_packed_encoder_symmetric_forward_checkpointed (this session's max-speed BDH: 981d8e1)",
        "param_count_ratio_transformer_over_bdh": transformer["param_count"] / bdh["param_count"],
        "fresh_subprocess_per_arm": True,
        "arms": results,
        "bdh_over_transformer_throughput": bdh["tokens_per_second_mean"] / transformer["tokens_per_second_mean"],
        "bdh_over_transformer_allocated_memory": bdh["peak_memory_allocated_bytes"] / transformer["peak_memory_allocated_bytes"],
        "bdh_over_transformer_reserved_memory": bdh["peak_memory_reserved_bytes"] / transformer["peak_memory_reserved_bytes"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--bdh-n-embd", type=int, default=2496)
    parser.add_argument("--bdh-multiplier", type=int, default=16)
    parser.add_argument("--transformer-d-model", type=int, default=1536)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--checkpoint-segment-size", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=list(ARMS), default="bdh_best", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.child:
        run_child(parsed)
    else:
        run_parent(parsed)
