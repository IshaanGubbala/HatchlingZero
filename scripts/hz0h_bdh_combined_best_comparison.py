#!/usr/bin/env python3
"""The headline comparison: stack EVERY confirmed positive finding from
the inherited-choices audit into one recipe and benchmark it against
raw BDH (true canonical oracle) and the matched Transformer -- both
validation loss and real throughput, not either alone.

Three arms, same data/recipe/token budget:
- `raw_bdh`: true unmodified oracle, canonical `mult=32`, standard
  attention, full real depth=8. Represents "what if none of this
  audit's work had happened."
- `combined_best`: `mult=16` (Part 2), `softmax_scaled` attention
  (Part 3), weight tying kept EXACTLY as upstream (Part 4 found
  untying load-bearing-BAD, so this recipe does not touch it), and a
  trained jump operator substituting for half the real recurrent
  iterations at eval time (Part 6: `real_prefix=4` + 2 jumps, the best
  hybrid point found).
- `matched_transformer`: the standing efficient baseline used
  throughout this project.

Throughput is measured on the ACTUAL compute path each arm uses at
inference (raw_bdh: full mult=32 depth=8; combined_best: mult=16
real_prefix=4 + 2 jump calls; transformer: its own fixed forward),
both with and without `torch.compile` -- confirmed empirically (not
assumed) to give a real ~1.55x speedup on this machine's MPS backend
before being included here.

Real, disclosed limits: single run per arm (not 3-seed -- this is a
capstone comparison of ALREADY-established findings, each of which was
separately 3-seed-confirmed earlier in the audit; the NEW thing being
tested here is whether they compose, not whether each one individually
holds), MPS/fp32, reduced token budget vs the 25M-token CUDA reference.
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

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_combined_best_torch import combined_bdh_forward, combined_bdh_forward_with_trajectory
from reference.hz0h_bdh_jump_operator_torch import JumpOperator
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, parse_stages, read_batch


def curriculum_stages(target_tokens: int, n_layer: int) -> list:
    quarter = target_tokens // 4
    depths = sorted({max(2, round(n_layer * f)) for f in (0.5, 0.75, 1.0)})
    boundaries = [quarter * 2, quarter * 3, target_tokens][-len(depths):]
    return parse_stages(",".join(f"{b}:{d}" for b, d in zip(boundaries, depths)))


def train_bdh(config: BDHConfig, args, device, use_softmax_scaled: bool) -> BDH:
    torch.manual_seed(args.seed)
    model = BDH(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    stages = curriculum_stages(args.target_tokens, config.n_layer)
    epochs = [0]
    tokens = 0
    started = time.perf_counter()
    with args.data.open() as handle:
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            depth = depth_at(tokens, stages)
            optimizer.zero_grad(set_to_none=True)
            if use_softmax_scaled:
                _, loss = combined_bdh_forward(model, None, idx, real_prefix_iterations=depth, num_jumps=0, targets=target)
            else:
                _, loss = bdh_variable_depth_forward(model, idx, depth, target)
            loss.backward()
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
    synchronize(device)
    print(f"[train_bdh softmax_scaled={use_softmax_scaled}] {tokens} tokens in {time.perf_counter()-started:.0f}s "
          f"final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model


def train_transformer(args, device) -> MatchedTransformerLM:
    torch.manual_seed(args.seed)
    config = MatchedTransformerConfig({
        "vocab_size": 256, "d_model": args.n_embd, "num_layers": args.n_layer,
        "num_heads": args.n_head, "head_dim": args.n_embd // args.n_head,
        "d_ff": args.n_embd * 4, "use_rope": True,
    })
    model = MatchedTransformerLM(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    epochs = [0]
    tokens = 0
    started = time.perf_counter()
    with args.data.open() as handle:
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            optimizer.zero_grad(set_to_none=True)
            logits = model(idx)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss.backward()
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
    synchronize(device)
    print(f"[train_transformer] {tokens} tokens in {time.perf_counter()-started:.0f}s final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model


def train_jump(model: BDH, args, device) -> JumpOperator:
    jump = JumpOperator(d_model=args.n_embd, hidden_mult=4, jump_size=2).to(device)
    optimizer = torch.optim.AdamW(jump.parameters(), lr=1e-3)
    epochs = [0]
    starting_depths = [r for r in range(0, args.n_layer - 1, 2)]
    started = time.perf_counter()
    with args.data.open() as handle:
        for step in range(args.jump_steps):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx = data[:, :-1].contiguous()
            with torch.no_grad():
                x_states = combined_bdh_forward_with_trajectory(model, idx, args.n_layer)
            r = random.choice(starting_depths)
            x_r, x_target = x_states[r].detach(), x_states[r + 2].detach()
            optimizer.zero_grad(set_to_none=True)
            predicted = jump(x_r)
            state_loss = torch.nn.functional.mse_loss(predicted, x_target)
            B, _, T, D = predicted.shape
            with torch.no_grad():
                target_logits = x_target.view(B, T, D) @ model.lm_head
            predicted_logits = predicted.view(B, T, D) @ model.lm_head
            logits_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(predicted_logits, dim=-1),
                torch.nn.functional.softmax(target_logits, dim=-1), reduction="batchmean",
            )
            (state_loss + 0.1 * logits_loss).backward()
            optimizer.step()
    synchronize(device)
    print(f"[train_jump] {args.jump_steps} steps in {time.perf_counter()-started:.0f}s "
          f"final_state_loss={float(state_loss):.4f}", flush=True)
    jump.eval()
    return jump


def evaluate_loss(forward_fn, args, device) -> float:
    epochs = [0]
    losses = []
    with args.validation_data.open() as handle, torch.no_grad():
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            _, loss = forward_fn(idx, target)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def measure_throughput(forward_only_fn, args, device, compile_it: bool) -> dict:
    idx = torch.randint(256, (args.batch_size, args.sequence_length), device=device)
    fn = torch.compile(forward_only_fn) if compile_it else forward_only_fn
    for _ in range(3):
        fn(idx)
    synchronize(device)
    steps = 10
    started = time.perf_counter()
    for _ in range(steps):
        fn(idx)
    synchronize(device)
    elapsed = time.perf_counter() - started
    tokens = steps * args.batch_size * args.sequence_length
    return {"compiled": compile_it, "seconds": elapsed, "tokens": tokens, "tokens_per_second": tokens / elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--jump-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=4)
    args = parser.parse_args()

    device = pick_device(args.device)
    random.seed(args.seed)
    report = {"device": str(device), "shape": vars(args).copy()}
    report["shape"] = {k: str(v) for k, v in report["shape"].items()}

    print("=== training raw_bdh (canonical mult=32) ===", flush=True)
    raw_config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                            mlp_internal_dim_multiplier=32, vocab_size=256, dropout=0.0)
    raw_model = train_bdh(raw_config, args, device, use_softmax_scaled=False)
    raw_loss = evaluate_loss(
        lambda idx, target: bdh_variable_depth_forward(raw_model, idx, args.n_layer, target), args, device,
    )
    raw_params = sum(p.numel() for p in raw_model.parameters())
    print(f"[raw_bdh] validation_loss={raw_loss:.4f} params={raw_params/1e6:.2f}M", flush=True)

    print("=== training combined_best teacher (mult=16, softmax_scaled) ===", flush=True)
    combined_config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                 mlp_internal_dim_multiplier=16, vocab_size=256, dropout=0.0)
    combined_model = train_bdh(combined_config, args, device, use_softmax_scaled=True)
    print("=== distilling jump operator against combined_best's trajectories ===", flush=True)
    jump = train_jump(combined_model, args, device)
    combined_loss = evaluate_loss(
        lambda idx, target: combined_bdh_forward(combined_model, jump, idx, real_prefix_iterations=4, num_jumps=2, targets=target),
        args, device,
    )
    combined_params = sum(p.numel() for p in combined_model.parameters()) + sum(p.numel() for p in jump.parameters())
    print(f"[combined_best] validation_loss={combined_loss:.4f} params={combined_params/1e6:.2f}M", flush=True)

    print("=== training matched_transformer ===", flush=True)
    transformer_model = train_transformer(args, device)

    def transformer_forward(idx, target):
        logits = transformer_model(idx)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), target.reshape(-1))
        return logits, loss

    transformer_loss = evaluate_loss(transformer_forward, args, device)
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    print(f"[matched_transformer] validation_loss={transformer_loss:.4f} params={transformer_params/1e6:.2f}M", flush=True)

    print("=== measuring throughput (uncompiled + compiled) ===", flush=True)
    throughput = {}
    for compile_it in (False, True):
        throughput.setdefault("raw_bdh", {})[str(compile_it)] = measure_throughput(
            lambda idx: bdh_variable_depth_forward(raw_model, idx, args.n_layer), args, device, compile_it,
        )
        throughput.setdefault("combined_best", {})[str(compile_it)] = measure_throughput(
            lambda idx: combined_bdh_forward(combined_model, jump, idx, real_prefix_iterations=4, num_jumps=2), args, device, compile_it,
        )
        throughput.setdefault("matched_transformer", {})[str(compile_it)] = measure_throughput(
            lambda idx: transformer_model(idx), args, device, compile_it,
        )
        for name, results in throughput.items():
            r = results[str(compile_it)]
            print(f"[throughput] {name} compiled={compile_it}: {r['tokens_per_second']:.0f} tok/s", flush=True)

    report["results"] = {
        "raw_bdh": {"validation_loss": raw_loss, "parameter_count": raw_params},
        "combined_best": {"validation_loss": combined_loss, "parameter_count": combined_params,
                          "recipe": "mult=16 + softmax_scaled attention + weight-tying kept + real_prefix=4,jumps=2"},
        "matched_transformer": {"validation_loss": transformer_loss, "parameter_count": transformer_params},
    }
    report["throughput"] = throughput
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
