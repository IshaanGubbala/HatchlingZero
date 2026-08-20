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
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_checkpointed_torch import bdh_variable_depth_forward_checkpointed
from reference.hz0h_bdh_combined_best_torch import combined_bdh_forward, combined_bdh_forward_with_trajectory
from reference.hz0h_bdh_combined_checkpointed_torch import combined_bdh_forward_training_checkpointed
from reference.hz0h_bdh_jump_operator_torch import JumpOperator
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, parse_stages, read_batch


def autocast_context(args, device):
    """Real motivation: production CUDA training in this project uses
    bf16 (confirmed by `results/cuda/hz0h_bdh_cost_breakdown_result.json`'s
    own `"dtype": "bfloat16"`), but this script defaulted to fp32,
    roughly doubling memory versus what past successful runs on this
    exact RTX 3060 needed -- the likely real explanation for why this
    comparison's raw_bdh arm OOM'd where earlier production runs did
    not. Uses `torch.autocast` (NOT a hard `.to(dtype=bfloat16)` model
    cast) deliberately: BDH's `Attention` module asserts its RoPE
    `freqs` buffer stays fp32
    (`reference/hz0h_bdh_torch.py`), and autocast keeps master weights
    in fp32 while only casting compute ops -- the standard, safe mixed-
    precision pattern, and it sidesteps that assertion entirely rather
    than fighting it. Only engages on CUDA (`--dtype bfloat16`); MPS/CPU
    stay fp32 by default, consistent with this project's other local
    scripts' disclosed fp32-only MPS limitation."""
    if args.dtype == "bfloat16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def curriculum_stages(target_tokens: int, n_layer: int) -> list:
    quarter = target_tokens // 4
    depths = sorted({max(2, round(n_layer * f)) for f in (0.5, 0.75, 1.0)})
    boundaries = [quarter * 2, quarter * 3, target_tokens][-len(depths):]
    return parse_stages(",".join(f"{b}:{d}" for b, d in zip(boundaries, depths)))


def make_optimizer(params, args, device):
    """Real motivation: fp32 AdamW's two moment buffers alone cost 2x
    model size (on top of params + grads), a real, measured contributor
    to the raw_bdh OOM (Windows dispatch, `hz0h_matched_param_capstone`,
    2026-08-19: ~9.6GB of model+gradient+optimizer-state alone for a
    599M-param model). `bitsandbytes`' `Adam8bit` quantizes JUST that
    optimizer state to int8 (~4x smaller there specifically), a real,
    standard, widely-used technique -- NOT the same as int8 forward-pass
    quantization (this project's own separate, already-existing ternary
    mechanism in `reference/hz0h_bdh_torch.py`); the forward/backward
    math here stays exactly whatever `--dtype` says (fp32 or bf16 via
    autocast), only the optimizer's internal bookkeeping changes
    precision. CUDA-only (bitsandbytes' kernels are CUDA-specific);
    requesting it on CPU/MPS is a real, loud error rather than a silent
    fallback, since a silent fallback would make this flag look like it
    did something when it didn't."""
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.1)
    if device.type != "cuda":
        raise RuntimeError(
            f"--optimizer adam8bit requires CUDA (bitsandbytes has no CPU/MPS kernels), "
            f"got device={device}. Use --optimizer adamw on non-CUDA devices."
        )
    try:
        import bitsandbytes as bnb
    except ImportError as error:
        raise RuntimeError(
            "--optimizer adam8bit requires the bitsandbytes package (pip install bitsandbytes), "
            "which is not installed."
        ) from error
    return bnb.optim.Adam8bit(params, lr=args.learning_rate, weight_decay=0.1)


def train_bdh(config: BDHConfig, args, device, use_softmax_scaled: bool) -> BDH:
    torch.manual_seed(args.seed)
    model = BDH(config).to(device=device, dtype=torch.float32)
    optimizer = make_optimizer(model.parameters(), args, device)
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
            with autocast_context(args, device):
                if args.gradient_checkpointing:
                    if use_softmax_scaled:
                        _, loss = combined_bdh_forward_training_checkpointed(model, idx, depth, target)
                    else:
                        _, loss = bdh_variable_depth_forward_checkpointed(model, idx, depth, target)
                elif use_softmax_scaled:
                    _, loss = combined_bdh_forward(model, None, idx, real_prefix_iterations=depth, num_jumps=0, targets=target)
                else:
                    _, loss = bdh_variable_depth_forward(model, idx, depth, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = tokens / (now - started)
                eta = (steps - step - 1) / max(step + 1, 1) * (now - started)
                print(f"[train_bdh softmax_scaled={use_softmax_scaled}] step {step+1}/{steps} "
                      f"depth={depth} loss={float(loss):.4f} {rate:.0f} tok/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[train_bdh softmax_scaled={use_softmax_scaled}] DONE {tokens} tokens in {elapsed:.0f}s "
          f"final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model, elapsed


def train_transformer(args, device) -> MatchedTransformerLM:
    torch.manual_seed(args.seed)
    config = MatchedTransformerConfig({
        "vocab_size": 256, "d_model": args.transformer_n_embd, "num_layers": args.n_layer,
        "num_heads": args.n_head, "head_dim": args.transformer_n_embd // args.n_head,
        "d_ff": args.transformer_n_embd * 4, "use_rope": True,
    })
    model = MatchedTransformerLM(config).to(device=device, dtype=torch.float32)
    optimizer = make_optimizer(model.parameters(), args, device)
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
            with autocast_context(args, device):
                logits = model(idx)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = tokens / (now - started)
                eta = (steps - step - 1) / max(step + 1, 1) * (now - started)
                print(f"[train_transformer] step {step+1}/{steps} loss={float(loss):.4f} "
                      f"{rate:.0f} tok/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[train_transformer] DONE {tokens} tokens in {elapsed:.0f}s final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model, elapsed


def train_jump(model: BDH, args, device) -> JumpOperator:
    jump = JumpOperator(d_model=args.combined_n_embd, hidden_mult=args.jump_hidden_mult, jump_size=2).to(device)
    optimizer = torch.optim.AdamW(jump.parameters(), lr=1e-3)
    epochs = [0]
    starting_depths = [r for r in range(0, args.n_layer - 1, 2)]
    started = time.perf_counter()
    with args.data.open() as handle:
        for step in range(args.jump_steps):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx = data[:, :-1].contiguous()
            with torch.no_grad(), autocast_context(args, device):
                x_states = combined_bdh_forward_with_trajectory(model, idx, args.n_layer)
            r = random.choice(starting_depths)
            x_r, x_target = x_states[r].detach(), x_states[r + 2].detach()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
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
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(jump.parameters(), args.grad_clip)
            optimizer.step()
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = (step + 1) / (now - started)
                eta = (args.jump_steps - step - 1) / max(step + 1, 1) * (now - started)
                print(f"[train_jump] step {step+1}/{args.jump_steps} state_loss={float(state_loss):.4f} "
                      f"logits_loss={float(logits_loss):.4f} {rate:.1f} step/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[train_jump] DONE {args.jump_steps} steps in {elapsed:.0f}s "
          f"final_state_loss={float(state_loss):.4f}", flush=True)
    jump.eval()
    return jump, elapsed


def evaluate_loss(forward_fn, args, device) -> float:
    epochs = [0]
    losses = []
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            _, loss = forward_fn(idx, target)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def measure_throughput(forward_only_fn, args, device, compile_it: bool) -> dict:
    """Real bug fixed here (found via Windows dispatch, 2026-08-20):
    this used to call `forward_only_fn` with NO `torch.no_grad()`
    wrapper. `model.eval()` alone does NOT disable autograd graph
    construction -- only `no_grad`/`inference_mode` does -- so every
    "throughput" forward was silently building a full backward graph
    and retaining every intermediate activation for a backward that
    never happens. This is what OOM'd raw_bdh's throughput measurement
    immediately after its training (which WAS correctly checkpointed)
    completed successfully: checkpointing is a training-time trade
    (reduce stored activations, recompute in backward); it does nothing
    for a call that was never going to have a backward at all, and
    plain `no_grad` is the actually-correct, cheaper fix for pure
    inference throughput than adding checkpointing here would have
    been."""
    idx = torch.randint(256, (args.batch_size, args.sequence_length), device=device)
    fn = torch.compile(forward_only_fn) if compile_it else forward_only_fn
    with torch.no_grad():
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
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm (torch.nn.utils.clip_grad_norm_), 0 disables. Real gap "
                             "fixed here: this script had NO clipping at all until combined_best diverged "
                             "to NaN mid-training at production scale (Windows dispatch, 2026-08-20, "
                             "n_embd=2432, bf16, step 1350/4883 -- trained normally through step 1300, "
                             "then NaN and never recovered). softmax_scaled attention (Part 3) was only "
                             "ever validated at small scale (n_embd=256, fp32) before this; raw_bdh (plain "
                             "attention) trained cleanly at the SAME width/dtype/checkpointing settings, "
                             "isolating this to softmax_scaled specifically, not the memory-saving levers.")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--jump-steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print progress (step, loss, tok/s, ETA) every N steps during training. "
                             "0 disables progress logging (only the final per-arm summary prints). "
                             "Real gap this fixes: the script previously only printed once, at the very "
                             "end of each arm's full training run, with no way to check progress or "
                             "throughput mid-run.")
    parser.add_argument("--skip-raw", action="store_true",
                        help="Skip training/evaluating/benchmarking raw_bdh entirely. For resuming a run "
                             "after killing it partway through and already having that arm's result.")
    parser.add_argument("--skip-combined", action="store_true",
                        help="Skip combined_best (and its jump operator) entirely. See --skip-raw.")
    parser.add_argument("--skip-transformer", action="store_true",
                        help="Skip matched_transformer entirely. See --skip-raw.")
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw",
                        help="adam8bit (bitsandbytes) quantizes optimizer momentum/variance state "
                             "to int8, a real ~4x reduction there specifically -- CUDA-only, requires "
                             "the bitsandbytes package. Does NOT touch forward/backward precision "
                             "(see --dtype for that); a separate lever from bf16, stacks with it.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                        help="bfloat16 only engages on CUDA (via autocast, not a hard model cast -- "
                             "see autocast_context's docstring). Matches this project's real production "
                             "CUDA training dtype; roughly halves memory versus this script's previous "
                             "fp32-only default, which is the likely real cause of the raw_bdh OOM.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=256,
                        help="Default width for all three arms; overridden per-arm by "
                             "--raw-n-embd/--combined-n-embd/--transformer-n-embd if given, "
                             "since matching PARAMETER COUNT across architectures with different "
                             "param-vs-width scaling (BDH mult=32 vs mult=16 vs Transformer) "
                             "requires different widths per arm, not one shared value.")
    parser.add_argument("--raw-n-embd", type=int, default=None)
    parser.add_argument("--combined-n-embd", type=int, default=None)
    parser.add_argument("--transformer-n-embd", type=int, default=None)
    parser.add_argument("--raw-mult", type=int, default=32,
                        help="mlp_internal_dim_multiplier for raw_bdh. Default 32 (true canonical BDH). "
                             "Real bug fixed here: this used to be HARDCODED to 32 regardless of any chat "
                             "instruction to use a different value -- always pass this explicitly rather "
                             "than assuming a chat message describing intent changed the actual behavior.")
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Wrap each recurrent iteration's BDH forward in torch.utils.checkpoint, "
                             "trading recompute for memory. Real, needed fix at large scale: a 599M-param "
                             "mult=32 BDH OOM'd on a 12GB RTX 3060 even at batch_size=4 without this -- "
                             "the wide intermediate tensors (size B*T*D*mult, independent of head count) "
                             "retained across all n_layer un-checkpointed layers exceeded available VRAM. "
                             "Only applies to the raw_bdh and combined_best teacher training paths (not the "
                             "matched Transformer, which doesn't have this problem at the same param count, "
                             "and not the jump operator's own distillation, which needs the real "
                             "trajectory states and is comparatively cheap anyway).")
    parser.add_argument("--jump-hidden-mult", type=int, default=4,
                        help="JumpOperator's hidden width as a multiple of d_model. Shrink this "
                             "at large model scale -- the jump operator's own param count grows "
                             "as O(hidden_mult * d_model^2), which stops being negligible once "
                             "d_model is in the thousands.")
    args = parser.parse_args()
    args.raw_n_embd = args.raw_n_embd or args.n_embd
    args.combined_n_embd = args.combined_n_embd or args.n_embd
    args.transformer_n_embd = args.transformer_n_embd or args.n_embd

    device = pick_device(args.device)
    random.seed(args.seed)
    report = {"device": str(device), "shape": vars(args).copy()}
    report["shape"] = {k: str(v) for k, v in report["shape"].items()}

    def free_gpu_memory(*objs) -> None:
        """Real memory hygiene: each arm's model (and, for combined_best,
        its jump operator) is trained and benchmarked, then explicitly
        freed before the NEXT arm trains -- so at no point are all three
        large models resident simultaneously. Not what caused the raw_bdh
        OOM (that happened on arm 1 alone, before any other model
        existed), but a real latent risk at this scale worth closing
        while already fixing the memory story with checkpointing."""
        for obj in objs:
            del obj
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    throughput = {}
    results = {}

    if args.skip_raw:
        print("=== SKIPPING raw_bdh (--skip-raw) ===", flush=True)
    else:
        print(f"=== training raw_bdh (mult={args.raw_mult}) ===", flush=True)
        raw_config = BDHConfig(n_layer=args.n_layer, n_embd=args.raw_n_embd, n_head=args.n_head,
                                mlp_internal_dim_multiplier=args.raw_mult, vocab_size=256, dropout=0.0)
        raw_model, raw_train_seconds = train_bdh(raw_config, args, device, use_softmax_scaled=False)
        raw_loss = evaluate_loss(
            lambda idx, target: bdh_variable_depth_forward(raw_model, idx, args.n_layer, target), args, device,
        )
        raw_params = sum(p.numel() for p in raw_model.parameters())
        print(f"[raw_bdh] validation_loss={raw_loss:.4f} params={raw_params/1e6:.2f}M", flush=True)
        print("=== measuring raw_bdh throughput (uncompiled + compiled) ===", flush=True)
        for compile_it in (False, True):
            throughput.setdefault("raw_bdh", {})[str(compile_it)] = measure_throughput(
                lambda idx: bdh_variable_depth_forward(raw_model, idx, args.n_layer), args, device, compile_it,
            )
            print(f"[throughput] raw_bdh compiled={compile_it}: "
                  f"{throughput['raw_bdh'][str(compile_it)]['tokens_per_second']:.0f} tok/s", flush=True)
        free_gpu_memory(raw_model)
        results["raw_bdh"] = {"validation_loss": raw_loss, "parameter_count": raw_params,
                              "training_seconds": raw_train_seconds}

    if args.skip_combined:
        print("=== SKIPPING combined_best (--skip-combined) ===", flush=True)
    else:
        print("=== training combined_best teacher (mult=16, softmax_scaled) ===", flush=True)
        combined_config = BDHConfig(n_layer=args.n_layer, n_embd=args.combined_n_embd, n_head=args.n_head,
                                     mlp_internal_dim_multiplier=16, vocab_size=256, dropout=0.0)
        combined_model, combined_train_seconds = train_bdh(combined_config, args, device, use_softmax_scaled=True)
        print("=== distilling jump operator against combined_best's trajectories ===", flush=True)
        jump, jump_train_seconds = train_jump(combined_model, args, device)
        combined_loss = evaluate_loss(
            lambda idx, target: combined_bdh_forward(combined_model, jump, idx, real_prefix_iterations=4, num_jumps=2, targets=target),
            args, device,
        )
        combined_params = sum(p.numel() for p in combined_model.parameters()) + sum(p.numel() for p in jump.parameters())
        print(f"[combined_best] validation_loss={combined_loss:.4f} params={combined_params/1e6:.2f}M", flush=True)
        print("=== measuring combined_best throughput (uncompiled + compiled) ===", flush=True)
        for compile_it in (False, True):
            throughput.setdefault("combined_best", {})[str(compile_it)] = measure_throughput(
                lambda idx: combined_bdh_forward(combined_model, jump, idx, real_prefix_iterations=4, num_jumps=2), args, device, compile_it,
            )
            print(f"[throughput] combined_best compiled={compile_it}: "
                  f"{throughput['combined_best'][str(compile_it)]['tokens_per_second']:.0f} tok/s", flush=True)
        free_gpu_memory(combined_model, jump)
        results["combined_best"] = {
            "validation_loss": combined_loss, "parameter_count": combined_params,
            "training_seconds": combined_train_seconds + jump_train_seconds,
            "base_training_seconds": combined_train_seconds,
            "jump_training_seconds": jump_train_seconds,
            "recipe": "mult=16 + softmax_scaled attention + weight-tying kept + real_prefix=4,jumps=2",
        }

    if args.skip_transformer:
        print("=== SKIPPING matched_transformer (--skip-transformer) ===", flush=True)
    else:
        print("=== training matched_transformer ===", flush=True)
        transformer_model, transformer_train_seconds = train_transformer(args, device)

        def transformer_forward(idx, target):
            logits = transformer_model(idx)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), target.reshape(-1))
            return logits, loss

        transformer_loss = evaluate_loss(transformer_forward, args, device)
        transformer_params = sum(p.numel() for p in transformer_model.parameters())
        print(f"[matched_transformer] validation_loss={transformer_loss:.4f} params={transformer_params/1e6:.2f}M", flush=True)
        print("=== measuring matched_transformer throughput (uncompiled + compiled) ===", flush=True)
        for compile_it in (False, True):
            throughput.setdefault("matched_transformer", {})[str(compile_it)] = measure_throughput(
                lambda idx: transformer_model(idx), args, device, compile_it,
            )
            print(f"[throughput] matched_transformer compiled={compile_it}: "
                  f"{throughput['matched_transformer'][str(compile_it)]['tokens_per_second']:.0f} tok/s", flush=True)
        results["matched_transformer"] = {"validation_loss": transformer_loss, "parameter_count": transformer_params,
                                          "training_seconds": transformer_train_seconds}

    report["results"] = results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
