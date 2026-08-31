#!/usr/bin/env python3
"""Step B of plans/newnewplan.md's revised experimental sequence
(section 28): replace the single global g1 scalar with a tiny
state-dependent gate (reference/hz0h_bdh_adaptive_gate_torch.py),
tested in ISOLATION at full refresh (--n-refresh == --n-layer, i.e.
every iteration re-addresses, matching how the g1 champion itself was
trained) against the single-gate champion (val_loss 1.4142/1.4326).
Not combined with any reduced-refresh schedule yet -- section 28D
("no bundled experiments before isolated wins").

Same methodology as every other arm this session: seed=7,
hz0h_bytes_25m data, matched --target-tokens budget, batch=8/seq=256,
adamw, bfloat16, gradient checkpointing, decoder_up/decoder_down
SVD-warmstarted from the same checkpoint every other arm uses. The
controller itself (113 params at hidden=16) starts protected at the
empirically-observed g~0.58 attractor -- see the reference file's
docstring for why that's not an arbitrary choice.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_adaptive_gate_torch import add_adaptive_gate, bdh_adaptive_gate_forward_checkpointed, _adaptive_g, _address
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, make_optimizer
from scripts.hz0h_bdh_vb_subspace_decoder_quality_check import svd_warmstart_decoder
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import lr_at, read_batch


def train(config, args, device):
    torch.manual_seed(args.seed)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    if args.init_checkpoint is not None:
        svd_warmstart_decoder(model, args.init_checkpoint, config.subspace_rank, device)
    add_adaptive_gate(model, hidden=args.gate_hidden, g_init=args.g_init, state_independent=args.state_independent)
    optimizer = make_optimizer(model.parameters(), args, device)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    tokens = 0
    started = time.perf_counter()
    trajectory = []
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
                print(f"[train_adaptive_gate] step {step+1}/{steps} loss={float(loss):.4f} "
                      f"{rate:.0f} tok/s eta={eta:.0f}s", flush=True)
            if args.gate_trajectory_every and (step + 1) % args.gate_trajectory_every == 0:
                # Real trajectory point: SAME fixed val batch (seed=0) every
                # time, so the only thing changing across points is the
                # model's own weights -- directly answers plans/newnewplan.md's
                # "did g drift during training even though it looks flat at
                # the end" question, not something reconstructable after the
                # fact from a single final checkpoint.
                gs = gate_stats(model, args, device, batch_size=min(2, args.batch_size), seed=0)
                trajectory.append({"step": step + 1, "tokens": tokens, **gs})
                print(f"[gate_trajectory] step {step+1}/{steps} tokens={tokens} {gs}", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[train_adaptive_gate] DONE {tokens} tokens in {elapsed:.0f}s final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model, elapsed, trajectory


def evaluate_loss(model, args, device):
    epochs = [0]
    losses = []
    gate_values = []
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            _, loss = bdh_adaptive_gate_forward_checkpointed(model, idx, args.n_layer, args.n_refresh, target)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def _gate_forward_pass(model, idx):
    """Shared real forward through encoder->address->value path up to the
    gate's own inputs -- used by both the bf16 (autocast) and fp32 gate
    stats functions so they differ ONLY in precision, not in what's
    computed."""
    C = model.config
    D, nh = C.n_embd, C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    x = model.ln(model.embed(idx).unsqueeze(1))
    e = _address(x, model, nh, N)
    x_sparse = F.relu(x @ model.encoder)
    y_latent = e @ model.encoder_v
    xy_sparse = model.drop(x_sparse * F.relu(y_latent))
    alpha = torch.matmul(xy_sparse, model.decoder_up.view(nh, N, -1)).sum(dim=1, keepdim=True)
    y1 = model.ln(alpha @ model.decoder_down)
    return _adaptive_g(x, y1, x, e, model)


def gate_stats(model, args, device, batch_size=None, seed=0):
    """Real gate-value distribution on a held-out batch, under the SAME
    bf16 autocast every other number in this project's eval path uses --
    the whole point of making it adaptive is that it should vary by
    input; a flat distribution near g_init would mean it learned nothing
    state-dependent (or that bf16 is hiding it -- see gate_stats_fp32)."""
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        data = read_batch(handle, batch_size or args.batch_size, args.sequence_length, device, [seed])
        idx = data[:, :-1].contiguous()
        g = _gate_forward_pass(model, idx)
    return {"mean": float(g.mean()), "std": float(g.std()), "min": float(g.min()), "max": float(g.max())}


def gate_stats_fp32(model, args, device, batch_size=None, seed=0):
    """The cheapest of the two candidate explanations from
    plans/newnewplan.md's "adaptive gate isolation" section: does REAL
    fp32 precision (no autocast at all, not even fp32-params-under-bf16-
    autocast) reveal state-dependent variation that bf16 rounds away?
    Reports per-(batch,token) spread, not just a single flat number --
    real per-position variance is what the "state-dependent" hypothesis
    predicts, not just a wider histogram."""
    model_was_training = model.training
    model.eval()
    with args.validation_data.open() as handle, torch.no_grad():
        data = read_batch(handle, batch_size or args.batch_size, args.sequence_length, device, [seed])
        idx = data[:, :-1].contiguous()
        g = _gate_forward_pass(model, idx)  # no autocast context -- real fp32 throughout
    if model_was_training:
        model.train()
    g_flat = g.reshape(-1)
    return {
        "mean": float(g_flat.mean()), "std": float(g_flat.std()),
        "min": float(g_flat.min()), "max": float(g_flat.max()),
        "std_across_tokens_within_batch0": float(g[0].reshape(-1).std()),  # one sequence, varies by position only
        "n_values": int(g_flat.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("results/local/hz0h_bdh_checkpoint_for_ablation.pt"))
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
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--n-refresh", type=int, default=8, help="step B tests at full refresh (==n_layer); step 28D would lower this")
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--g-init", type=float, default=0.58)
    parser.add_argument("--state-independent", action="store_true",
                         help="The killer control for the fixed-g1 sweep's decisive result "
                              "(every frozen scalar, including g=0.55, was worse than the "
                              "adaptive controller's 1.4023 by a real +0.028): keep the IDENTICAL "
                              "controller architecture/param count/protected init, but feed it a "
                              "constant input instead of real state features, so g_r = C_theta(1) "
                              "-- structurally incapable of varying by token/state/round. See "
                              "reference/hz0h_bdh_adaptive_gate_torch.py's add_adaptive_gate docstring "
                              "for the three-way outcome decomposition this is designed to produce.")
    parser.add_argument("--gate-trajectory-every", type=int, default=0,
                         help="0 = disabled. N>0 records gate_stats on a FIXED held-out batch every "
                              "N training steps, to reconstruct whether g drifted during training "
                              "(e.g. 0.58->0.61->0.57->0.55) even though it looks flat at the end -- "
                              "not reconstructable after the fact from a single final checkpoint.")
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )
    model, elapsed, trajectory = train(config, args, device)
    val_loss = evaluate_loss(model, args, device)
    gstats = gate_stats(model, args, device)
    gstats_fp32 = gate_stats_fp32(model, args, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[adaptive_gate] validation_loss={val_loss} params={params/1e6:.2f}M elapsed={elapsed:.0f}s "
          f"n_refresh={args.n_refresh}/{args.n_layer} state_independent={args.state_independent} "
          f"gate_stats={gstats} gate_stats_fp32={gstats_fp32}", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(config), "state_dict": model.state_dict()}, args.save_checkpoint)
        print(f"[adaptive_gate] saved checkpoint to {args.save_checkpoint}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "validation_loss": val_loss, "params": params, "elapsed_s": elapsed,
        "n_layer": args.n_layer, "n_refresh": args.n_refresh, "gate_hidden": args.gate_hidden,
        "g_init": args.g_init, "state_independent": args.state_independent,
        "gate_stats": gstats, "gate_stats_fp32": gstats_fp32, "gate_trajectory": trajectory,
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
