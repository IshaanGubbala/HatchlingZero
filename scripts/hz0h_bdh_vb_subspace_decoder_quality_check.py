#!/usr/bin/env python3
"""Real training-quality check for the compound VB-frozen-identity +
subspace-decoder architecture (reference/hz0h_bdh_vb_subspace_decoder_torch.py).
Same methodology as every quality-check run this session (seed=7,
hz0h_bytes_25m data, 5M tokens, batch=8/seq=256, adamw, bfloat16,
gradient checkpointing, same depth curriculum) -- only the model
differs, so val_loss is directly comparable to exact BDH (1.8585), VB
frozen-forever alone (1.7999/1.8014), and the subspace decoder alone
(1.7972/1.7970).

decoder_up/decoder_down are SVD-warmstarted from the same trained dense
checkpoint used for the standalone subspace-decoder warmstart result
(results/local/hz0h_bdh_checkpoint_for_ablation.pt) -- P/O don't need a
learned warmstart, they're deterministically frozen at truncated
identity per BDHVBSubspaceDecoder's own __init__.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_subspace_decoder_checkpointed_torch import bdh_vb_subspace_decoder_forward_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_gated_residual_torch import add_gated_residual_stream, bdh_vb_subspace_decoder_forward_gated_residual_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_moe_torch import add_moe_decoder, bdh_vb_subspace_decoder_forward_moe_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_mtp_torch import add_mtp_heads, bdh_vb_subspace_decoder_forward_mtp_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_round_embed_torch import add_round_embeddings, bdh_vb_subspace_decoder_forward_round_embed_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_ngram_torch import add_ngram_memory, bdh_vb_subspace_decoder_forward_ngram_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from reference.hz0h_muon_optimizer import HybridOptimizer, make_muon_hybrid_optimizer
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, curriculum_stages, make_optimizer
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, read_batch


def svd_warmstart_decoder(model: BDHVBSubspaceDecoder, checkpoint_path: Path, rank: int, device) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    dense_decoder = ckpt["state_dict"]["decoder"].to(dtype=torch.float32)  # (nh*N, D)
    U, S, V = torch.svd_lowrank(dense_decoder, q=rank + 10, niter=4)
    U, S, V = U[:, :rank], S[:rank], V[:, :rank]
    sqrt_s = S.sqrt()
    decoder_up = U * sqrt_s.unsqueeze(0)
    decoder_down = (V * sqrt_s.unsqueeze(0)).T
    recon_err = torch.linalg.norm(decoder_up @ decoder_down - dense_decoder) / torch.linalg.norm(dense_decoder)
    print(f"[warmstart] rank={rank} decoder relative_reconstruction_error={float(recon_err):.4f}", flush=True)
    with torch.no_grad():
        model.decoder_up.copy_(decoder_up.to(device=device))
        model.decoder_down.copy_(decoder_down.to(device=device))


def train(config, args, device):
    torch.manual_seed(args.seed)
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    if args.init_checkpoint is not None:
        svd_warmstart_decoder(model, args.init_checkpoint, config.subspace_rank, device)
    if args.mtp_order > 0:
        add_mtp_heads(model, list(range(2, args.mtp_order + 1)))
    if args.ngram_order > 0:
        add_ngram_memory(model, args.ngram_table_params, args.ngram_order)
    if args.gated_residual:
        add_gated_residual_stream(model, single_stream=args.gated_residual_single_stream)
    if args.moe_experts > 0:
        add_moe_decoder(model, n_experts=args.moe_experts, top_k=args.moe_top_k)
    if args.round_embed:
        add_round_embeddings(model)
    if args.optimizer == "muon_hybrid":
        optimizer = make_muon_hybrid_optimizer(model, muon_lr=args.muon_lr, adamw_lr=args.learning_rate)
    else:
        optimizer = make_optimizer(model.parameters(), args, device)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    stages = curriculum_stages(args.target_tokens, config.n_layer)
    tokens = 0
    started = time.perf_counter()
    epochs = [0]
    # Real prior-session win (docs/restart/hz0h_rope_hoist_and_compile_mode_results.md,
    # scripts/hz0h_bdh_combined_best_comparison.py): 1.82x CUDA speedup alone, 2.61x
    # compounded with the depth curriculum, lower peak memory too -- never previously
    # applied to any checkpointed+VB+subspace training path. Real, disclosed, UNTESTED
    # combination per combined_best_comparison.py's own --compile-training flag: compile
    # + gradient checkpointing together had not been validated anywhere in this project's
    # history before this flag existed here. torch.compile's own guard system handles
    # recompiling automatically when `depth` changes across curriculum stages.
    forward_fn = torch.compile(bdh_vb_subspace_decoder_forward_checkpointed, mode=args.compile_mode) if args.compile_training else bdh_vb_subspace_decoder_forward_checkpointed
    with args.data.open() as handle:
        for step in range(steps):
            if isinstance(optimizer, HybridOptimizer):
                optimizer.set_lr_scale(lr_at(step, steps, args.warmup_steps, 1.0))
            else:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            depth = depth_at(tokens, stages)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                if args.mtp_order > 0:
                    extra_targets = {k: data[:, k:].contiguous() for k in range(2, args.mtp_order + 1)}
                    _, loss = bdh_vb_subspace_decoder_forward_mtp_checkpointed(model, idx, depth, target, extra_targets)
                elif args.ngram_order > 0:
                    _, loss = bdh_vb_subspace_decoder_forward_ngram_checkpointed(model, idx, depth, target)
                elif args.gated_residual:
                    _, loss = bdh_vb_subspace_decoder_forward_gated_residual_checkpointed(model, idx, depth, target)
                elif args.moe_experts > 0:
                    _, loss = bdh_vb_subspace_decoder_forward_moe_checkpointed(model, idx, depth, target)
                elif args.round_embed:
                    _, loss = bdh_vb_subspace_decoder_forward_round_embed_checkpointed(model, idx, depth, target)
                else:
                    _, loss = forward_fn(model, idx, depth, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
            if args.log_every and (step + 1) % args.log_every == 0:
                now = time.perf_counter()
                rate = tokens / (now - started)
                eta = (steps - step - 1) / max(step + 1, 1) * (now - started)
                print(f"[train_vb_subspace] step {step+1}/{steps} depth={depth} loss={float(loss):.4f} "
                      f"{rate:.0f} tok/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[train_vb_subspace] DONE {tokens} tokens in {elapsed:.0f}s final_loss={float(loss):.4f}", flush=True)
    model.eval()
    return model, elapsed


def evaluate_loss(model, args, device):
    epochs = [0]
    losses = []
    with args.validation_data.open() as handle, torch.no_grad(), autocast_context(args, device):
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            if args.ngram_order > 0:
                # Real architectural component (not a train-only auxiliary
                # loss like MTP) -- must be present at eval time too, or
                # val_loss would measure a different forward path than the
                # one actually being tested.
                _, loss = bdh_vb_subspace_decoder_forward_ngram_checkpointed(model, idx, model.config.n_layer, target)
            elif args.gated_residual:
                _, loss = bdh_vb_subspace_decoder_forward_gated_residual_checkpointed(model, idx, model.config.n_layer, target)
            elif args.moe_experts > 0:
                # aux_loss_coef=0.0: val_loss should be the plain next-token
                # loss only, comparable across every arm -- the load-balancing
                # term is a training-only regularizer, not part of the metric.
                _, loss = bdh_vb_subspace_decoder_forward_moe_checkpointed(model, idx, model.config.n_layer, target, aux_loss_coef=0.0)
            elif args.round_embed:
                _, loss = bdh_vb_subspace_decoder_forward_round_embed_checkpointed(model, idx, model.config.n_layer, target)
            else:
                _, loss = bdh_vb_subspace_decoder_forward_checkpointed(model, idx, model.config.n_layer, target)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("results/local/hz0h_bdh_checkpoint_for_ablation.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tokens", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit", "muon_hybrid"], default="adamw")
    parser.add_argument("--muon-lr", type=float, default=0.02,
                         help="LR for Muon's hidden-matrix group (encoder/encoder_v/decoder_up/decoder_down) "
                              "when --optimizer muon_hybrid. Real Muon reference recipes use lr~0.02, roughly "
                              "20-60x --learning-rate's typical AdamW scale -- deliberately NOT tied to "
                              "--learning-rate, since the two optimizers want different absolute magnitudes; "
                              "only the warmup/decay SHAPE (lr_at's 0..1 curriculum) is shared between them.")
    parser.add_argument("--mtp-order", type=int, choices=[0, 2, 3, 4], default=0,
                         help="0 = baseline (no auxiliary loss, unchanged behavior). N>=2 adds auxiliary "
                              "t+2..t+N prediction heads (real, separate small linear heads on the SAME "
                              "final hidden state -- no extra recurrent-round compute) with the plan's "
                              "1.0/0.5/0.25/0.125 weight schedule. Validation loss is still measured on "
                              "plain next-token prediction only (evaluate_loss never touches the aux heads), "
                              "so val_loss stays directly comparable across mtp_order arms.")
    parser.add_argument("--ngram-order", type=int, choices=[0, 2, 3, 4], default=0,
                         help="0 = disabled. N>=2 adds a real hashed n-gram embedding table "
                              "(Qwen3.8-Flash-Next-inspired, Phase 3 of the integration plan): "
                              "table[hash(x_{t-N+1:t})] injected additively into the input embedding "
                              "before the recurrent round loop, gated by one learnable scalar starting "
                              "near zero. Unlike --mtp-order, this IS present at eval/inference time "
                              "(it's a real architectural component, not a train-only auxiliary loss).")
    parser.add_argument("--ngram-table-params", type=int, default=25_000_000,
                         help="Target real parameter count for the n-gram table (table_size = "
                              "this // n_embd). Only used when --ngram-order > 0.")
    parser.add_argument("--gated-residual", action="store_true",
                         help="Phase 4 of the integration plan: adds a second, small, randomly-initialized "
                              "factored-decoder stream gated by a learnable scalar g2 (starts at 0.01), "
                              "alongside the existing decoder stream now gated by g1 (starts at exactly "
                              "1.0) -- at initialization this reproduces the plain compound model almost "
                              "exactly, unlike --optimizer muon_hybrid/--mtp-order/--ngram-order which all "
                              "perturb the model from step 0. Real architectural component, present at "
                              "both train and eval time.")
    parser.add_argument("--gated-residual-single-stream", action="store_true",
                         help="Isolating ablation for the real 2026-08-28 result (g2 ended at ~0.0002, "
                              "unchanged from init, while g1 dropped 1.0->0.583): builds ONLY g1 gating "
                              "the existing decoder stream, no decoder2/g2 at all. Only used when "
                              "--gated-residual is also set.")
    parser.add_argument("--moe-experts", type=int, default=0,
                         help="Phase 7 of the integration plan: 0 = disabled. N>=1 adds N routed "
                              "output experts (real value/output MoE, NOT addressing -- encoder/Q/K "
                              "stay dense per this project's own addressing-resists-compression finding). "
                              "Shared expert IS the existing decoder_up/decoder_down (unchanged, "
                              "warmstart-compatible); routed experts are new decoder_down_experts, gated "
                              "by a single scalar g_moe starting near zero (same conservative-init lesson "
                              "as --gated-residual). Real architectural component, present at both train "
                              "and eval time (eval uses aux_loss_coef=0.0 so val_loss stays comparable).")
    parser.add_argument("--moe-top-k", type=int, default=2,
                         help="Experts activated per token when --moe-experts > 0.")
    parser.add_argument("--round-embed", action="store_true",
                         help="Phase 2 of the internal-computation phase: z_{r+1} = F(z_r, x, e_r) instead "
                              "of z_{r+1} = F(z_r, x) -- a small learnable per-round embedding injected "
                              "additively into the residual stream before each round's computation. Real "
                              "architectural component, present at both train and eval time.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--compile-training", action="store_true",
                         help="torch.compile the training forward pass. Real, measured 1.82x-2.61x "
                              "win elsewhere in this project, never applied to this checkpointed VB+subspace "
                              "path -- combine + gradient checkpointing together is a real, disclosed, "
                              "previously-untested combination on THIS model.")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="max-autotune",
                         help="torch.compile mode. Defaults to max-autotune per this project's own prior "
                              "CUDA finding (default mode OOM'd on the same card family; max-autotune was "
                              "independently faster AND far lower peak memory).")
    parser.add_argument("--save-checkpoint", type=Path, default=None,
                         help="Real gap this script had every run before: it only ever wrote JSON metrics, "
                              "never the trained model itself -- nothing to load afterward for real "
                              "generation/inference. Saves model.state_dict() + config here if set.")
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
    print(f"[vb_subspace] validation_loss={val_loss} params={params/1e6:.2f}M", flush=True)
    gate_values = None
    if args.gated_residual:
        gate_values = {"g1": float(model.g1)}
        if not args.gated_residual_single_stream:
            gate_values["g2"] = float(model.g2)
        print(f"[gated_residual] final gate_values={gate_values}", flush=True)
    if args.moe_experts > 0:
        gate_values = {"g_moe": float(model.g_moe)}
        print(f"[moe] final gate_values={gate_values}", flush=True)

    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "d_state": config.d_state,
        "subspace_rank": config.subspace_rank,
        "results": {"vb_subspace_decoder": {"validation_loss": val_loss, "parameter_count": params, "training_seconds": elapsed, "gate_values": gate_values}},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": {"n_layer": config.n_layer, "n_embd": config.n_embd, "n_head": config.n_head,
                       "mlp_internal_dim_multiplier": config.mlp_internal_dim_multiplier, "vocab_size": config.vocab_size,
                       "dropout": config.dropout, "d_state": config.d_state, "subspace_rank": config.subspace_rank},
            "seed": args.seed, "target_tokens": args.target_tokens,
            "elapsed_seconds": elapsed, "validation_loss": val_loss,
            "has_round_embed": args.round_embed,
        }, args.save_checkpoint)
        print(f"[done] wrote real checkpoint to {args.save_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
