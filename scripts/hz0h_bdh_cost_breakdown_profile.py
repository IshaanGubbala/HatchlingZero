#!/usr/bin/env python3
"""Real per-stage cost breakdown for exact BDH -- the Stage 2 "efficiency
ceiling" / Gate B profiling pass that `plans/hatchlingzero_bdh_transformer_planning.md`
explicitly calls for and that has never actually been run.

The real question this exists to answer, stated plainly: BDH trains ~5.3x
slower and uses ~10-11x more memory than the matched Transformer
(`docs/restart/hz0h_phase_f_same_gpu_comparison_results.md`). Every
execution-layout remap (wide-GEMM encoder 1.71x, bmm encoder_v 1.51x)
made the SAME math run faster without closing that gap. So which part of
BDH's math actually dominates? There are three real, distinct
architectural cost drivers and nobody has measured which one is the
bottleneck:

  1. The 32x width expansion (`N = D * mlp_internal_dim_multiplier / nh`,
     2048 per head at production shape vs a Transformer FFN's 4x).
  2. The recurrence multiplier (the whole block re-runs `n_layer` times).
  3. Attention running at that expanded 2048 width rather than a
     Transformer's narrow per-head `head_dim` (64 at this shape).

Turning the wrong knob has a real, documented cost in this project:
reducing recurrence may directly destroy the recurrent-depth curriculum
that is the credited mechanism behind BDH's only real quality win
(1.582 vs the Transformer's 1.738), and reducing width the naive way is
what FactorizedBDH already tried and lost with. This script measures
first so the next architecture change is evidence-driven.

Method, and its real, disclosed limits:

- Per-stage wall-clock via `torch.cuda.Event` around each stage of ONE
  real recurrent level, plus an analytic FLOP count per stage. Comparing
  measured-time share against FLOP share is the real point: a stage that
  eats more time than FLOPs is an EXECUTION problem (fixable by remap/
  fusion, the category that already worked here); a stage whose time
  share matches its FLOP share is an ARCHITECTURE cost (only fixable by
  changing what BDH computes).
- Correctness gate FIRST: this script transcribes the oracle's own loop
  body to instrument it (never modifying `reference/hz0h_bdh_torch.py`,
  which is read-only upstream). Before ANY timing is reported it asserts
  the transcription reproduces the real oracle's own `forward` output --
  otherwise the profile would describe something that isn't BDH.
- Both forward-only (inference-shaped) and forward+backward
  (training-shaped) are measured, since BDH's real deficit vs the
  Transformer is training-side while its real win is inference-side.
- Timing individual stages requires per-stage synchronization, which
  itself adds real overhead and prevents kernel overlap. Absolute
  per-stage times are therefore NOT comparable to the end-to-end
  benchmark numbers in other docs; the SHARES are what this script is
  for.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig


def analytic_flops(batch_size: int, sequence_length: int, n_embd: int, n_head: int, latent_width: int) -> dict:
    """Real multiply-add FLOP counts (2 per MAC) for one recurrent level,
    derived directly from the oracle's own forward shapes."""
    B, T, D, nh, N = batch_size, sequence_length, n_embd, n_head, latent_width
    return {
        # x (B,1,T,D) @ encoder (nh,D,N) -> (B,nh,T,N)
        "encoder": 2 * B * nh * T * D * N,
        # QR (B,nh,T,N) @ KR.mT (B,nh,N,T) -> (B,nh,T,T): the real cost of
        # running attention at the EXPANDED width N, not a narrow head_dim.
        "attention_scores": 2 * B * nh * T * T * N,
        # scores (B,nh,T,T) @ V (B,1,T,D) -> (B,nh,T,D)
        "attention_values": 2 * B * nh * T * T * D,
        # yKV (B,nh,T,D) @ encoder_v (nh,D,N) -> (B,nh,T,N)
        "encoder_v": 2 * B * nh * T * D * N,
        # xy_sparse (B,1,T,N*nh) @ decoder (nh*N,D) -> (B,1,T,D)
        "decoder": 2 * B * T * (N * nh) * D,
    }


def matched_transformer_flops(batch_size: int, sequence_length: int, d_model: int, n_layer: int,
                               n_heads: int, head_dim: int, d_ff: int) -> dict:
    """Real FLOP count for the matched Transformer baseline this project
    compares against (config from `scripts/hz0h_factorized_quality_probe.py`
    and the Phase F comparison), so BDH's cost can be stated as a real
    ratio rather than an impression."""
    B, T, D = batch_size, sequence_length, d_model
    per_layer = {
        "qkv_projections": 3 * 2 * B * T * D * D,
        "attention_scores": 2 * B * n_heads * T * T * head_dim,
        "attention_values": 2 * B * n_heads * T * T * head_dim,
        "output_projection": 2 * B * T * D * D,
        "feedforward": 2 * 2 * B * T * D * d_ff,
    }
    return {"per_layer": per_layer, "per_layer_total": sum(per_layer.values()),
            "whole_model": sum(per_layer.values()) * n_layer}


def _timed(callable_stage, iterations: int):
    """Real CUDA-event timing for one stage, median of `iterations`."""
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        result = callable_stage()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], result


def profile_one_level(model: BDH, x: torch.Tensor, iterations: int) -> tuple[dict, torch.Tensor]:
    """Instrumented transcription of the oracle's own recurrent-level body
    (reference/hz0h_bdh_torch.py's `forward` loop), stage by stage."""
    C = model.config
    B, _, T, D = x.shape
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    stage_ms = {}

    stage_ms["encoder"], x_latent = _timed(lambda: x @ model._w(model.encoder), iterations)
    stage_ms["relu_x"], x_sparse = _timed(lambda: F.relu(x_latent), iterations)

    # Attention, split into its own real internals so the expanded-width
    # cost is visible separately from the rest of the block.
    freqs = model.attn.freqs
    r_phases = torch.arange(0, T, device=freqs.device, dtype=freqs.dtype).view(1, 1, -1, 1) * freqs
    stage_ms["attention_rope"], QR = _timed(lambda: model.attn.rope(r_phases, x_sparse), iterations)
    stage_ms["attention_scores"], scores = _timed(lambda: (QR @ QR.mT).tril(diagonal=-1), iterations)
    stage_ms["attention_values"], yKV_raw = _timed(lambda: scores @ x, iterations)
    stage_ms["ln_attention"], yKV = _timed(lambda: model.ln(yKV_raw), iterations)

    stage_ms["encoder_v"], y_latent = _timed(lambda: yKV @ model._w(model.encoder_v), iterations)
    stage_ms["relu_y"], y_sparse = _timed(lambda: F.relu(y_latent), iterations)
    stage_ms["gate_multiply"], xy_sparse = _timed(lambda: x_sparse * y_sparse, iterations)
    stage_ms["dropout"], xy_dropped = _timed(lambda: model.drop(xy_sparse), iterations)

    width = D * C.mlp_internal_dim_multiplier // nh
    stage_ms["decoder"], yMLP = _timed(
        lambda: xy_dropped.transpose(1, 2).reshape(B, 1, T, width * nh) @ model._w(model.decoder), iterations
    )
    stage_ms["ln_residual"], x_next = _timed(lambda: model.ln(x + model.ln(yMLP)), iterations)
    return stage_ms, x_next


def verify_transcription_matches_oracle(model: BDH, idx: torch.Tensor) -> float:
    """Real correctness gate: the instrumented decomposition must reproduce
    the oracle's own forward. Without this, the profile could describe
    something that is not actually BDH."""
    C = model.config
    with torch.no_grad():
        oracle_logits, _ = model(idx)

        x = model.ln(model.embed(idx).unsqueeze(1))
        B, _, T, D = x.shape
        nh = C.n_head
        width = D * C.mlp_internal_dim_multiplier // nh
        freqs = model.attn.freqs
        r_phases = torch.arange(0, T, device=freqs.device, dtype=freqs.dtype).view(1, 1, -1, 1) * freqs
        for _level in range(C.n_layer):
            x_sparse = F.relu(x @ model._w(model.encoder))
            QR = model.attn.rope(r_phases, x_sparse)
            scores = (QR @ QR.mT).tril(diagonal=-1)
            yKV = model.ln(scores @ x)
            y_sparse = F.relu(yKV @ model._w(model.encoder_v))
            xy_sparse = model.drop(x_sparse * y_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, width * nh) @ model._w(model.decoder)
            x = model.ln(x + model.ln(yMLP))
        transcribed_logits = x.view(B, T, D) @ model.lm_head
    return float((oracle_logits - transcribed_logits).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-d-ff", type=int, default=2048)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this profile requires real CUDA hardware")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=256, dropout=0.0,
    )
    model = BDH(config).to(device=device, dtype=torch.bfloat16).eval()
    model.attn.freqs = model.attn.freqs.to(torch.float32)
    idx = torch.randint(256, (args.batch_size, args.sequence_length), device=device)
    targets = torch.randint(256, idx.shape, device=device)

    max_transcription_difference = verify_transcription_matches_oracle(model, idx)
    if max_transcription_difference > 5e-2:
        raise RuntimeError(
            f"transcription does not match the real oracle (max abs diff {max_transcription_difference}); "
            "refusing to report a profile of something that is not BDH"
        )

    latent_width = args.n_embd * args.mlp_internal_dim_multiplier // args.n_head
    flops = analytic_flops(args.batch_size, args.sequence_length, args.n_embd, args.n_head, latent_width)

    with torch.no_grad():
        x = model.ln(model.embed(idx).unsqueeze(1))
        for _ in range(args.warmup):
            profile_one_level(model, x, 1)
        stage_ms, _ = profile_one_level(model, x, args.iterations)

    # Real end-to-end forward-only (inference-shaped) and forward+backward
    # (training-shaped) timings, for the real recurrence-multiplier question.
    def time_end_to_end(train: bool) -> float:
        times = []
        for _ in range(args.warmup + args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            if train:
                model.zero_grad(set_to_none=True)
                _, loss = model(idx, targets)
                loss.backward()
            else:
                with torch.no_grad():
                    model(idx)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times = sorted(times[args.warmup:])
        return times[len(times) // 2]

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    forward_only_ms = time_end_to_end(train=False)
    train_step_ms = time_end_to_end(train=True)

    total_stage_ms = sum(stage_ms.values())
    total_flops = sum(flops.values())
    stages = {}
    for stage, milliseconds in sorted(stage_ms.items(), key=lambda item: -item[1]):
        stage_flops = flops.get(stage, 0)
        stages[stage] = {
            "milliseconds": milliseconds,
            "time_share": milliseconds / total_stage_ms,
            "flops": stage_flops,
            "flop_share": stage_flops / total_flops if stage_flops else 0.0,
            # >1 means the stage costs more time than its FLOP share implies
            # (an EXECUTION problem); ~1 means its cost is real arithmetic
            # (an ARCHITECTURE problem).
            "time_over_flop_share": (
                (milliseconds / total_stage_ms) / (stage_flops / total_flops) if stage_flops else None
            ),
        }

    report = {
        "device": torch.cuda.get_device_name(device),
        "dtype": "bfloat16",
        "shape": {
            "batch_size": args.batch_size, "sequence_length": args.sequence_length,
            "n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
            "mlp_internal_dim_multiplier": args.mlp_internal_dim_multiplier,
            "latent_width_per_head": latent_width,
        },
        "transcription_max_abs_difference_vs_oracle": max_transcription_difference,
        "per_level_stages": stages,
        "per_level_total_stage_milliseconds": total_stage_ms,
        "per_level_total_flops": total_flops,
        "whole_model_flops_all_levels": total_flops * args.n_layer,
        "end_to_end_forward_only_milliseconds": forward_only_ms,
        "end_to_end_train_step_milliseconds": train_step_ms,
        "backward_over_forward_ratio": train_step_ms / forward_only_ms,
        "attention_share_of_level_flops": (
            (flops["attention_scores"] + flops["attention_values"]) / total_flops
        ),
        "wide_projection_share_of_level_flops": (
            (flops["encoder"] + flops["encoder_v"] + flops["decoder"]) / total_flops
        ),
    }

    transformer = matched_transformer_flops(
        args.batch_size, args.sequence_length, args.n_embd, args.transformer_layers,
        args.transformer_heads, args.n_embd // args.transformer_heads, args.transformer_d_ff,
    )
    bdh_whole_model_flops = total_flops * args.n_layer
    report["matched_transformer"] = {
        "config": {
            "n_layer": args.transformer_layers, "n_heads": args.transformer_heads,
            "head_dim": args.n_embd // args.transformer_heads, "d_ff": args.transformer_d_ff,
        },
        "per_layer_flops": transformer["per_layer_total"],
        "whole_model_flops": transformer["whole_model"],
    }
    # The single most decisive number this script produces: if BDH's FLOP
    # ratio is LARGER than its measured wall-clock slowdown ratio (5.3x,
    # docs/restart/hz0h_phase_f_same_gpu_comparison_results.md), then BDH is
    # already achieving BETTER hardware efficiency than the Transformer and
    # the remaining gap is real arithmetic -- ARCHITECTURE, not execution
    # debt -- meaning further layout/fusion remaps have little headroom left.
    report["bdh_over_transformer_flop_ratio"] = bdh_whole_model_flops / transformer["whole_model"]
    report["measured_training_slowdown_ratio_phase_f"] = 5.3
    report["implied_relative_hardware_efficiency"] = (
        report["bdh_over_transformer_flop_ratio"] / 5.3
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
