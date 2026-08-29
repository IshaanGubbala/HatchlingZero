#!/usr/bin/env python3
"""Real analytic HBM byte-traffic + arithmetic-intensity ledger for one
BDHVBSubspaceDecoder recurrent round, proposed 2026-08-29 as the first,
cheap (no GPU needed) step before any FlashBDH-style kernel investment:
figure out whether TRAINING is actually memory-bandwidth-bound before
assuming it is. Decode is already confirmed memory-bound elsewhere in
this project (persistent state, batching hurts, CUDA graphs barely
help) -- training's status is the real open question this answers.

Real, disclosed scope/limits:
- Analytic, not measured. This computes bytes moved under an explicit,
  stated assumption ("unfused": every producer writes its full output
  to HBM, every consumer reads it back) -- the true number depends on
  what the CUDA/cuBLASLt kernels PyTorch actually dispatches, which
  this script does not observe. Real profiler cross-check (torch.profiler
  memory timeline, real GPU) is the natural next step if this analytic
  ledger suggests training might be memory-bound -- not a replacement
  for it.
- Weight matrices (encoder/encoder_v/decoder/P/O) are assumed re-read
  from HBM every round -- real, physically grounded: at production
  scale (n_embd=2496, mult=16) encoder/encoder_v/decoder are each
  individually ~199MB in bf16, several times larger than even an
  RTX 5090's 96MB L2 cache, so cross-round weight residency in cache
  is not physically available regardless of kernel fusion cleverness.
- Backward pass roughly doubles FLOPs and, under gradient checkpointing
  (used throughout this project's real training runs), roughly
  RE-RUNS the forward's memory traffic during the backward recompute --
  reported separately, not silently folded into the "training" total,
  since that recompute-vs-cache tradeoff is exactly the tension a real
  fused-kernel proposal needs to resolve (see the script's own
  docstring discussion, not repeated here).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BF16_BYTES = 2

# Real public spec-sheet numbers (bf16 dense tensor-core peak TFLOPS,
# HBM/GDDR bandwidth GB/s) -- stated as such, not independently verified
# against a live device here. Meant for an order-of-magnitude roofline
# check, not a precision claim.
GPU_SPECS = {
    "rtx4090": {"peak_tflops_bf16": 165.0, "bandwidth_gb_s": 1008.0, "l2_mb": 72},
    "rtx5090": {"peak_tflops_bf16": 209.5, "bandwidth_gb_s": 1792.0, "l2_mb": 96},
}


def matmul_flops(m: int, k: int, n: int, batch: int = 1) -> int:
    return 2 * batch * m * k * n


def tensor_bytes(*dims: int) -> int:
    total = 1
    for d in dims:
        total *= d
    return total * BF16_BYTES


def build_ledger(B: int, T: int, D: int, mult: int, nh: int, d_state: int, r: int) -> dict:
    N = D * mult // nh

    ops = []

    def add(name: str, flops: int, weight_bytes: int, activation_read_bytes: int, activation_write_bytes: int):
        ops.append({
            "op": name, "flops": flops, "weight_bytes": weight_bytes,
            "activation_read_bytes": activation_read_bytes, "activation_write_bytes": activation_write_bytes,
            "total_bytes": weight_bytes + activation_read_bytes + activation_write_bytes,
        })

    x_bytes = tensor_bytes(B, 1, T, D)
    xlatent_bytes = tensor_bytes(B, nh, T, N)
    encoder_w = tensor_bytes(nh, D, N)
    add("x_latent = x @ encoder", matmul_flops(T, D, N, batch=B * nh), encoder_w, x_bytes, xlatent_bytes)

    add("x_sparse = relu(x_latent)", B * nh * T * N, 0, xlatent_bytes, xlatent_bytes)

    vbottleneck_bytes = tensor_bytes(B, 1, T, d_state)
    p_w = tensor_bytes(D, d_state)
    add("v_bottleneck = x @ P", matmul_flops(T, D, d_state, batch=B), p_w, x_bytes, vbottleneck_bytes)

    scores_bytes = tensor_bytes(B, nh, T, T)
    add("attn scores = QR @ KR^T", matmul_flops(T, N, T, batch=B * nh), 0, 2 * xlatent_bytes, scores_bytes)
    ykvbottleneck_bytes = tensor_bytes(B, nh, T, d_state)
    add("attn out = scores @ V", matmul_flops(T, T, d_state, batch=B * nh), 0, scores_bytes + vbottleneck_bytes, ykvbottleneck_bytes)

    ykv_bytes = tensor_bytes(B, 1, T, D)
    o_w = tensor_bytes(d_state, D)
    add("yKV = yKV_bottleneck @ O", matmul_flops(T, d_state, D, batch=B), o_w, ykvbottleneck_bytes, ykv_bytes)
    add("yKV = LN(yKV)", B * T * D, 0, ykv_bytes, ykv_bytes)

    ylatent_bytes = tensor_bytes(B, nh, T, N)
    encoder_v_w = tensor_bytes(nh, D, N)
    add("y_latent = yKV @ encoder_v", matmul_flops(T, D, N, batch=B * nh), encoder_v_w, ykv_bytes, ylatent_bytes)
    add("y_sparse = relu(y_latent)", B * nh * T * N, 0, ylatent_bytes, ylatent_bytes)
    xy_bytes = tensor_bytes(B, nh, T, N)
    add("xy_sparse = x_sparse * y_sparse", B * nh * T * N, 0, xlatent_bytes + ylatent_bytes, xy_bytes)

    alpha_bytes = tensor_bytes(B, 1, T, r)
    decoder_up_w = tensor_bytes(nh, N, r)
    add("alpha = xy_sparse @ decoder_up (summed over heads)", matmul_flops(T, N, r, batch=B * nh), decoder_up_w, xy_bytes, alpha_bytes)

    y_bytes = tensor_bytes(B, 1, T, D)
    decoder_down_w = tensor_bytes(r, D)
    add("yMLP = alpha @ decoder_down", matmul_flops(T, r, D, batch=B), decoder_down_w, alpha_bytes, y_bytes)
    add("y = LN(yMLP)", B * T * D, 0, y_bytes, y_bytes)
    add("x = LN(x + y)", B * T * D, 0, x_bytes + y_bytes, x_bytes)

    total_flops = sum(o["flops"] for o in ops)
    total_weight_bytes = sum(o["weight_bytes"] for o in ops)
    total_activation_bytes = sum(o["activation_read_bytes"] + o["activation_write_bytes"] for o in ops)
    total_bytes = total_weight_bytes + total_activation_bytes

    return {
        "shape": {"B": B, "T": T, "D": D, "mult": mult, "nh": nh, "N": N, "d_state": d_state, "r": r},
        "ops": ops,
        "totals": {
            "flops_per_round": total_flops,
            "weight_bytes_per_round": total_weight_bytes,
            "activation_bytes_per_round": total_activation_bytes,
            "total_bytes_per_round": total_bytes,
            "arithmetic_intensity_flops_per_byte": total_flops / total_bytes,
        },
    }


def roofline_check(ledger: dict, n_layer: int) -> dict:
    fwd_flops = ledger["totals"]["flops_per_round"] * n_layer
    fwd_bytes = ledger["totals"]["total_bytes_per_round"] * n_layer
    # Real, standard estimate: backward ~2x forward FLOPs. Under gradient
    # checkpointing, backward also RE-RUNS the forward's memory traffic
    # during recompute -- reported as its own line, not silently merged,
    # since resolving that tradeoff is the actual crux of any fused-kernel
    # proposal (see module docstring).
    total_flops_with_checkpointed_backward = fwd_flops * 3
    total_bytes_with_checkpointed_backward = fwd_bytes * 2  # one fwd pass + one recompute pass
    results = {}
    for gpu, spec in GPU_SPECS.items():
        knee_ai = (spec["peak_tflops_bf16"] * 1e12) / (spec["bandwidth_gb_s"] * 1e9)
        ai = total_flops_with_checkpointed_backward / total_bytes_with_checkpointed_backward
        results[gpu] = {
            "roofline_knee_flops_per_byte": knee_ai,
            "achieved_arithmetic_intensity": ai,
            "regime": "compute-bound (AI above knee)" if ai > knee_ai else "memory-bound (AI below knee)",
            "ratio_to_knee": ai / knee_ai,
        }
    return {
        "n_layer": n_layer,
        "forward_flops_all_rounds": fwd_flops,
        "forward_bytes_all_rounds": fwd_bytes,
        "total_flops_with_checkpointed_backward_recompute": total_flops_with_checkpointed_backward,
        "total_bytes_with_checkpointed_backward_recompute": total_bytes_with_checkpointed_backward,
        "per_gpu": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    args = parser.parse_args()

    ledger = build_ledger(args.batch_size, args.sequence_length, args.n_embd, args.mult,
                           args.n_head, args.d_state, args.subspace_rank)
    roofline = roofline_check(ledger, args.n_layer)

    print("=== per-op ledger (one round) ===", flush=True)
    for op in ledger["ops"]:
        ai = op["flops"] / op["total_bytes"] if op["total_bytes"] > 0 else float("inf")
        print(f"  {op['op']:45s} flops={op['flops']/1e9:8.3f} GFLOP  "
              f"bytes={op['total_bytes']/1e6:8.2f} MB  AI={ai:8.2f} FLOP/byte", flush=True)
    t = ledger["totals"]
    print(f"\n[round totals] flops={t['flops_per_round']/1e9:.3f} GFLOP "
          f"bytes={t['total_bytes_per_round']/1e6:.2f} MB "
          f"AI={t['arithmetic_intensity_flops_per_byte']:.2f} FLOP/byte", flush=True)

    print(f"\n=== roofline check, n_layer={args.n_layer}, gradient-checkpointed backward ===", flush=True)
    for gpu, r in roofline["per_gpu"].items():
        print(f"  {gpu}: knee={r['roofline_knee_flops_per_byte']:.1f} FLOP/byte, "
              f"achieved={r['achieved_arithmetic_intensity']:.1f} FLOP/byte "
              f"({r['ratio_to_knee']:.2f}x knee) -> {r['regime']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"ledger": ledger, "roofline": roofline}, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
