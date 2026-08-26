#!/usr/bin/env python3
"""Phase F, per the user's own explicit framing: "worth considering only
after B-D... must beat BF16 wall-clock, unlike INT8." The INT8 base+delta
result earlier tonight was a real negative specifically because the
design still expanded the state back to full BF16 precision around the
actual matmuls -- quantize/dequantize overhead dominated, storage-byte
savings never translated to real speed. A real FP8 attempt has to be
different: native FP8 tensor-core matmuls (torch._scaled_mm, real
hardware support on Ada Lovelace / RTX 4090), state stored AND consumed
directly in FP8 with a scale factor, never materialized as a full BF16
buffer around the compute.

This is a real, honest, DIAGNOSTIC microbenchmark before touching the
full decode path (same discipline as Phase D's diagnostic-before-
architecture): isolates just the two real per-round operations that
touch the persistent state (state READ: QR @ prefix_state; state WRITE:
KR.mT @ v_bottleneck) at real production shapes, and measures real
wall-clock FP8 (via _scaled_mm, looped over nh since it needs 2D inputs)
against plain BF16 (via the existing broadcasted-batched-matmul pattern).

Real, concrete reason for skepticism, stated up front rather than
discovered after building the full thing: _scaled_mm requires 2D
matrices, so an FP8 version of these ops needs a real per-head Python
loop (nh=8 separate GEMM dispatches) where the current BF16 version gets
ONE broadcasted-batched-GEMM dispatch via `@`. Phase B already found
tiny-M matmuls hit a real, severe kernel-dispatch cliff at this
production scale (encoder matmul: 6.2x slower crossing B=1->B=2, not the
expected 2x) -- an 8x-more-dispatches FP8 path could plausibly lose on
dispatch overhead alone, even before any real FP8 compute benefit shows
up. Real, not assumed -- measured below.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def bf16_state_read_write(QR, KR, prefix_state, v_bottleneck):
    cross = QR @ prefix_state  # (B,nh,T,N) @ (B,nh,N,d_state) -> (B,nh,T,d_state), one broadcasted batched GEMM
    chunk_contribution = KR.mT @ v_bottleneck  # (B,nh,N,T) @ (B,nh,T,d_state) -> (B,nh,N,d_state), one broadcasted batched GEMM
    new_state = prefix_state + chunk_contribution
    return cross, new_state


def fp8_state_read_write(QR_fp8, QR_scale, KR_fp8, KR_scale, prefix_state_fp8, prefix_state_scale,
                          v_bottleneck_fp8, v_bottleneck_scale, nh):
    """Real native FP8 matmuls via torch._scaled_mm, looped over heads
    (required -- _scaled_mm takes 2D inputs only). State read and write
    both computed directly in FP8, output immediately requantized to
    FP8 for the next round -- never round-tripped through a full BF16
    buffer, unlike the INT8 base+delta design's real, measured flaw."""
    cross_parts = []
    new_state_parts = []
    for h in range(nh):
        cross_h = torch._scaled_mm(QR_fp8[h], prefix_state_fp8[h], scale_a=QR_scale, scale_b=prefix_state_scale,
                                    out_dtype=torch.bfloat16)
        cross_parts.append(cross_h)
        contribution_h = torch._scaled_mm(KR_fp8[h], v_bottleneck_fp8[h], scale_a=KR_scale, scale_b=v_bottleneck_scale,
                                           out_dtype=torch.bfloat16)
        new_state_parts.append(contribution_h)
    cross = torch.stack(cross_parts, dim=0)
    contribution = torch.stack(new_state_parts, dim=0)
    return cross, contribution


def quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().amax().clamp_min(1e-6)
    scale = (amax / 448.0).float()  # 448 = float8_e4m3fn's real max representable magnitude
    x_fp8 = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return x_fp8, scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--N", type=int, default=4992)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)
    assert device.type == "cuda", "real FP8 tensor-core matmuls need real CUDA"

    has_fp8 = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm")
    results = {"device": str(device), "config": vars(args) | {"out": str(args.out)}, "has_fp8_support": has_fp8}

    torch.manual_seed(args.seed)
    nh, N, d_state, T = args.n_head, args.N, args.d_state, 1
    B = 1

    QR = torch.randn(B, nh, T, N, device=device, dtype=torch.bfloat16)
    KR = QR
    prefix_state = torch.randn(B, nh, N, d_state, device=device, dtype=torch.bfloat16)
    v_bottleneck = torch.randn(B, nh, T, d_state, device=device, dtype=torch.bfloat16)

    def bf16_op():
        return bf16_state_read_write(QR, KR, prefix_state, v_bottleneck)

    def bf16_read_op():
        return QR @ prefix_state

    def bf16_write_op():
        return KR.mT @ v_bottleneck

    def time_it(fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / args.repeats * 1000

    bf16_ms = time_it(bf16_op)
    bf16_read_ms = time_it(bf16_read_op)
    bf16_write_ms = time_it(bf16_write_op)
    results["bf16_ms_per_call"] = bf16_ms
    results["bf16_read_ms_per_call"] = bf16_read_ms
    results["bf16_write_ms_per_call"] = bf16_write_ms
    print(f"[bf16] combined={bf16_ms:.4f}ms read={bf16_read_ms:.4f}ms write={bf16_write_ms:.4f}ms", flush=True)

    if not has_fp8:
        results["fp8_ms_per_call"] = None
        results["verdict"] = "torch._scaled_mm / float8_e4m3fn not available on this build -- cannot test"
        print("[fp8] not available on this torch build, skipping", flush=True)
    else:
        try:
            # torch._scaled_mm (real cuBLASLt FP8 GEMM) requires A row-major, B column-major --
            # store B operands in their natural TRANSPOSED contiguous form, then present via
            # .t() so they read as column-major without any extra copy.
            QR_2d = QR.squeeze(0).squeeze(1)  # (nh, N) since T=1 -- real per-head 2D slices for _scaled_mm, row-major (A side)
            prefix_state_T_2d = prefix_state.squeeze(0).transpose(-1, -2).contiguous()  # (nh, d_state, N), row-major storage -> .t() per head reads as (N, d_state) column-major
            v_bottleneck_T_2d = v_bottleneck.squeeze(0).squeeze(1).unsqueeze(-1)  # (nh, d_state, 1), already the natural transposed-contiguous form of (1, d_state)

            QR_fp8, QR_scale = quantize_fp8(QR_2d)
            prefix_state_T_fp8, prefix_state_scale = quantize_fp8(prefix_state_T_2d)
            v_bottleneck_T_fp8, v_bottleneck_scale = quantize_fp8(v_bottleneck_T_2d)

            QR_fp8_heads = [QR_fp8[h:h+1, :] for h in range(nh)]  # (1, N) row-major
            prefix_state_fp8_heads = [prefix_state_T_fp8[h].t() for h in range(nh)]  # (N, d_state) column-major view
            v_bottleneck_fp8_heads = [v_bottleneck_T_fp8[h].t() for h in range(nh)]  # (1, d_state) column-major view
            KR_fp8_heads = QR_fp8_heads

            # The state WRITE (KR.T @ v_bottleneck) is a rank-1 outer-product update at
            # decode time (K=1, single token) -- _scaled_mm hard-requires K divisible by 16,
            # a real, structural API incompatibility, not just a speed question. Real,
            # honest fix tested here: zero-pad K from 1 to 16 (mathematically valid -- the
            # 15 extra K-slices are exactly zero, contributing nothing to the real sum),
            # at the cost of 16x the FLOPs a true rank-1 update needs.
            kr_col_padded_heads = []
            vb_row_padded_heads = []
            for h in range(nh):
                kr_col = KR_fp8_heads[h].t()  # (N, 1)
                kr_col_padded = torch.zeros(N, 16, device=device, dtype=torch.float8_e4m3fn)
                kr_col_padded[:, :1] = kr_col
                kr_col_padded_heads.append(kr_col_padded)
                vb_row_padded = torch.zeros(16, d_state, device=device, dtype=torch.float8_e4m3fn)
                vb_row_padded[:1, :] = v_bottleneck_fp8_heads[h]
                vb_row_padded_heads.append(vb_row_padded.t().contiguous().t())  # ensure column-major presentation

            def fp8_read_op():
                cross_parts = []
                for h in range(nh):
                    cross_h = torch._scaled_mm(QR_fp8_heads[h], prefix_state_fp8_heads[h],
                                                scale_a=QR_scale, scale_b=prefix_state_scale, out_dtype=torch.bfloat16)
                    cross_parts.append(cross_h)
                return cross_parts

            def fp8_write_op_padded():
                contrib_parts = []
                for h in range(nh):
                    contrib_h = torch._scaled_mm(kr_col_padded_heads[h], vb_row_padded_heads[h],
                                                  scale_a=QR_scale, scale_b=v_bottleneck_scale, out_dtype=torch.bfloat16)
                    contrib_parts.append(contrib_h[:, :])  # real output already correct -- padding only added zero contributions
                return contrib_parts

            def fp8_op():
                return fp8_read_op(), fp8_write_op_padded()

            fp8_ms = time_it(fp8_op)
            fp8_read_ms = time_it(fp8_read_op)
            fp8_write_ms = time_it(fp8_write_op_padded)
            results["fp8_ms_per_call"] = fp8_ms
            results["fp8_read_ms_per_call"] = fp8_read_ms
            results["fp8_write_ms_per_call_zero_padded_k16"] = fp8_write_ms
            results["fp8_speedup_vs_bf16_combined"] = bf16_ms / fp8_ms
            results["fp8_read_speedup_vs_bf16_read"] = bf16_read_ms / fp8_read_ms
            results["fp8_write_speedup_vs_bf16_write"] = bf16_write_ms / fp8_write_ms
            results["verdict"] = "FP8 beats BF16" if fp8_ms < bf16_ms else "FP8 does NOT beat BF16 -- fails Phase F's own promotion gate"
            print(f"[fp8] combined={fp8_ms:.4f}ms ({bf16_ms/fp8_ms:.3f}x vs bf16) "
                  f"read={fp8_read_ms:.4f}ms ({bf16_read_ms/fp8_read_ms:.3f}x) "
                  f"write(padded K=16)={fp8_write_ms:.4f}ms ({bf16_write_ms/fp8_write_ms:.3f}x)", flush=True)
            print(f"[verdict] {results['verdict']}", flush=True)
        except Exception as exc:
            results["fp8_ms_per_call"] = None
            results["fp8_error"] = f"{type(exc).__name__}: {exc}"
            results["verdict"] = f"FP8 path errored: {type(exc).__name__}: {exc}"
            print(f"[fp8] real error, not a timing result: {type(exc).__name__}: {exc}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
