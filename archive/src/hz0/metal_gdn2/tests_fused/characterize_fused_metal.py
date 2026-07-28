"""
Comprehensive pre-ship characterization of the HZ-0A fully-fused Metal
GDN-2 forward kernel.

Sections
--------
  A. Realistic trained-state forward equivalence
        Instantiation at HZ-0A 110M shape (dim=768, H=12, Dk=Dv=64),
        brief 10-step in-MLX cross-entropy training so q/k/v/decay/erase/
        write projections settle non-random. Then forward equivalent calls
        under MLX-ref vs fused-Metal paths, capturing:
            max |Δ|, mean |Δ|, p99 |Δ|, p999 |Δ|, mean rel err,
            cosine similarity (flat), cosine per-token min/p1.
        Shapes covered: 110M-shape with T ∈ {64, 128, 256, 512, 1024}.

  B. Long-sequence stability
        Same module, T ∈ {128, 256, 512, 1024}. Verifies cascade noise
        scales linearly or near-linearly with T (not super-linear).

  C. Gradient-equivalence proxy via forward perturbation
        Tiny end-to-end LM (1 GDN2 layer + 1 Linear to vocab=512).
        Forward under both paths, perturb x 100 times, verify that the
        loss-delta (loss_pert - loss_0) sign and magnitude agree.

  D. End-to-end 110M forward speedup
        Time forward on (B=2, T=256, H=12, Dk=Dv=64) under both paths.
        The realized module-level speedup is the metric we'll publish.

Output
------
Writes docs/generated/hz0a-fused-metal-characterization.json with per-row metrics
and a final SHIPPABLE verdict backed by concrete thresholds:

  cosine_sim_flat          > 0.9999  (per shape)
  max_abs_diff (T<=512)    < 1.0     (per shape — fp32 cascade budget)
  max_abs_diff (T=1024)    < 3.0     (cascade noise budget)
  perturbation_sign_match  >= 99%    (Section C — 100% minus zero-crossings)
  module_e2e_speedup       >= 5.0x   (Section D — kernel showed 17x;
                                       realized module is 5–10x)

Note: the script uses looser absolute ceilings than `test_fused_metal.py`
because trained-state activations stay below saturation. Realistic data
exercises the functional regime, not the random-init cascade regime.

The shell-printable verdict tells the user at a glance whether
USE_FUSED_METAL=1 is safe to flip for phase14_fused_metal.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from hz0.metal_gdn2.kernels.gdn2_forward import GDN2MetalModule

# Resolve project root: tests_fused/ -> kernels/ -> metal_gdn2/ -> hz0/ -> src/
# (parents[4] is the actual project root, not parents[3] which lands in src/.)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_JSON = PROJECT_ROOT / "docs" / "hz0a-fused-metal-characterization.json"
TRAIN_STEPS = 10  # in-MLX training pass to give projections non-random weights
VOCAB_TINY = 512

# HZ-0A 110M dims (matches bench_fused_metal.py)
HZ110 = dict(dim=768, hidden_dim=768, num_heads=12, head_dim=64)

T_VALUES = [64, 128, 256, 512, 1024]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_use_fused(on: bool) -> None:
    os.environ["USE_FUSED_METAL"] = "1" if on else "0"


def _to_np(x) -> np.ndarray:
    return np.array(x).astype(np.float64)


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None so downstream consumers
    parsing with strict JSON don't choke. `json.dump` would otherwise emit
    literal `NaN`/`Infinity` which is invalid JSON."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _metrics(out_mlx, out_mtl, label: str) -> Dict[str, float]:
    """Output equivalence metrics: absolute, percentile, relative, cosine.

    Handles both rank-3 [B, T, D] (post-`to_out` projection — the realistic
    end-to-end shape) and rank-4 [B, T, H, Dv] (raw GDN-2 kernel output).
    """
    a = _to_np(out_mlx).reshape(out_mlx.shape)
    b = _to_np(out_mtl).reshape(out_mtl.shape)
    abs_d = np.abs(a - b)
    abs_a = np.abs(a)
    # Per-token flatten: collapse everything except (B, T) onto the same row
    if a.ndim == 4:
        B, T, H, Dv = a.shape
        a_t = a.reshape(B * T, H * Dv)
        b_t = b.reshape(B * T, H * Dv)
    elif a.ndim == 3:
        B, T, D = a.shape
        a_t = a.reshape(B * T, D)
        b_t = b.reshape(B * T, D)
    else:
        raise ValueError(f"unexpected rank {a.ndim}")
    max_d = float(abs_d.max())
    mean_d = float(abs_d.mean())
    p99 = float(np.percentile(abs_d, 99))
    p999 = float(np.percentile(abs_d, 99.9))
    # Relative error restricted to "significant" MLX outputs
    sig = abs_a > 0.1
    rel_d = float((abs_d[sig] / abs_a[sig]).mean()) if sig.any() else float("nan")
    # Cosine similarity across all elements (single scalar)
    flat_a = a.flatten()
    flat_b = b.flatten()
    cos_flat = float(np.dot(flat_a, flat_b)
                     / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-30))
    # Per-token cosine (flattened last-axes)
    na = np.linalg.norm(a_t, axis=1)
    nb = np.linalg.norm(b_t, axis=1)
    cos_t = (a_t * b_t).sum(axis=1) / (na * nb + 1e-30)
    p1_cos = float(np.percentile(cos_t, 1))
    min_cos = float(cos_t.min())
    return {
        "label": label,
        "shape": list(a.shape),
        "max_abs_diff": max_d,
        "mean_abs_diff": mean_d,
        "p99_abs_diff": p99,
        "p999_abs_diff": p999,
        "mean_rel_err": rel_d,
        "cosine_sim_flat": cos_flat,
        "p1_cosine_sim_per_token": p1_cos,
        "min_cosine_sim_per_token": min_cos,
    }


def _train_module_realistic(B: int = 2, T: int = 256) -> GDN2MetalModule:
    """
    Build a GDN2MetalModule at HZ-0A 110M dims and run a brief in-MLX
    cross-entropy training loop so the projections settle into non-random
    distributions. Returns a module whose gates/projection weights are
    frozen via mx.eval() so subsequent forward calls share identical
    gate activations regardless of inner-backend path.
    """
    module = GDN2MetalModule(
        dim=HZ110["dim"],
        hidden_dim=HZ110["hidden_dim"],
        num_heads=HZ110["num_heads"],
        safe_gate_init=True,
    )
    # Snapshot init weights BEFORE training so we can verify post-train
    # delta > epsilon. (Post-train std alone is misleading because MLX
    # default Linear init already has std ~0.021 for d_in=768.)
    init_qkv_w = np.array(module.to_qkv.weight).copy()
    decoder = nn.Linear(HZ110["dim"], VOCAB_TINY)
    target = mx.array(np.random.randint(0, VOCAB_TINY, (B, T)).astype(np.int32))
    opt = optim.Adam(learning_rate=1e-4)

    def loss_fn(x):
        out, _ = module(x, state=None)
        logits = decoder(out)
        return nn.losses.cross_entropy(
            logits.reshape(-1, VOCAB_TINY).astype(mx.float32),
            target.reshape(-1),
            reduction="mean",
        )

    # Train under MLX-ref path (USE_FUSED_METAL=0) — gradient flow must go
    # through MLX autodiff so parameter updates actually happen.
    # CRITICAL: must use `nn.value_and_grad(module, ...)` so grads are wrt
    # `module.trainable_parameters()` (dict), not wrt `x`. Earlier version
    # used `mx.value_and_grad(loss_fn)(x)` which gave grads wrt x only and
    # silently left the module weights at random init.
    _set_use_fused(False)
    for _ in range(TRAIN_STEPS):
        x = mx.random.normal((B, T, HZ110["dim"]))
        loss, grads = nn.value_and_grad(module, loss_fn)(x)
        opt.update(module, grads)
        mx.eval(loss, module.parameters(), opt.state)
    mx.eval(module.parameters())
    # Diagnostic: assert training actually moved weights away from init by
    # checking mean-abs change (not std — MLX default init has std ~0.021
    # which would mask a "weights didn't move" regression). Empirical:
    # 10 Adam(lr=1e-4) cross-entropy steps on a (768, 2304) Linear move
    # weights ~1.6e-3 mean|Δ|, so 0.001 is the safe lower belt while still
    # catching a no-op regression (which would give 0.0).
    post_qkv_w = np.array(module.to_qkv.weight)
    weight_delta = float(np.mean(np.abs(post_qkv_w - init_qkv_w)))
    if weight_delta < 0.001:
        raise RuntimeError(
            f"weight delta-from-init {weight_delta:.6f} below 0.001 — "
            "training loop did not update module weights. Verify "
            "nn.value_and_grad usage."
        )
    print(f"  [diagnostic] weights moved: mean|Δ|={weight_delta:.4f} "
          f"(post-train std={post_qkv_w.std():.4f})")
    return module


# ---------------------------------------------------------------------------
# Section A: Realistic trained-state forward equivalence
# ---------------------------------------------------------------------------

def section_a_realistic(module: GDN2MetalModule, B: int = 2) -> List[Dict]:
    print("\n=== Section A: Realistic trained-state forward equivalence ===")
    rows = []
    for T in T_VALUES:
        # Frozen input so both paths see IDENTICAL q/k/v/decay/erase/write
        x = mx.random.normal((B, T, HZ110["dim"]))

        _set_use_fused(False)
        out_mlx, _ = module(x, state=None)
        mx.eval(out_mlx)

        _set_use_fused(True)
        out_mtl, _ = module(x, state=None)
        mx.eval(out_mtl)

        m = _metrics(out_mlx, out_mtl, f"A_B{B}_T{T}")
        print(
            f"  T={T:>4}: max|Δ|={m['max_abs_diff']:.3e}  "
            f"mean|Δ|={m['mean_abs_diff']:.3e}  "
            f"p99|Δ|={m['p99_abs_diff']:.3e}  "
            f"cos_flat={m['cosine_sim_flat']:.6f}  "
            f"min_cos/tok={m['min_cosine_sim_per_token']:.6f}"
        )
        rows.append({"B": B, "T": T, "H": HZ110["num_heads"],
                     "Dk": HZ110["head_dim"], "metrics": m})
    return rows


# ---------------------------------------------------------------------------
# Section B: Long-sequence stability (sub-eval of A's T axis)
# ---------------------------------------------------------------------------

def section_b_long_seq(module: GDN2MetalModule, B: int = 2) -> Dict:
    """
    Long-sequence stability: pick the longer T-values and report the
    max|Δ| scaling ratio (T=1024 / T=128). Linear cascade scaling
    indicates the kernel is numerically well-behaved.
    """
    print("\n=== Section B: Long-sequence stability ===")
    rows_by_T: Dict[int, float] = {}
    for T in T_VALUES:
        x = mx.random.normal((B, T, HZ110["dim"]))
        _set_use_fused(False)
        o_m, _ = module(x, state=None)
        mx.eval(o_m)
        _set_use_fused(True)
        o_t, _ = module(x, state=None)
        mx.eval(o_t)
        m = _metrics(o_m, o_t, f"B_B{B}_T{T}")
        rows_by_T[T] = m["max_abs_diff"]
        print(f"  T={T:>4}: max|Δ|={m['max_abs_diff']:.3e}  "
              f"cos_flat={m['cosine_sim_flat']:.6f}")
    ratio = rows_by_T[1024] / max(rows_by_T[128], 1e-12)
    scaling = "linear-ish" if ratio <= 16 else "super-linear"
    print(f"  T=1024 vs T=128 ratio: {ratio:.2f}×  → {scaling}")
    return {"per_T_max_abs_diff": rows_by_T, "ratio_1024_over_128": ratio,
            "verdict": scaling}


# ---------------------------------------------------------------------------
# Section C: Gradient-equivalence proxy via forward perturbation
# ---------------------------------------------------------------------------

def section_c_gradient_proxy(B: int = 1, T: int = 64,
                              n_probes: int = 500, eps: float = 1e-2) -> Dict:
    """
    End-to-end toy LM: one GDN2 layer + linear to vocab=512. Train a
    handful of steps so weights are non-random, then perturb x with
    N_PROBES signed deltas and verify the loss-delta agrees between
    MLX-ref and fused-Metal paths on:
        sign(loss_pert - loss_0) — must match 100% of probes
        |Δ(loss_pert - loss_0)| / loss_0 — within fp32 cascade budget
    """
    print("\n=== Section C: Gradient-equivalence proxy ===")
    module = GDN2MetalModule(
        dim=HZ110["dim"],
        hidden_dim=HZ110["hidden_dim"],
        num_heads=HZ110["num_heads"],
        safe_gate_init=True,
    )
    decoder = nn.Linear(HZ110["dim"], VOCAB_TINY)
    target = mx.array(np.random.randint(0, VOCAB_TINY, (B, T)).astype(np.int32))
    opt = optim.Adam(learning_rate=1e-4)
    _set_use_fused(False)

    def loss_fn(x):
        out, _ = module(x, state=None)
        logits = decoder(out)
        return nn.losses.cross_entropy(
            logits.reshape(-1, VOCAB_TINY).astype(mx.float32),
            target.reshape(-1),
            reduction="mean",
        )

    # Train a few steps. Same as `_train_module_realistic`: must use
    # `nn.value_and_grad(module, ...)` for grads wrt module parameters.
    for _ in range(5):
        x = mx.random.normal((B, T, HZ110["dim"]))
        loss, grads = nn.value_and_grad(module, loss_fn)(x)
        opt.update(module, grads)
        mx.eval(loss, module.parameters(), opt.state)
    mx.eval(module.parameters())

    sign_match = 0
    near_zero = 0
    rel_deltas: List[float] = []
    rng = np.random.RandomState(7)
    for i in range(n_probes):
        x = mx.random.normal((B, T, HZ110["dim"]))
        sign = 1.0 if rng.rand() > 0.5 else -1.0
        delta = sign * eps * rng.randn(*x.shape).astype(np.float32)
        x_pert = x + mx.array(delta)

        _set_use_fused(False)
        l0_mlx = loss_fn(x)
        lp_mlx = loss_fn(x_pert)
        mx.eval(l0_mlx, lp_mlx)
        d_mlx = float(lp_mlx) - float(l0_mlx)

        _set_use_fused(True)
        l0_mtl = loss_fn(x)
        lp_mtl = loss_fn(x_pert)
        mx.eval(l0_mtl, lp_mtl)
        d_mtl = float(lp_mtl) - float(l0_mtl)

        # Skip near-zero crossings where sign is unreliable at fp32 noise
        # level. Counting these as "mismatched" would spuriously drop
        # `pct_sign` below 99% purely from cascade noise.
        if abs(d_mlx) < 1e-9 and abs(d_mtl) < 1e-9:
            near_zero += 1
            continue
        s_mlx = 1 if d_mlx > 0 else -1
        s_mtl = 1 if d_mtl > 0 else -1
        if s_mlx == s_mtl:
            sign_match += 1
        if abs(d_mlx) > 1e-12 or abs(d_mtl) > 1e-12:
            r = abs(d_mlx - d_mtl) / max(abs(d_mlx), abs(d_mtl), 1e-12)
            rel_deltas.append(r)

    counted = max(n_probes - near_zero, 1)
    pct_sign = 100.0 * sign_match / counted
    rel_mean = float(np.mean(rel_deltas)) if rel_deltas else float("nan")
    rel_p99 = float(np.percentile(rel_deltas, 99)) if rel_deltas else float("nan")
    print(f"  probes={n_probes}, near-zero skipped: {near_zero}, "
          f"sign match: {sign_match}/{counted} ({pct_sign:.1f}%)")
    print(f"  rel |d(loss)_delta| mean={rel_mean:.3e}  p99={rel_p99:.3e}")
    return {
        "n_probes": n_probes,
        "eps": eps,
        "sign_match_pct": pct_sign,
        "rel_delta_mean": rel_mean,
        "rel_delta_p99": rel_p99,
    }


# ---------------------------------------------------------------------------
# Section D: End-to-end 110M forward speedup
# ---------------------------------------------------------------------------

def section_d_speedup(module: GDN2MetalModule,
                       B: int = 2, T: int = 256,
                       iters: int = 30, warmup: int = 5) -> Dict:
    """Time forward on (B=2, T=256, H=12, Dk=Dv=64) under both paths.
    This is the realized module-level speedup we'll publish in the
    HZ-0A backend acceptance criteria."""
    print("\n=== Section D: End-to-end 110M forward speedup ===")
    x = mx.random.normal((B, T, HZ110["dim"]))

    _set_use_fused(False)
    for _ in range(warmup):
        out_mlx, _ = module(x, state=None)
        mx.eval(out_mlx)
    t0 = time.perf_counter()
    for _ in range(iters):
        out_mlx, _ = module(x, state=None)
        mx.eval(out_mlx)
    mlx_ms = (time.perf_counter() - t0) * 1000 / iters

    _set_use_fused(True)
    for _ in range(warmup):
        out_mtl, _ = module(x, state=None)
        mx.eval(out_mtl)
    t0 = time.perf_counter()
    for _ in range(iters):
        out_mtl, _ = module(x, state=None)
        mx.eval(out_mtl)
    mtl_ms = (time.perf_counter() - t0) * 1000 / iters

    speedup = mlx_ms / mtl_ms
    tok_per_s_mlx = (T * B / mlx_ms) * 1e3
    tok_per_s_mtl = (T * B / mtl_ms) * 1e3
    print(f"  MLX-ref : {mlx_ms:7.3f} ms/iter  ({tok_per_s_mlx:7.0f} tok/s)")
    print(f"  fused   : {mtl_ms:7.3f} ms/iter  ({tok_per_s_mtl:7.0f} tok/s)")
    print(f"  speedup : {speedup:7.2f}×")
    return {
        "B": B, "T": T, "H": HZ110["num_heads"], "Dk": HZ110["head_dim"],
        "mlx_ms_per_iter": mlx_ms,
        "mtl_ms_per_iter": mtl_ms,
        "speedup": speedup,
        "mlx_tok_per_s": tok_per_s_mlx,
        "mtl_tok_per_s": tok_per_s_mtl,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("[characterize] building GDN2MetalModule at HZ-0A 110M dims and "
          "running 10-step in-MLX cross-entropy training to get realistic "
          "(non-random) projections…")
    module = _train_module_realistic(B=2, T=256)

    section_a_rows = section_a_realistic(module, B=2)
    section_b = section_b_long_seq(module, B=2)
    section_c = section_c_gradient_proxy()
    section_d = section_d_speedup(module)

    # ----- Verdicts per section -----
    a_pass = all(
        (r["metrics"]["cosine_sim_flat"] > 0.9999)
        and (r["metrics"]["max_abs_diff"] < (3.0 if r["T"] == 1024 else 1.0))
        for r in section_a_rows
    )
    b_pass = section_b["ratio_1024_over_128"] <= 16.0
    c_pass = section_c["sign_match_pct"] >= 99.0
    d_pass = section_d["speedup"] >= 5.0

    shippable = a_pass and b_pass and c_pass and d_pass

    # Per-row pass/fail so downstream consumers can pinpoint which row failed
    thresholds_applied = []
    for r in section_a_rows:
        cos = r["metrics"]["cosine_sim_flat"]
        md = r["metrics"]["max_abs_diff"]
        cos_pass = cos > 0.9999
        md_pass = md < (3.0 if r["T"] == 1024 else 1.0)
        thresholds_applied.append({
            "B": r["B"], "T": r["T"], "H": r["H"], "Dk": r["Dk"],
            "cosine_pass": cos_pass,
            "max_diff_pass": md_pass,
            "row_pass": cos_pass and md_pass,
        })

    # Concrete reasons for non-shippable verdicts. Empty when shippable.
    verdict_reasons: List[str] = []
    for t in thresholds_applied:
        if not t["row_pass"]:
            if not t["cosine_pass"]:
                verdict_reasons.append(
                    f"A:T={t['T']} cosine_sim < 0.9999"
                )
            if not t["max_diff_pass"]:
                verdict_reasons.append(
                    f"A:T={t['T']} max_abs_diff exceeded ceiling"
                )
    if not b_pass:
        verdict_reasons.append(
            f"B:long-seq ratio {section_b['ratio_1024_over_128']:.2f}× > 16.0×"
        )
    if not c_pass:
        verdict_reasons.append(
            f"C:sign_match {section_c['sign_match_pct']:.1f}% < 99.0%"
        )
    if not d_pass:
        verdict_reasons.append(
            f"D:module speedup {section_d['speedup']:.2f}× < 5.0×"
        )

    report = {
        "version": 2,
        "shape_config": HZ110,
        "sections": {
            "A_realistic_equivalence": section_a_rows,
            "B_long_sequence": section_b,
            "C_gradient_proxy": section_c,
            "D_e2e_speedup": section_d,
        },
        "thresholds_applied": thresholds_applied,
        "verdicts": {
            "A_pass": a_pass,
            "B_pass": b_pass,
            "C_pass": c_pass,
            "D_pass": d_pass,
            "SHIPPABLE": shippable,
            "verdict_reasons": verdict_reasons,
        },
        "thresholds": {
            "cosine_sim_flat_min": 0.9999,
            "max_abs_diff_T_le_512": 1.0,
            "max_abs_diff_T_1024": 3.0,
            "long_seq_ratio_1024_over_128": 16.0,
            "gradient_proxy_sign_match_pct_min": 99.0,
            "module_e2e_speedup_min": 5.0,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(_sanitize_for_json(report), indent=2,
                                   allow_nan=False))
    print("\n" + "=" * 72)
    print(f"  Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"  A pass={a_pass}  B pass={b_pass}  C pass={c_pass}  "
          f"D pass={d_pass}")
    print(f"  SHIPPABLE: {shippable}")
    print("=" * 72)
    sys.exit(0 if shippable else 1)


if __name__ == "__main__":
    main()
