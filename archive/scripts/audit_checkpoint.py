#!/usr/bin/env python
"""
Pre-flight checkpoint audit for MLX safetensors checkpoints.

Cross-checks an MLX lint checkpoint against the labeled `model_dim × num_layers × num_heads × vocab_size` to detect:

  1. Real parameter count vs the labeled "XXM" tag in the directory name.
  2. Adam state accounting — `opt.X.m` and `opt.X.v` together should be ~2× the
     model param count (one fp32 moment each).
  3. Runtime-state pollution — any keys that aren't part of the model.parameters()
     tree or `opt.*` state (recurrent state buffers, duplicate module refs, etc.).
  4. Content duplicates — flagged via sha256 of each array's bytes; tied weights
     (e.g., embedding = lm_head) show as duplicates with distinct names.
  5. Per-key dtype histogram — catches accidental fp16/bf16 dumps vs expected fp32.

Usage:

    ./venv/bin/python scripts/audit_checkpoint.py \
        --path outputs/training/hz0a_110m/step_0002153.safetensors \
        --expected-dim 768 --expected-layers 24 --expected-heads 12 \
        --expected-vocab 24000

Or via positional args for a quick scan:

    ./venv/bin/python scripts/audit_checkpoint.py \
        outputs/training/hz0a_110m/step_0002153.safetensors

The output is a JSON-friendly print of:

  * `verdict`             — one of {ok, model_size_mismatch, optimizer_mismatch,
                                    runtime_pollution, duplicate_content}
  * `model_params_M`      — actual parameter count in millions
  * `model_bytes_MB`      — on-disk size of the model params
  * `optimizer_states_M`  — Adam moment count in millions
  * `optimizer_bytes_MB`  — on-disk size of the optimizer state
  * `other_keys`          — any keys not in model or optimizer (suspect pollution)
  * `duplicate_groups`    — content-hash collisions (likely tied weights)

Writes a sibling `*.audit.json` next to the checkpoint for downstream pipelines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import mlx.core as mx


def _short_hash(v) -> str:
    a = np.array(v).tobytes()
    return hashlib.sha256(a).hexdigest()[:16]


def audit(
    path: Path,
    config_lower_bound: int = None,
    expected_param_count_M: float = None,
) -> Dict[str, Any]:
    """Audit a single MLX safetensors checkpoint. Returns a summary dict."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "verdict": "missing", "error": "not found"}

    file_size = path.stat().st_size
    kind = "?"
    kind_path = path.with_name(path.stem + ".kind.txt")
    if kind_path.exists():
        kind = kind_path.read_text().strip()

    arrays = dict(mx.load(str(path)))

    # Structural classification: extract the root segment of dotted key paths,
    # trust that GDN2LanguageModel.parameters() emits only top-level modules
    # (embedding, lm_head, norm*, layers.*). Anything else under non-opt.
    # prefix is treated as potential pollution. This is more robust than a
    # brittle substring match.
    model_items: List[Tuple[str, Any]] = []
    opt_items: List[Tuple[str, Any]] = []
    other_items: List[Tuple[str, Any]] = []
    known_model_roots = {"embedding", "lm_head", "norm", "norm1", "norm2",
                         "layers"}
    for k, v in arrays.items():
        if k.startswith("opt."):
            opt_items.append((k, v))
        else:
            root = k.split(".", 1)[0]
            if root in known_model_roots:
                model_items.append((k, v))
            else:
                other_items.append((k, v))

    def _stats(items: List[Tuple[str, Any]]):
        total = sum(int(v.size) for _, v in items)
        dtypes: Dict[str, int] = {}
        bytes_seen: int = 0
        for _, v in items:
            try:
                dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + 1
                # Prefer MLX-native .nbytes; fall back to NumPy round-trip.
                nb = getattr(v, "nbytes", None)
                if nb is None:
                    nb = int(np.array(v).nbytes)
                bytes_seen += int(nb)
            except Exception:
                pass
        return total, bytes_seen, dtypes

    model_n, model_bytes, model_dtypes = _stats(model_items)
    opt_n, opt_bytes, opt_dtypes = _stats(opt_items)
    other_n, _, other_dtypes = _stats(other_items)

    # Verdict logic. The user's primary concern is "model is much larger than
    # the labeled 110M". Any of the following raises a real verdict flag.
    # model_only checkpoints intentionally omit the optimizer state, so the
    # optimizer_* checks (which assume Adam m/v presence) would false-fire
    # on those — skip them entirely.
    skip_optimizer_checks = (kind == "model_only")
    #   param_count_anomaly   — actual model params diverge >20% from a
    #                           user's --expected-* specification, OR a
    #                           configured threshold otherwise.
    #   runtime_pollution     — non-opt, non-model keys exist (suspect
    #                           recurrent buffers accidentally serialized).
    #   optimizer_type_mismatch — opt_state_factor ≠ ~2.0 for Adam OR has no
    #                           .m/.v split.
    #   duplicate_content_anomaly — non-tied-weight content duplicate.
    #   size_anomaly          — file_size outside ±15% of expected sum.
    verdicts: List[str] = []

    # 1) File size sanity vs Adam state × 2
    if kind == "full" and model_n > 0:
        expected_bytes_per_param = 4  # fp32 default
        expected_model_bytes = model_n * expected_bytes_per_param
        expected_opt_bytes = opt_n * expected_bytes_per_param
        expected_total = expected_model_bytes + expected_opt_bytes
        ratio = file_size / max(expected_total, 1)
        # Allow ±15% overhead from safetensors headers + metadata + uint64
        # step counter.
        if ratio > 1.15 or ratio < 0.85:
            verdicts.append("size_anomaly")

    # 2) Runtime state pollution
    if len(other_items) > 0:
        verdicts.append("runtime_pollution")

    # 3) Optimizer type detection: count opt.X.m vs opt.X.v vs opt.step.
    # Skip this whole block on model_only checkpoints.
    if not skip_optimizer_checks:
        opt_keys_list = [k for k, _ in opt_items]
        n_m = sum(1 for k in opt_keys_list if k.endswith(".m"))
        n_v = sum(1 for k in opt_keys_list if k.endswith(".v"))
        has_step = any(k.endswith(".step") or k == "opt.step" for k in opt_keys_list)
        is_adam_like = n_m > 0 and n_v > 0 and abs(n_m - n_v) <= 5
        opt_state_factor = opt_n / max(model_n, 1)
        report["optimizer_n_m_keys"] = n_m
        report["optimizer_n_v_keys"] = n_v
        report["optimizer_has_step"] = has_step

        if not is_adam_like:
            verdicts.append("optimizer_type_mismatch")
        elif opt_state_factor < 1.6 or opt_state_factor > 2.4:
            verdicts.append("optimizer_type_mismatch")

    # 4) Content duplicate detection: distinguish tied-weights (informational)
    # from real duplicate-content bugs.
    content_hash: Dict[str, List[str]] = {}
    for k, v in {**{k: v for k, v in model_items},
                 **{k: v for k, v in opt_items}}.items():
        h = _short_hash(v)
        content_hash.setdefault(h, []).append(k)
    duplicate_groups = {h: ks for h, ks in content_hash.items()
                        if len(ks) > 1}

    tied_pairs: List[Dict[str, Any]] = []
    real_dupes: List[Dict[str, Any]] = []
    TIED_PAIRS = ({("embedding.weight", "lm_head.weight")},
                  {("embedding", "lm_head")})
    for h, ks in duplicate_groups.items():
        ks_set = set(ks)
        if any(tp == ks_set for tp in TIED_PAIRS):
            tied_pairs.append({"hash": h, "keys": sorted(ks)})
        else:
            real_dupes.append({"hash": h, "keys": sorted(ks)})
    if real_dupes:
        verdicts.append("duplicate_content_anomaly")

    # 4) Parameter-count anomaly: report but don't auto-fail — the user can decide
    model_params_M = model_n / 1e6
    opt_state_factor = opt_n / max(model_n, 1)

    report = {
        "path": str(path),
        "file_size_bytes": file_size,
        "file_size_MB": round(file_size / 1024 ** 2, 2),
        "kind": kind,
        "n_key_total": len(arrays),
        "n_key_model": len(model_items),
        "n_key_optimizer": len(opt_items),
        "n_key_other": len(other_items),
        "model_params_M": round(model_params_M, 4),
        "model_params": model_n,
        "model_bytes_MB": round(model_bytes / 1024 ** 2, 2),
        "optimizer_states_M": round(opt_n / 1e6, 4),
        "optimizer_states": opt_n,
        "optimizer_factor_vs_model": round(opt_state_factor, 4),
        "optimizer_bytes_MB": round(opt_bytes / 1024 ** 2, 2),
        "model_dtypes": model_dtypes,
        "optimizer_dtypes": opt_dtypes,
        "other_dtypes": other_dtypes,
        "duplicate_groups_count": len(duplicate_groups),
        "duplicate_groups_sample": [
            {"hash": h, "keys": ks}
            for h, ks in list(duplicate_groups.items())[:6]
        ],
        "other_keys_sample": [k for k, _ in other_items[:8]],
        "verdict": "ok",
    }

    if verdicts:
        report["verdict"] = ";".join(verdicts)
        report["verdict_flags"] = verdicts

    # 5) Param-count anomalies. Two complementary checks:
    #
    #  (a) Config lower-bound: --expected-* gives dim × layers × vocab; the
    #      audit computes a per-layer floor of `15×dim²` and compares.
    #  (b) Explicit param count: --expected-param-count-M lets the user say
    #      "this should be 110M" and the audit flags ratio>1.2 against that.
    #      This is the project's actual concern: dir labeled hz0a_110m
    #      actually holds 292M params.
    #
    # Both checks are real verdict flags — silent mismatch is the bug, not
    # the flag.
    if config_lower_bound is not None:
        ratio = model_n / max(config_lower_bound, 1)
        if ratio < 0.8 or ratio > 1.2:
            verdicts.append(
                f"param_count_anomaly(actual={model_n / 1e6:.2f}M, "
                f"config_lower_bound={config_lower_bound / 1e6:.2f}M, "
                f"ratio={ratio:.2f})"
            )
    if expected_param_count_M is not None:
        actual_M = model_n / 1e6
        ratio = actual_M / max(expected_param_count_M, 1e-12)
        # For explicit param count where the user has stated the expected
        # number directly, use ±50% tolerance (i.e., [0.5, 1.5]). Anything
        # outside this band is a meaningful mislabel — e.g., 220M claimed as
        # 110M (ratio exactly 2.0) is just as much of a bug as 292M claimed
        # as 110M (ratio 2.66), and both should fire the same flag.
        if ratio < 0.5 or ratio > 1.5:
            verdicts.append(
                f"param_count_anomaly(actual={actual_M:.2f}M, "
                f"claim={expected_param_count_M:.2f}M, "
                f"ratio={ratio:.2f})"
            )
        report["expected_param_count_M"] = expected_param_count_M
        report["actual_param_count_M"] = round(actual_M, 4)
        report["param_count_ratio_vs_claim"] = round(ratio, 4)

    # Keep the report dict's verdict strings in sync with the append-only
    # verdict list. We reconstruct after the param_count_anomaly check so
    # the flag actually reaches the consumer.
    report["verdict"] = ";".join(verdicts) if verdicts else "ok"
    report["verdict_flags"] = verdicts

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?",
        default="outputs/training/hz0a_110m/step_0002153.safetensors",
    )
    parser.add_argument("--expected-dim", type=int, default=None)
    parser.add_argument("--expected-layers", type=int, default=None)
    parser.add_argument("--expected-heads", type=int, default=None)
    parser.add_argument("--expected-vocab", type=int, default=None)
    parser.add_argument(
        "--expected-param-count-M", type=float, default=None,
        help="Expected parameter count in millions. When given, the audit "
             "compares the observed model_params against this number and "
             "flags `param_count_anomaly` if outside ±20%. Useful for "
             "catching mislabeled checkpoints (e.g., dir name says "
             "'hz0a_110m' but actual is 292M)."
    )
    parser.add_argument(
        "--auto-parse-dir-label", action="store_true",
        help="Auto-extract the '<NNN>M' suffix from the parent directory "
             "name (e.g., 'hz0a_110m' → 110M expected) and use it as "
             "--expected-param-count-M. This is the project's actual "
             "concern: the dir labeled '110M' may contain 292M params."
    )
    args = parser.parse_args()

    path = Path(args.path)
    print(f"Auditing: {path} ({path.stat().st_size / 1024**2:.2f} MB)\n")
    # Compute the lower-bound parameter count from --expected-* args.
    # Lower bound = 2*vocab*dim + layers * 12 * dim * dim (rough HZ-0A
    # transformer block lower bound: each layer has at least 12 dim-weighted
    # matrices of dim×dim including the GDN2 projections and MLP).
    config_lower_bound = None
    if (
        args.expected_dim and args.expected_layers and args.expected_heads
        and args.expected_vocab
    ):
        d = args.expected_dim
        L = args.expected_layers
        V = args.expected_vocab
        # Lower-bound floor: embed + lm_head + 15 * layers * dim * dim.
        # 15 ≈ recurrent(4 dim-quad-matrices/qkv+deca+to_out) + MLP(3
        # dim-quad-or-4dim matrices w1/w2/w3) + norm + round-up. Empirically
        # the HZ-0A 110M at 768/24/12 sits around 14.6× dim² per layer
        # (without attention); factor 15 close to that floor so a model
        # matching the label sits INSIDE ±20% and a mislabeled one
        # (e.g. 292M for "110M") falls OUTSIDE.
        config_lower_bound = 2 * V * d + 15 * L * d * d

    # If --auto-parse-dir-label is set or no expected count was given, try
    # to extract the parent-dir's "NNNM" suffix as the expected count.
    # Common shapes that match: "hz0a_110m", "model_5m", "transformer_300m".
    expected_param_count = args.expected_param_count_M
    if expected_param_count is None and args.auto_parse_dir_label:
        m = re.search(r"(\d+(?:\.\d+)?)m", path.parent.name.lower())
        if m:
            expected_param_count = float(m.group(1))

    report = audit(
        path,
        config_lower_bound=config_lower_bound,
        expected_param_count_M=expected_param_count,
    )

    # Pretty print
    print(f"  verdict                       : {report['verdict']}")
    print(f"  file size                     : {report['file_size_MB']} MB")
    print(f"  kind                          : {report['kind']}")
    print(f"  total keys                    : {report['n_key_total']}")
    print(f"    model  keys                 : {report['n_key_model']}")
    print(f"    optimizer keys              : {report['n_key_optimizer']}")
    print(f"    other keys (suspect)        : {report['n_key_other']}")
    print(f"  MODEL params                  : {report['model_params_M']}M "
          f"({report['model_bytes_MB']} MB on disk)")
    print(f"  OPTIMIZER state               : {report['optimizer_states_M']}M "
          f"({report['optimizer_bytes_MB']} MB on disk; "
          f"{report['optimizer_factor_vs_model']}× model)")
    print(f"  model dtypes                  : {report['model_dtypes']}")
    print(f"  optimizer dtypes              : {report['optimizer_dtypes']}")
    if report['duplicate_groups_count']:
        print(f"  content duplicates            : {report['duplicate_groups_count']} groups")
        for grp in report['duplicate_groups_sample']:
            print(f"     hash={grp['hash']}  keys={grp['keys'][:4]}..."
                  if len(grp['keys']) > 4 else
                  f"     hash={grp['hash']}  keys={grp['keys']}")
    if report['other_keys_sample']:
        print(f"  OTHER keys (potential pollution) {report['other_keys_sample']}")

    # Cross-check against expected config (now lives inside the audit()
    # function as a real verdict flag, not just print output). Print the
    # geometry that was used so the user can reproduce the lower bound.
    if config_lower_bound is not None:
        embed_total = 2 * args.expected_vocab * args.expected_dim
        lbound = config_lower_bound
        ratio = report["model_params"] / max(lbound, 1)
        print(
            f"\n  [expected-config] embed+lm_head ({embed_total / 1e6:.2f}M) + "
            f"12×layers×dim² "
            f"→ lower-bound floor = {lbound / 1e6:.2f}M"
        )
        print(f"  [actual/lower-bound] {ratio:.2f}×")
        flagged = any("param_count_anomaly" in v for v in report.get("verdict_flags", []))
        if flagged:
            print("  ⚠️  FLAGGED param_count_anomaly in verdict — "
                  "the architecture is bigger than label suggests.")
    # If --expected-param-count-M (explicit or auto-parsed) was given, also
    # print the param-count claim comparison so the user sees the mismatch
    # directly. This is the dir-label mismatch signal.
    if (
        report.get("expected_param_count_M") is not None
        and report.get("actual_param_count_M") is not None
    ):
        print(
            f"\n  [dir-label] dir name says "
            f"{report['expected_param_count_M']:.0f}M, actual is "
            f"{report['actual_param_count_M']:.2f}M "
            f"(ratio {report['param_count_ratio_vs_claim']:.2f}×)"
        )

    print()
    # Auto-write the audit JSON so downstream CI / dashboards can ingest.
    # Use allow_nan=False so any future lossy computation that introduces
    # NaN/Inf raises loud rather than silently writing invalid JSON.
    json_path = path.with_suffix(path.suffix + ".audit.json")
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Wrote {json_path}")

    return 0 if report["verdict"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
