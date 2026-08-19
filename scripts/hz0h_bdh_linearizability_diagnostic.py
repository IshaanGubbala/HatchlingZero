#!/usr/bin/env python3
"""BDH trajectory linearizability diagnostic -- tests whether BDH's
recurrent dynamics live on a low-dimensional, approximately LINEAR
manifold, before building any Koopman-operator/jump-ahead architecture
around that idea.

Real motivation: BDH reuses the SAME weights every recurrent iteration,
but the transform F is nonlinear and input-dependent -- there is no
fixed matrix A with x_{r+1} = A x_r, so "just precompute A^n" fails
literally (see `reference/hz0h_bdh_trajectory_torch.py`'s docstring for
why: the attention term alone makes F roughly cubic in x). The open
question is whether F is nonetheless well-approximated, in some LATENT
space, by a single linear operator K that COMPOSES correctly across
multiple steps -- i.e. z_{r+k} ~= K^k z_r, not just z_{r+1} ~= K z_r.
One-step fit is a weak test (almost anything fits one step locally);
multi-step composition is the real bar this script measures.

Method, entirely closed-form (no gradient training for the diagnostic
itself -- only the underlying BDH model needs real gradient training):
1. Train a small real BDH (CPU, to not contend with any concurrent MPS
   run) on the real corpus, fixed depth=8, real curriculum.
2. Capture real trajectories (x_0..x_8) over held-out validation data
   via `bdh_forward_with_trajectory`.
3. Measure raw per-step drift: cosine(x_r, x_{r+1}), relative delta
   norm, and encoder ReLU-mask IoU between consecutive steps.
4. Project pooled states onto their top-K principal components (SVD),
   getting a low-dim latent z_r = x_r @ V_k.
5. Fit ONE shared one-step operator K (least squares, pooled across all
   r -- valid to pool because the same weights are literally reused
   every iteration, so approximate time-invariance is a natural
   assumption to test) such that z_r @ K ~= z_{r+1}.
6. THE key test: compare predicting z_{r+2} via K^2 (composed from the
   one-step fit) against a DIRECTLY-fit 2-step operator A_2 (z_r @ A_2
   ~= z_{r+2}), same for 4-step. If composed K^k tracks the directly-fit
   A_k closely, single-operator composition is real and a jump-ahead
   architecture is well-motivated. If K^k is much worse than A_k, the
   dynamics are only locally (not globally/compositionally) linear, and
   a jump network would need per-horizon training rather than matrix
   powers.

Real, disclosed limits: this is a SINGLE trained model (not a 3-seed
statistical claim -- it's a structural diagnostic, closer to a
profiling tool than a quality ablation), CPU/fp32, small scale. What it
answers is a qualitative structural question (does composition hold at
all, roughly) not a precise quantitative one.
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

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_trajectory_torch import bdh_forward_with_trajectory
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward
from scripts.hz0h_factorized_curriculum_full_comparison import depth_at, lr_at, parse_stages, read_batch


def train_small_model(args, device) -> BDH:
    torch.manual_seed(args.seed)
    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
    )
    model = BDH(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    steps = math.ceil(args.target_tokens / (args.batch_size * args.sequence_length))
    quarter = args.target_tokens // 4
    depths = sorted({max(2, round(args.n_layer * f)) for f in (0.5, 0.75, 1.0)})
    boundaries = [quarter * 2, quarter * 3, args.target_tokens][-len(depths):]
    stages = parse_stages(",".join(f"{b}:{d}" for b, d in zip(boundaries, depths)))
    epochs = [0]
    tokens = 0
    started = time.perf_counter()
    with args.data.open() as handle:
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, steps, args.warmup_steps, args.learning_rate)
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            depth = depth_at(tokens, stages)
            optimizer.zero_grad(set_to_none=True)
            _, loss = bdh_variable_depth_forward(model, data[:, :-1].contiguous(), depth, data[:, 1:].contiguous())
            loss.backward()
            optimizer.step()
            tokens += args.batch_size * args.sequence_length
    print(f"[train] {tokens} tokens in {time.perf_counter() - started:.0f}s, final loss={float(loss):.4f}", flush=True)
    model.eval()
    return model


def collect_trajectories(model: BDH, args, device):
    """Real trajectories over held-out validation data. Returns a list
    of length depth+1, each element a (N, D) tensor pooling all
    (batch, position) states across all validation batches."""
    depth = args.n_layer
    D = args.n_embd
    per_step = [[] for _ in range(depth + 1)]
    mask_agreement_sums = [0.0] * depth
    mask_agreement_counts = [0] * depth
    epochs = [0]
    with args.validation_data.open() as handle, torch.no_grad():
        for _ in range(args.eval_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx = data[:, :-1].contiguous()
            _, _, x_states, relu_masks = bdh_forward_with_trajectory(model, idx, depth)
            for r, state in enumerate(x_states):
                per_step[r].append(state.reshape(-1, D))
            for r in range(len(relu_masks) - 1):
                a, b = relu_masks[r], relu_masks[r + 1]
                intersection = (a & b).sum().item()
                union = (a | b).sum().item()
                mask_agreement_sums[r] += intersection / max(union, 1)
                mask_agreement_counts[r] += 1
    pooled = [torch.cat(chunks, dim=0) for chunks in per_step]
    # IoU between mask r and mask r+1 only makes sense for r in 0..depth-2
    # (depth-1 pairs total, one fewer than the number of masks).
    mask_iou = [
        mask_agreement_sums[r] / mask_agreement_counts[r]
        for r in range(depth - 1)
    ]
    return pooled, mask_iou


def fit_lstsq(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """Closed-form ridge-regularized least squares: solve for K in
    x @ K ~= y. Ridge stabilizes when N is not >> feature dim."""
    d = x.shape[1]
    xtx = x.T @ x + ridge * torch.eye(d, dtype=x.dtype)
    xty = x.T @ y
    K = torch.linalg.solve(xtx, xty)
    return K


def relative_error(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    return (torch.linalg.norm(predicted - actual) / torch.linalg.norm(actual).clamp_min(1e-8)).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu",
                        help="Defaults to CPU specifically so this can run alongside an MPS-based "
                             "training sweep without resource contention.")
    parser.add_argument("--target-tokens", type=int, default=1_500_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--latent-dims", default="32,64,128")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = train_small_model(args, device)
    pooled, mask_iou = collect_trajectories(model, args, device)
    depth = args.n_layer
    D = args.n_embd

    print("[mask_iou] encoder ReLU-mask IoU between consecutive steps:", flush=True)
    for r, iou in enumerate(mask_iou):
        print(f"  step {r}->{r+1}: {iou:.4f}", flush=True)

    print("[raw drift] cosine and relative delta norm between consecutive steps:", flush=True)
    raw_drift = []
    for r in range(depth):
        a, b = pooled[r], pooled[r + 1]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()
        delta_ratio = (torch.linalg.norm(b - a, dim=1) / torch.linalg.norm(a, dim=1).clamp_min(1e-8)).mean().item()
        raw_drift.append({"step": r, "cosine": cos, "relative_delta_norm": delta_ratio})
        print(f"  step {r}->{r+1}: cosine={cos:.4f}  |dx|/|x|={delta_ratio:.4f}", flush=True)

    latent_dims = [int(k) for k in args.latent_dims.split(",")]
    all_states = torch.cat(pooled, dim=0)
    mean = all_states.mean(dim=0, keepdim=True)
    centered = all_states - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2 / (S ** 2).sum()).cumsum(0)

    results_by_dim = {}
    for k in latent_dims:
        k = min(k, Vh.shape[0])
        V_k = Vh[:k].T  # (D, k)
        z = [((state - mean) @ V_k) for state in pooled]  # list of (N, k), depth+1 entries

        # Pool one-step pairs across ALL r (time-invariance assumption).
        z_r_all = torch.cat(z[:-1], dim=0)
        z_r1_all = torch.cat(z[1:], dim=0)
        K = fit_lstsq(z_r_all, z_r1_all)
        one_step_error = relative_error(z_r_all @ K, z_r1_all)

        horizon_results = {}
        for horizon in (2, 4):
            if depth < horizon + 1:
                continue
            pairs_r = torch.cat([z[r] for r in range(depth - horizon + 1)], dim=0)
            pairs_rk = torch.cat([z[r + horizon] for r in range(depth - horizon + 1)], dim=0)
            K_power = torch.linalg.matrix_power(K, horizon)
            composed_error = relative_error(pairs_r @ K_power, pairs_rk)
            A_direct = fit_lstsq(pairs_r, pairs_rk)
            direct_error = relative_error(pairs_r @ A_direct, pairs_rk)
            horizon_results[f"k={horizon}"] = {
                "composed_K_power_relative_error": composed_error,
                "direct_fit_relative_error": direct_error,
                "composition_gap": composed_error - direct_error,
            }
            print(f"[latent_dim={k}] horizon={horizon}: composed_K^{horizon}_error={composed_error:.4f} "
                  f"direct_fit_error={direct_error:.4f} gap={composed_error - direct_error:+.4f}", flush=True)

        results_by_dim[k] = {
            "one_step_relative_error": one_step_error,
            "horizons": horizon_results,
        }
        print(f"[latent_dim={k}] one_step_relative_error={one_step_error:.4f}", flush=True)

    report = {
        "device": str(device),
        "single_model_structural_diagnostic_not_a_3seed_claim": (
            "This is one trained model characterizing recurrence STRUCTURE, closer to a "
            "profiling tool than a quality ablation. Treat as qualitative/directional."
        ),
        "shape": {
            "n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
            "mult": args.mult, "target_tokens": args.target_tokens,
        },
        "pca_explained_variance_cumulative": explained.tolist(),
        "encoder_relu_mask_iou_consecutive_steps": mask_iou,
        "raw_drift": raw_drift,
        "latent_linear_operator_results": {str(k): v for k, v in results_by_dim.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
