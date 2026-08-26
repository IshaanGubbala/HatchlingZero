#!/usr/bin/env python3
"""Tier 4 item 23's required first diagnostic (plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md,
section 13, Subspace BDH): does the LAST-round gate `g_t = x_sparse * y_sparse`
live in a shared low-dimensional subspace (`g_t ~= U @ alpha_t`, `alpha_t
in R^r`, `r << N`) well enough to matter for the model's real output?

The plan explicitly warns: "do not proceed based only on SVD
participation ratio" -- a low-rank approximation can look numerically
close in raw reconstruction error while still meaningfully changing the
model's actual predictions (or vice versa, given ReLU's own
discontinuity, a modest-looking vector error could flip decisions). This
script measures BOTH:

1. Raw reconstruction error (relative Frobenius norm) at ranks
   16/32/64/128/256, using a shared basis U fit via SVD on a FIT sample
   and evaluated on a SEPARATE held-out sample (avoids overfitting the
   basis to the exact data being scored, same discipline as every other
   diagnostic tonight).
2. Real downstream-logit sensitivity: substitute the rank-r
   reconstruction of g at the LAST recurrent round back into the model's
   own forward computation (decoder -> residual update -> logits) and
   compare the resulting real logits against the model's own true
   logits -- KL divergence and top-1 argmax agreement, not just a vector
   distance in isolation.
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
from scripts.hz0h_factorized_curriculum_full_comparison import read_batch


def run_to_last_round(model: BDH, idx: torch.Tensor):
    """Runs the model up through the LAST round's g computation, returns
    everything needed to both (a) reproduce the real logits and (b)
    substitute a reconstructed g and get NEW logits: the pre-last-round
    x, the real g, and a closure to finish the forward pass given any g."""
    config = model.config
    B, T = idx.shape
    D = config.n_embd
    nh = config.n_head
    N = D * config.mlp_internal_dim_multiplier // nh
    last_round = config.n_layer - 1

    x = model.ln(model.embed(idx).unsqueeze(1))
    with torch.no_grad():
        for level in range(last_round):
            x_latent = x @ model._w(model.encoder)
            x_sparse = F.relu(x_latent)
            yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = model.ln(yKV)
            y_latent = yKV @ model._w(model.encoder_v)
            y_sparse = F.relu(y_latent)
            g = x_sparse * y_sparse
            xy_sparse = model.drop(g)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
            y = model.ln(yMLP)
            x = model.ln(x + y)

        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)
        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        g_real = x_sparse * y_sparse  # (B, nh, T, N)

    def finish(g: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            xy_sparse = model.drop(g)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
            y = model.ln(yMLP)
            x_final = model.ln(x + y)
            logits = x_final.view(B, T, D) @ model.lm_head
        return logits

    real_logits = finish(g_real)
    return g_real, real_logits, finish


def collect_g_and_logits(model, config, handle, batches, batch_size, seq_len, device, epochs):
    nh, D = config.n_head, config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    all_g = []
    all_real_logits = []
    finishers = []
    for _ in range(batches):
        data = read_batch(handle, batch_size, seq_len, device, epochs)
        idx = data[:, :-1].contiguous()
        g_real, real_logits, finish = run_to_last_round(model, idx)
        B, _, T, _ = g_real.shape
        g_flat = g_real.permute(0, 2, 1, 3).reshape(B * T, nh * N)  # (tokens, nh*N)
        all_g.append(g_flat)
        all_real_logits.append(real_logits.reshape(B * T, -1))
        finishers.append((finish, B, T, nh, N))
    return torch.cat(all_g, dim=0), torch.cat(all_real_logits, dim=0), finishers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--fit-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--ranks", type=str, default="16,32,64,128,256")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    config = BDHConfig(**payload["config"])
    model = BDH(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    epochs = [0]
    with args.validation_data.open() as handle:
        fit_g, _, _ = collect_g_and_logits(model, config, handle, args.fit_batches, args.batch_size, args.sequence_length, device, epochs)
        eval_g, eval_real_logits, eval_finishers = collect_g_and_logits(model, config, handle, args.eval_batches, args.batch_size, args.sequence_length, device, epochs)

    # Shared basis U via RANDOMIZED truncated SVD on the FIT sample only
    # (torch.svd_lowrank -- a full SVD of a (tokens, n_total) matrix is
    # far too slow on CPU at production scale; we only need components up
    # to the largest requested rank anyway, so a randomized truncated SVD
    # with a small oversampling buffer is both correct-enough and fast).
    fit_mean = fit_g.mean(dim=0, keepdim=True)
    ranks = [int(r) for r in args.ranks.split(",")]
    q = min(max(ranks) + 10, fit_g.shape[0], fit_g.shape[1])
    U_svd, S, V = torch.svd_lowrank(fit_g - fit_mean, q=q, niter=4)
    basis = V.T  # (q, n_total) -- rows are principal directions in neuron space
    results = []
    eval_g_centered = eval_g - fit_mean

    offset = 0
    for r in ranks:
        Ur = basis[:r]  # (r, n_total)
        alpha = eval_g_centered @ Ur.T  # (eval_tokens, r)
        eval_g_reconstructed = (alpha @ Ur) + fit_mean  # (eval_tokens, n_total)

        rel_error = (eval_g_reconstructed - eval_g).norm() / eval_g.norm().clamp(min=1e-8)

        # Downstream logit sensitivity: re-run the finish() closure with reconstructed g, per original batch.
        kl_divs = []
        argmax_matches = []
        cursor = 0
        for finish, B, T, nh, N in eval_finishers:
            n_tok = B * T
            g_chunk = eval_g_reconstructed[cursor:cursor + n_tok]
            g_chunk = g_chunk.view(B, T, nh, N).permute(0, 2, 1, 3)  # back to (B, nh, T, N)
            recon_logits = finish(g_chunk).reshape(n_tok, -1)
            real_logits_chunk = eval_real_logits[cursor:cursor + n_tok]

            log_p_real = F.log_softmax(real_logits_chunk, dim=-1)
            log_p_recon = F.log_softmax(recon_logits, dim=-1)
            kl = F.kl_div(log_p_recon, log_p_real, log_target=True, reduction="batchmean")
            kl_divs.append(float(kl))
            argmax_matches.append(float((real_logits_chunk.argmax(-1) == recon_logits.argmax(-1)).float().mean()))
            cursor += n_tok

        result = {
            "rank": r,
            "relative_reconstruction_error": float(rel_error),
            "mean_kl_divergence": sum(kl_divs) / len(kl_divs),
            "mean_argmax_agreement": sum(argmax_matches) / len(argmax_matches),
        }
        results.append(result)
        print(f"[rank={r}] rel_error={result['relative_reconstruction_error']:.4f} "
              f"KL={result['mean_kl_divergence']:.6f} argmax_agree={result['mean_argmax_agreement']:.4f}", flush=True)

    singular_value_energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    report = {
        "checkpoint": str(args.checkpoint),
        "n_total_neurons": fit_g.shape[1],
        "fit_tokens": fit_g.shape[0], "eval_tokens": eval_g.shape[0],
        "results_by_rank": results,
        "singular_value_cumulative_energy_at_ranks": {r: float(singular_value_energy[r - 1]) for r in ranks if r <= len(singular_value_energy)},
        "note": "explicitly reports BOTH raw reconstruction error AND downstream logit sensitivity (KL, argmax agreement) per the plan's own warning not to rely on SVD participation ratio alone. Basis computed via randomized truncated SVD (torch.svd_lowrank, q=max_rank+10) for tractability at production scale -- singular_value_cumulative_energy is normalized against the truncated spectrum's own total, not the true full-rank total, so it may read slightly optimistic; reconstruction_error and the logit-sensitivity metrics are unaffected by this and are the load-bearing numbers.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
