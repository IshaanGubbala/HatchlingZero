#!/usr/bin/env python3
"""Phase D of the batch-scaling investigation (user's own proposed order):
"do NOT build the architecture first -- run the same kind of diagnostic
that discovered the decoder subspace win." Key-State Subspace BDH
proposes compressing the attention's N-axis (currently 4992) by
projecting the RoPE-transformed Q=K=x_sparse vectors through a shared
dense basis U (N -> r_k) before they interact with the recurrent state
-- unlike VB (compresses the value/state WIDTH, d_state) or the subspace
decoder (compresses the decoder's rank), this would compress the
persistent state's OTHER large axis (N), the one VB and the subspace
decoder both left untouched. If real, this is the piece that could turn
B4-class serving economics into B64-256-class economics (per the user's
own math: N-axis compression multiplies with d_state compression rather
than substituting for it).

This diagnostic does NOT touch the recurrent state or retrain anything
-- it fits a real SVD basis on REAL RoPE-transformed Q/K vectors
collected from the exact-BDH baseline checkpoint
(results/local/hz0h_bdh_checkpoint_for_ablation.pt, the same real
production-trained model used for every other diagnostic tonight),
then measures REAL downstream logit sensitivity (KL divergence, top-1
argmax agreement against the model's own true logits) of substituting
the attention score computation `QR @ KR.mT` with the low-rank-projected
`(QR @ U) @ (KR @ U).mT` at EVERY layer/round -- not just the last round
(unlike item 23's gate-reconstruction diagnostic, which only touched
the final round's decoder input). Fit and eval use SEPARATE samples,
same discipline as item 23.
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


def bdh_forward_with_qk_subspace(model: BDH, idx: torch.Tensor, U: torch.Tensor | None) -> torch.Tensor:
    """Byte-for-byte BDH.forward, except the attention score computation
    `QR @ KR.mT` is replaced with `(QR @ U) @ (KR @ U).mT` at every round
    if `U` is given (shape (N, r_k)); `U=None` reproduces the exact
    dense baseline (used to sanity-check this function against the real
    model.forward before trusting any substituted-U result)."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)

    for _level in range(C.n_layer):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)

        T_ = x_sparse.shape[2]
        r_phases = (torch.arange(0, T_, device=x.device, dtype=model.attn.freqs.dtype).view(1, 1, -1, 1)) * model.attn.freqs
        QR = model.attn.rope(r_phases, x_sparse)
        KR = QR
        if U is not None:
            scores = ((QR @ U) @ (KR @ U).mT).tril(diagonal=-1)
        else:
            scores = (QR @ KR.mT).tril(diagonal=-1)
        yKV = scores @ x
        yKV = model.ln(yKV)

        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, T, D) @ model.lm_head
    return logits


def collect_qr_vectors(model: BDH, data_path: Path, n_batches: int, batch_size: int, seq_len: int, device) -> torch.Tensor:
    """Real RoPE-transformed QR=KR vectors, pooled across every layer/round
    and every token, from real forward passes -- the fit sample for the
    shared basis U."""
    C = model.config
    nh, D = C.n_head, C.n_embd
    N = D * C.mlp_internal_dim_multiplier // nh
    all_qr = []
    epochs = [0]
    with data_path.open() as handle, torch.no_grad():
        for _ in range(n_batches):
            data = read_batch(handle, batch_size, seq_len, device, epochs)
            idx = data[:, :-1].contiguous()
            x = model.embed(idx).unsqueeze(1)
            x = model.ln(x)
            for _level in range(C.n_layer):
                x_latent = x @ model._w(model.encoder)
                x_sparse = F.relu(x_latent)
                T_ = x_sparse.shape[2]
                r_phases = (torch.arange(0, T_, device=device, dtype=model.attn.freqs.dtype).view(1, 1, -1, 1)) * model.attn.freqs
                QR = model.attn.rope(r_phases, x_sparse)
                all_qr.append(QR.reshape(-1, N).float().cpu())
                scores = (QR @ QR.mT).tril(diagonal=-1)
                yKV = model.ln(scores @ x)
                y_latent = yKV @ model._w(model.encoder_v)
                y_sparse = F.relu(y_latent)
                xy_sparse = model.drop(x_sparse * y_sparse)
                B_, T_full = idx.shape
                yMLP = xy_sparse.transpose(1, 2).reshape(B_, 1, T_full, N * nh) @ model._w(model.decoder)
                x = model.ln(x + model.ln(yMLP))
    return torch.cat(all_qr, dim=0)


def collect_eval_batch(data_path: Path, batch_size: int, seq_len: int, device) -> torch.Tensor:
    epochs = [0]
    with data_path.open() as handle:
        return read_batch(handle, batch_size, seq_len, device, epochs)[:, :-1].contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/local/hz0h_bdh_checkpoint_for_ablation.pt"))
    parser.add_argument("--fit-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--eval-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fit-batches", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = BDHConfig(**ckpt["config"])
    model = BDH(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    N = config.n_embd * config.mlp_internal_dim_multiplier // config.n_head

    # Sanity check: U=None must reproduce the real model.forward exactly
    # before trusting any substituted-U result.
    sanity_idx = collect_eval_batch(args.eval_data, 2, 32, device)
    with torch.no_grad():
        real_logits, _ = model(sanity_idx)
        replicated_logits = bdh_forward_with_qk_subspace(model, sanity_idx, U=None)
    sanity_diff = float((real_logits - replicated_logits).abs().max())
    print(f"[sanity] bdh_forward_with_qk_subspace(U=None) vs real model.forward max diff (expect 0.0): {sanity_diff}", flush=True)
    assert sanity_diff < 1e-4, "custom forward does not reproduce the real model -- fix before trusting substituted-U results"

    print(f"[fit] collecting real QR vectors from {args.fit_batches} batches...", flush=True)
    fit_qr = collect_qr_vectors(model, args.fit_data, args.fit_batches, args.batch_size, args.sequence_length, device)
    print(f"[fit] collected {fit_qr.shape[0]} real (layer, token) QR vectors, N={N}", flush=True)

    fit_mean = fit_qr.mean(dim=0, keepdim=True)
    max_rank = max(args.ranks)
    print(f"[fit] running torch.svd_lowrank(q={max_rank + 10})...", flush=True)
    U_full, S_full, V_full = torch.svd_lowrank(fit_qr - fit_mean, q=max_rank + 10, niter=4)
    # V_full: (N, max_rank+10) -- the real shared basis directions, ordered by singular value.

    eval_idx = collect_eval_batch(args.eval_data, args.batch_size, args.sequence_length, device)
    with torch.no_grad():
        true_logits, _ = model(eval_idx)
        true_probs = F.softmax(true_logits.float(), dim=-1)
        true_argmax = true_logits.argmax(dim=-1)

    results = {
        "checkpoint": str(args.checkpoint), "N": N,
        "note": "real downstream-logit sensitivity of substituting QR@KR.mT with (QR@U)@(KR@U).mT at EVERY layer/round, U a shared SVD basis fit on real RoPE-transformed Q=K vectors. Fit and eval use separate samples.",
        "singular_value_cumulative_energy": {},
        "by_rank": {},
    }
    total_energy = float((S_full ** 2).sum())

    for rank in args.ranks:
        U = V_full[:, :rank].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            sub_logits = bdh_forward_with_qk_subspace(model, eval_idx, U=U)
            sub_probs = F.softmax(sub_logits.float(), dim=-1)
            kl = F.kl_div(sub_probs.clamp_min(1e-12).log(), true_probs, reduction="batchmean")
            sub_argmax = sub_logits.argmax(dim=-1)
            argmax_agreement = float((sub_argmax == true_argmax).float().mean())

        energy = float((S_full[:rank] ** 2).sum() / total_energy)
        results["singular_value_cumulative_energy"][str(rank)] = energy
        results["by_rank"][str(rank)] = {
            "mean_kl_divergence": float(kl),
            "mean_argmax_agreement": argmax_agreement,
        }
        print(f"[rank={rank:5d}] cum_energy={energy:.4f} KL={float(kl):.4f} argmax_agreement={argmax_agreement:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
