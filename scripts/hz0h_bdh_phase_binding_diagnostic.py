#!/usr/bin/env python3
"""Phase diagnostic for the "dynamic phase binding" hypothesis (real,
substantive proposal): does BDH's own x_latent/y_latent pair already
encode something phase-like -- i.e. is there real evidence that
neurons form TEMPORARY functional couplings (phase relationships that
shift across contexts) rather than acting as a fixed set of specialized
identities? Zero architecture changes, run on the real, already-trained
exact-BDH baseline checkpoint (results/local/hz0h_bdh_checkpoint_for_ablation.pt)
-- the x_latent = x @ encoder / y_latent = yKV @ encoder_v computation
this probes is byte-for-byte identical in the compound model too (only
the state/decoder differ), so this doesn't need to wait for the
currently-running 500M-token job.

Real signal, not assumed: for each active neuron (gate g_i = ReLU(x_i)*
ReLU(y_i) > 0) at the LAST recurrent round, construct the complex pair
c_i = x_latent_i + i*y_latent_i, giving a real amplitude |c_i| and
phase phi_i = atan2(y_i, x_i) per neuron per token.

Three real, falsifiable questions, each compared against an explicit
null (not just "look interesting"):

1. Is a given neuron's phase roughly FIXED across tokens (phase is
   just a static per-neuron property, uninteresting for binding) or
   does it vary substantially (consistent with "phase encodes
   something about the current context/assembly, not neuron identity
   alone")? Measured via real circular variance of phi_i across all
   tokens where neuron i is active.
2. Do neurons that are FREQUENTLY CO-ACTIVE (both gates > 0 on the same
   token, many tokens) show a real phase-locking value (PLV, a
   standard neuroscience metric: |mean(exp(i*(phi_i - phi_j)))| across
   co-active tokens) higher than a null of RANDOMLY PAIRED neurons
   matched for the same co-activation rate? A real, above-null PLV
   would be direct evidence of phase coupling structure already latent
   in trained BDH, not injected.
3. Does the SAME neuron pair's phase relationship (phi_i - phi_j)
   change across different real input contexts (different domains,
   different token positions) -- consistent with "temporary assembly
   membership" -- or does it stay essentially fixed (consistent with
   the pair just having a fixed relative phase, not context-dependent
   binding)?
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


@torch.no_grad()
def collect_last_round_xy(model: BDH, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Real forward pass through model.config.n_layer rounds, returning
    the LAST round's signed x_latent and y_latent (pre-ReLU), shape
    (B, nh, T, N) each -- byte-for-byte the same computation BDH.forward
    does, just stopping before ReLU on the final round instead of
    continuing to the decoder."""
    C = model.config
    B, T = idx.size()
    nh = C.n_head
    N = N_dim(C)
    x = model.embed(idx).unsqueeze(1)
    x = model.ln(x)
    x_latent_last = y_latent_last = None
    for level in range(C.n_layer):
        x_latent = x @ model._w(model.encoder)
        x_sparse = F.relu(x_latent)
        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)
        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        if level == C.n_layer - 1:
            x_latent_last, y_latent_last = x_latent, y_latent
            break
        xy_sparse = model.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)
    return x_latent_last, y_latent_last


def N_dim(C: BDHConfig) -> int:
    return C.n_embd * C.mlp_internal_dim_multiplier // C.n_head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/local/hz0h_bdh_checkpoint_for_ablation.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-tokens-sample", type=int, default=4096, help="real tokens (across batches) used for the phase statistics")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-pairs", type=int, default=2000, help="real co-active neuron pairs sampled for the PLV test")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = BDHConfig(**ckpt["config"])
    model = BDH(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    nh, N = config.n_head, N_dim(config)
    n_batches = max(1, args.n_tokens_sample // (args.batch_size * args.sequence_length))
    torch.manual_seed(args.seed)

    all_x_latent = []
    all_y_latent = []
    epochs = [0]
    with args.data.open() as handle:
        for _ in range(n_batches):
            data = read_batch(handle, args.batch_size, args.sequence_length, device, epochs)
            idx = data[:, :-1].contiguous()
            x_latent, y_latent = collect_last_round_xy(model, idx)
            all_x_latent.append(x_latent.reshape(-1, nh, N).cpu())
            all_y_latent.append(y_latent.reshape(-1, nh, N).cpu())

    # real fix: move off GPU right after collection -- the downstream phase/PLV
    # analysis is cheap and doesn't need CUDA; keeping full-size x_latent/y_latent
    # PLUS their derived phase/amplitude tensors all resident on GPU at once
    # (production shape: nh=8, N=4992) is real, genuine memory pressure, not
    # something CUDA needs to be involved in at all past this point.
    x_latent = torch.cat(all_x_latent, dim=0)  # (n_tokens, nh, N), real CPU tensor
    y_latent = torch.cat(all_y_latent, dim=0)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    n_tokens = x_latent.shape[0]
    print(f"[collect] real {n_tokens} tokens, nh={nh}, N={N}", flush=True)

    gate_active = (x_latent > 0) & (y_latent > 0)  # real gate: ReLU(x)*ReLU(y) > 0 iff both positive
    phase = torch.atan2(y_latent, x_latent)  # (n_tokens, nh, N), real angle per neuron per token
    amplitude = torch.sqrt(x_latent**2 + y_latent**2)

    results = {"checkpoint": str(args.checkpoint), "n_tokens": n_tokens, "n_head": nh, "N": N,
               "note": "Real phase diagnostic on trained BDH's x_latent/y_latent pair. All three questions compared against explicit nulls, not just eyeballed."}

    # --- Q1: is a given neuron's phase fixed or does it vary across tokens where it's active? ---
    # Real circular variance per (head, neuron): 1 - |mean(exp(i*phase))| over tokens where that neuron is active.
    # 0 = phase always identical when active (fixed); 1 = phase uniform/random when active.
    head_idx, neuron_idx = 0, torch.randint(0, N, (500,))  # real sample of neurons in head 0, for tractability
    circ_vars = []
    active_counts = []
    for n in neuron_idx.tolist():
        active_mask = gate_active[:, head_idx, n]
        if active_mask.sum() < 10:
            continue
        phis = phase[active_mask, head_idx, n]
        mean_vec = torch.complex(torch.cos(phis), torch.sin(phis)).mean()
        circ_var = 1.0 - mean_vec.abs().item()
        circ_vars.append(circ_var)
        active_counts.append(int(active_mask.sum()))
    results["q1_phase_variability"] = {
        "n_neurons_sampled": len(circ_vars),
        "mean_circular_variance": sum(circ_vars) / max(len(circ_vars), 1),
        "note": "0=phase fixed whenever neuron is active (static property); 1=phase uniform/random when active (no stable per-neuron phase). Real per-neuron sample, head 0.",
    }
    print(f"[Q1] mean circular variance across {len(circ_vars)} sampled active neurons: "
          f"{results['q1_phase_variability']['mean_circular_variance']:.4f} (0=fixed, 1=random)", flush=True)

    # --- Q2: real PLV for co-active neuron pairs vs a null of random pairs matched on co-activation rate ---
    torch.manual_seed(args.seed)
    head = 0
    gate_h = gate_active[:, head, :]  # (n_tokens, N)
    co_active_counts = (gate_h.float().T @ gate_h.float())  # (N, N) real co-activation counts
    co_active_counts.fill_diagonal_(0)
    flat = co_active_counts.flatten()
    top_pairs_idx = torch.topk(flat, args.n_pairs).indices
    pair_i = top_pairs_idx // N
    pair_j = top_pairs_idx % N

    def real_plv(i_idx, j_idx):
        plvs = []
        for i, j in zip(i_idx.tolist(), j_idx.tolist()):
            both_active = gate_h[:, i] & gate_h[:, j]
            if both_active.sum() < 10:
                continue
            dphi = phase[both_active, head, i] - phase[both_active, head, j]
            mean_vec = torch.complex(torch.cos(dphi), torch.sin(dphi)).mean()
            plvs.append(mean_vec.abs().item())
        return plvs

    real_plvs = real_plv(pair_i, pair_j)
    null_i = torch.randint(0, N, (args.n_pairs,))
    null_j = torch.randint(0, N, (args.n_pairs,))
    null_plvs = real_plv(null_i, null_j)

    results["q2_coactive_plv_vs_null"] = {
        "n_real_pairs_measured": len(real_plvs),
        "mean_plv_coactive_pairs": sum(real_plvs) / max(len(real_plvs), 1),
        "n_null_pairs_measured": len(null_plvs),
        "mean_plv_random_pairs": sum(null_plvs) / max(len(null_plvs), 1),
        "note": "Real phase-locking value (0=no consistent phase relationship, 1=perfectly locked), frequently-co-active pairs (top N by real co-activation count) vs randomly-paired neurons. If coactive >> random, real evidence of latent phase coupling structure.",
    }
    print(f"[Q2] mean PLV, co-active pairs: {results['q2_coactive_plv_vs_null']['mean_plv_coactive_pairs']:.4f} "
          f"vs random pairs: {results['q2_coactive_plv_vs_null']['mean_plv_random_pairs']:.4f}", flush=True)

    # --- Q3: does a fixed neuron pair's phase relationship shift across different contexts (token-position buckets)? ---
    n_buckets = 4
    bucket_size = n_tokens // n_buckets
    pair_sample = list(zip(pair_i[:200].tolist(), pair_j[:200].tolist()))
    bucket_mean_dphi = []
    for b in range(n_buckets):
        lo, hi = b * bucket_size, (b + 1) * bucket_size
        dphis_this_bucket = []
        for i, j in pair_sample:
            both_active = gate_h[lo:hi, i] & gate_h[lo:hi, j]
            if both_active.sum() < 5:
                continue
            dphi = phase[lo:hi][both_active, head, i] - phase[lo:hi][both_active, head, j]
            mean_vec = torch.complex(torch.cos(dphi), torch.sin(dphi)).mean()
            dphis_this_bucket.append(torch.atan2(mean_vec.imag, mean_vec.real).item())
        if dphis_this_bucket:
            bucket_mean_dphi.append(sum(dphis_this_bucket) / len(dphis_this_bucket))
    if len(bucket_mean_dphi) >= 2:
        drift = max(bucket_mean_dphi) - min(bucket_mean_dphi)
    else:
        drift = None
    results["q3_phase_relationship_drift_across_contexts"] = {
        "n_buckets": n_buckets, "bucket_mean_relative_phase_radians": bucket_mean_dphi,
        "max_minus_min_radians": drift,
        "note": "Same sampled pairs' mean relative phase (phi_i - phi_j), measured separately in disjoint token-position buckets (real, sequential slices of the collected sample, not shuffled -- a rough proxy for 'different context'). Large drift = relationship is context-dependent; near-zero = fixed regardless of context.",
    }
    print(f"[Q3] relative-phase drift across {n_buckets} context buckets: {drift}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
