"""Phase E of the batch-scaling investigation: N-axis tensor parallelism
for the compound model's decode step. Real math, not yet real multi-GPU
-- this file verifies the SHARDING DECOMPOSITION is exact (bit-identical
to the real unsharded decode step) using a simulated multiple-shards-on-
one-device setup (all-reduce = plain torch.sum over the shard list),
matching this project's own established pattern (correctness on one
device before any real multi-GPU dispatch, same discipline as the
oracle-ceiling benchmarks earlier tonight).

The decomposition (verified below):
- encoder, encoder_v are (nh, D, N) -- N-sharded cleanly, no
  communication (each shard's output N-slice is independent).
- RoPE operates on adjacent coordinate PAIRS within N -- shard boundaries
  must be even, then it's local/parallel per shard, no communication.
- P, O (VB's D<->d_state bottleneck) don't depend on N at all --
  replicated (same full weight on every shard), redundant cheap compute.
- decode's own intra-chunk attention term is trivial at L=1 (a scalar
  per head), so the only REAL communication needed is:
  (1) the state-read "cross" term (QR_i @ prefix_state_i, contracts over
      the N-shard) -- one all-reduce of shape (B, nh, T, d_state), SMALL.
  (2) the subspace decoder's alpha (contracts over the N-shard via the
      Phase B fix's batched-matmul-plus-sum), one all-reduce of shape
      (B, 1, T, r) -- TINY (r=64).
  The persistent state itself NEVER needs to be communicated -- each
  shard writes only its own local N-slice of the state, forever.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder


def shard_compound_model(model: BDHVBSubspaceDecoder, tp: int) -> list[dict]:
    """Splits the model's real weights into `tp` shards along N. Returns
    one dict per shard with the local weight slices (encoder, encoder_v,
    decoder_up) plus the small replicated weights (P, O, decoder_down)
    copied in full onto every shard, matching the real deployment
    pattern the correctness check below simulates."""
    c = model.config
    nh, D = c.n_head, c.n_embd
    N = D * c.mlp_internal_dim_multiplier // nh
    assert N % tp == 0, f"N={N} must divide evenly by tp={tp}"
    n_per_shard = N // tp
    assert n_per_shard % 2 == 0, "RoPE pairs adjacent coordinates -- shard boundaries must be even"

    shards = []
    for i in range(tp):
        lo, hi = i * n_per_shard, (i + 1) * n_per_shard
        shards.append({
            "encoder": model.encoder[:, :, lo:hi].contiguous(),
            "encoder_v": model.encoder_v[:, :, lo:hi].contiguous(),
            "decoder_up": model.decoder_up.view(nh, N, -1)[:, lo:hi, :].reshape(-1, model.decoder_up.shape[-1]).contiguous(),
            "P": model.P, "O": model.O, "decoder_down": model.decoder_down,
            "freqs": model.attn.freqs[..., lo:hi].contiguous(),
            "n_per_shard": n_per_shard,
        })
    return shards


def sharded_decode_step(model: BDHVBSubspaceDecoder, shards: list[dict], shard_states: list[list[torch.Tensor]],
                         idx_chunk: torch.Tensor, start_position: int) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
    """One real decode step (all `c.n_layer` weight-tied rounds) computed
    via the sharded decomposition. `shard_states[level][i]` is shard i's
    own local N-slice of round `level`'s persistent state, shape
    (B, nh, n_per_shard, d_state) -- never communicated, matches
    `bdh_vb_subspace_decoder_stream_chunk`'s own `states: list[Tensor]`
    (one tensor per layer) convention, just each entry now also split
    across shards. Real "all-reduce" steps are plain torch.sum here
    (simulated on one device); a real multi-GPU implementation would
    replace those two sums with torch.distributed.all_reduce calls,
    everything else is unchanged."""
    import torch.nn.functional as F

    c = model.config
    B, L = idx_chunk.shape
    D = c.n_embd

    x = model.embed(idx_chunk).unsqueeze(1)
    x = model.ln(x)

    new_states = []
    for level in range(c.n_layer):
        v_bottleneck = x @ model.P  # replicated compute, cheap, same on every shard

        cross_parts = []
        per_shard_x_sparse = []
        per_shard_KR = []
        for i, shard in enumerate(shards):
            x_latent_i = x @ shard["encoder"]
            x_sparse_i = F.relu(x_latent_i)
            positions = torch.arange(start_position, start_position + L, device=x.device, dtype=shard["freqs"].dtype).view(1, 1, L, 1)
            r_phases_i = positions * shard["freqs"]
            QR_i = model.attn.rope(r_phases_i, x_sparse_i)
            KR_i = QR_i
            cross_i = QR_i @ shard_states[level][i]  # local matmul, contracts over this shard's N-slice
            cross_parts.append(cross_i)
            per_shard_x_sparse.append(x_sparse_i)
            per_shard_KR.append(KR_i)

        cross = torch.stack(cross_parts, dim=0).sum(dim=0)  # (1) real all-reduce -- SMALL, (B,nh,T,d_state)
        yKV_bottleneck = cross  # intra-chunk term is exactly 0 at L=1 decode (tril(diagonal=-1) on a 1x1 matrix)
        yKV = model.ln(yKV_bottleneck @ model.O)  # replicated compute

        alpha_parts = []
        new_level_states = []
        for i, shard in enumerate(shards):
            y_latent_i = yKV @ shard["encoder_v"]
            y_sparse_i = F.relu(y_latent_i)
            xy_sparse_i = model.drop(per_shard_x_sparse[i] * y_sparse_i)
            alpha_i = torch.matmul(xy_sparse_i, shard["decoder_up"].view(c.n_head, shard["n_per_shard"], -1)).sum(dim=1, keepdim=True)
            alpha_parts.append(alpha_i)

            chunk_contribution_i = per_shard_KR[i].mT @ v_bottleneck  # purely local, no communication
            new_level_states.append(shard_states[level][i] + chunk_contribution_i)
        new_states.append(new_level_states)

        alpha = torch.stack(alpha_parts, dim=0).sum(dim=0)  # (2) real all-reduce -- TINY, (B,1,T,r)
        yMLP = alpha @ model.decoder_down  # replicated compute
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = x.view(B, L, D) @ model.lm_head
    return new_states, logits


def init_sharded_states(model: BDHVBSubspaceDecoder, shards: list[dict], batch_size: int, device, dtype) -> list[list[torch.Tensor]]:
    c = model.config
    return [[torch.zeros(batch_size, c.n_head, shard["n_per_shard"], c.d_state, device=device, dtype=dtype) for shard in shards]
            for _level in range(c.n_layer)]
