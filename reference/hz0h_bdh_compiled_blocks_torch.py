"""Persistent, physically packed BlockBDH execution.

Topology discovery happens outside the training hot loop. The resulting model
owns only selected contiguous RoPE-aligned columns, so normal training uses
regular dense matmuls without per-forward routing, index_select, or packing.
This is an architectural pruning path, not exact dense BDH; it is exact with
respect to the existing fixed-mask BlockBDH oracle for the same block order.
"""
from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from reference.hz0h_bdh_torch import Attention, BDH


@dataclasses.dataclass(frozen=True)
class CompiledBlockLayout:
    block_size: int
    active_fraction: float
    block_indices: tuple[int, ...]
    importance: tuple[float, ...]
    ordering_method: str = "weighted_reverse_cuthill_mckee"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _validate_layout(model: BDH, block_indices: torch.Tensor, block_size: int) -> None:
    if block_size <= 0 or block_size % 2:
        raise ValueError("block_size must be a positive even number to preserve RoPE pairs")
    config = model.config
    width = config.n_embd * config.mlp_internal_dim_multiplier // config.n_head
    if width % block_size:
        raise ValueError("latent width must be divisible by block_size")
    if block_indices.ndim != 1 or block_indices.numel() == 0:
        raise ValueError("block_indices must be a non-empty rank-1 tensor")
    if block_indices.unique().numel() != block_indices.numel():
        raise ValueError("block_indices must be unique")
    if int(block_indices.min()) < 0 or int(block_indices.max()) >= width // block_size:
        raise ValueError("block index is outside the latent width")
    active_width = block_indices.numel() * block_size
    if active_width * config.n_head % config.n_embd:
        raise ValueError("packed width must map to an integral BDH multiplier")


def _column_indices(block_indices: torch.Tensor, block_size: int) -> torch.Tensor:
    offsets = torch.arange(block_size, device=block_indices.device)
    return (block_indices[:, None] * block_size + offsets).reshape(-1)


def weighted_reverse_cuthill_mckee(coactivation: torch.Tensor) -> torch.Tensor:
    """Deterministic weighted RCM over a small dense block graph.

    Edges below the off-diagonal median are omitted. RCM only determines the
    permanent physical order; it does not itself remove computation.
    """
    if coactivation.ndim != 2 or coactivation.shape[0] != coactivation.shape[1]:
        raise ValueError("coactivation must be a square matrix")
    size = coactivation.shape[0]
    if size == 0:
        return torch.empty(0, dtype=torch.long)
    weights = coactivation.detach().to(device="cpu", dtype=torch.float64).clone()
    weights.fill_diagonal_(0.0)
    off_diagonal = weights[~torch.eye(size, dtype=torch.bool)]
    positive = off_diagonal[off_diagonal > 0]
    threshold = float(torch.quantile(positive, 0.5)) if positive.numel() else float("inf")
    adjacency = weights >= threshold
    adjacency.fill_diagonal_(False)
    degrees = adjacency.sum(dim=1).tolist()

    visited = [False] * size
    cuthill_mckee = []
    while len(cuthill_mckee) < size:
        start = min(
            (index for index in range(size) if not visited[index]),
            key=lambda index: (degrees[index], index),
        )
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            cuthill_mckee.append(node)
            neighbors = [
                index
                for index in range(size)
                if bool(adjacency[node, index]) and not visited[index]
            ]
            neighbors.sort(
                key=lambda index: (degrees[index], -float(weights[node, index]), index)
            )
            for neighbor in neighbors:
                visited[neighbor] = True
                queue.append(neighbor)
    return torch.tensor(list(reversed(cuthill_mckee)), dtype=torch.long)


@torch.no_grad()
def calibrate_compiled_block_layout(
    model: BDH,
    batches: Iterable[torch.Tensor],
    *,
    block_size: int,
    active_fraction: float,
) -> CompiledBlockLayout:
    """Measure dense shared-round block activity and choose a fixed layout."""
    if not 0 < active_fraction <= 1:
        raise ValueError("active_fraction must be in (0, 1]")
    config = model.config
    width = config.n_embd * config.mlp_internal_dim_multiplier // config.n_head
    probe = torch.arange(width // block_size, device=model.encoder.device)
    _validate_layout(model, probe, block_size)
    n_blocks = width // block_size
    importance = torch.zeros(n_blocks, device=model.encoder.device, dtype=torch.float64)
    coactivation = torch.zeros(
        n_blocks, n_blocks, device=model.encoder.device, dtype=torch.float64
    )
    sample_count = 0
    batch_count = 0
    was_training = model.training
    model.eval()
    try:
        for idx in batches:
            batch_count += 1
            x = model.ln(model.embed(idx).unsqueeze(1))
            batch_size, sequence_length = idx.shape
            for _ in range(config.n_layer):
                x_sparse = F.relu(x @ model._w(model.encoder))
                activity = x_sparse.reshape(
                    batch_size,
                    config.n_head,
                    sequence_length,
                    n_blocks,
                    block_size,
                ).abs().mean(dim=(1, 4))
                flat = activity.reshape(-1, n_blocks).to(torch.float64)
                importance += flat.sum(dim=0)
                coactivation += flat.mT @ flat
                sample_count += flat.shape[0]

                attended = model.ln(model.attn(x_sparse, x_sparse, x))
                y_sparse = F.relu(attended @ model._w(model.encoder_v))
                gated = model.drop(x_sparse * y_sparse)
                projected = (
                    gated.transpose(1, 2).reshape(
                        batch_size, 1, sequence_length, width * config.n_head
                    )
                    @ model._w(model.decoder)
                )
                x = model.ln(x + model.ln(projected))
    finally:
        model.train(was_training)
    if batch_count == 0 or sample_count == 0:
        raise ValueError("at least one non-empty calibration batch is required")

    importance /= sample_count
    coactivation /= sample_count
    active_count = max(1, round(n_blocks * active_fraction))
    selected = torch.argsort(importance, descending=True, stable=True)[:active_count]
    selected_coactivation = coactivation.index_select(0, selected).index_select(1, selected)
    local_order = weighted_reverse_cuthill_mckee(selected_coactivation)
    ordered = selected.index_select(0, local_order.to(selected.device)).to("cpu")
    return CompiledBlockLayout(
        block_size=block_size,
        active_fraction=active_fraction,
        block_indices=tuple(int(value) for value in ordered),
        importance=tuple(float(value) for value in importance.to("cpu")),
    )


class PackedBlockBDH(nn.Module):
    """Trainable fixed-topology BDH with physically compact parameters."""

    def __init__(
        self,
        source: BDH,
        block_indices: torch.Tensor,
        *,
        block_size: int,
    ) -> None:
        super().__init__()
        if source.config.ternary:
            raise ValueError("packed ternary parameters require a separate STE contract")
        block_indices = block_indices.detach().to(
            device=source.encoder.device, dtype=torch.long
        )
        _validate_layout(source, block_indices, block_size)
        columns = _column_indices(block_indices, block_size)
        source_config = source.config
        source_width = (
            source_config.n_embd
            * source_config.mlp_internal_dim_multiplier
            // source_config.n_head
        )
        active_width = columns.numel()
        packed_multiplier = active_width * source_config.n_head // source_config.n_embd
        self.config = dataclasses.replace(
            source_config,
            mlp_internal_dim_multiplier=packed_multiplier,
            ternary=False,
        )
        self.block_size = block_size
        self.register_buffer("packed_block_indices", block_indices.clone().contiguous())
        self.register_buffer("packed_column_indices", columns.clone().contiguous())

        self.encoder = nn.Parameter(
            source.encoder.detach().index_select(2, columns).clone().contiguous()
        )
        self.encoder_v = nn.Parameter(
            source.encoder_v.detach().index_select(2, columns).clone().contiguous()
        )
        decoder = source.decoder.detach().reshape(
            source_config.n_head, source_width, source_config.n_embd
        )
        self.decoder = nn.Parameter(
            decoder.index_select(1, columns)
            .reshape(source_config.n_head * active_width, source_config.n_embd)
            .clone()
            .contiguous()
        )
        self.attn = Attention(self.config)
        self.attn.freqs = (
            source.attn.freqs.detach().index_select(-1, columns).clone().contiguous()
        )
        self.ln = nn.LayerNorm(
            source_config.n_embd, elementwise_affine=False, bias=False
        )
        self.embed = nn.Embedding(source_config.vocab_size, source_config.n_embd)
        self.embed.weight = nn.Parameter(source.embed.weight.detach().clone().contiguous())
        self.drop = nn.Dropout(source_config.dropout)
        self.lm_head = nn.Parameter(source.lm_head.detach().clone().contiguous())

    @classmethod
    def from_layout(cls, source: BDH, layout: CompiledBlockLayout) -> "PackedBlockBDH":
        return cls(
            source,
            torch.tensor(layout.block_indices, device=source.encoder.device),
            block_size=layout.block_size,
        )

    @staticmethod
    def _w(parameter: torch.Tensor) -> torch.Tensor:
        return parameter

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        config = self.config
        batch_size, sequence_length = idx.shape
        dim = config.n_embd
        heads = config.n_head
        width = dim * config.mlp_internal_dim_multiplier // heads
        x = self.ln(self.embed(idx).unsqueeze(1))
        for _ in range(config.n_layer):
            x_sparse = F.relu(x @ self.encoder)
            attended = self.ln(self.attn(x_sparse, x_sparse, x))
            y_sparse = F.relu(attended @ self.encoder_v)
            gated = self.drop(x_sparse * y_sparse)
            projected = (
                gated.transpose(1, 2).reshape(
                    batch_size, 1, sequence_length, width * heads
                )
                @ self.decoder
            )
            x = self.ln(x + self.ln(projected))
        logits = x.view(batch_size, sequence_length, dim) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss
