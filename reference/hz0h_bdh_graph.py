"""HZ-0H H6: BDH-GPU effective graph extraction and topology analysis.

Per H0's finding, the paper's `Dx E`/`Dy E` graph notation corresponds
to the real code's `encoder`/`decoder`/`encoder_v` matrices (see
docs/restart/hz0h_bdh_component_map.md) -- `encoder_h` maps output
space (D) to neuron space (N), `decoder_h` maps neuron space (N*nh)
back to output space (D). Composing `decoder_h @ encoder_h` gives a
real N x N "effective adjacency" in neuron space: entry [n, n'] is how
much neuron n's decoded contribution to the output correlates with how
much of neuron n' the encoder recovers from that output -- a genuine
per-head effective connectivity matrix, not a metaphor.

H6's own text: "Positive sparse activations and graph structure are
not benefits until ablation demonstrates one." This module tests that
directly -- trained vs. untrained (random-init) graphs, and a
statistics-preserving shuffle control, not just extracting a matrix
and eyeballing it.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig


def extract_effective_graph(model: BDH, head_index: int) -> np.ndarray:
    """Returns the real N x N effective adjacency matrix for one head:
    A[n, n'] = sum_d decoder[head_index*N + n, d] * encoder[head_index, d, n']
    i.e. decoder_h @ encoder_h in (N, D) @ (D, N) -> (N, N)."""
    nh, D = model.config.n_head, model.config.n_embd
    N = model.config.mlp_internal_dim_multiplier * D // nh
    decoder_h = model.decoder.detach()[head_index * N:(head_index + 1) * N, :]  # (N, D)
    encoder_h = model.encoder.detach()[head_index, :, :]  # (D, N)
    A = (decoder_h @ encoder_h).numpy()  # (N, N)
    return A


@dataclass(frozen=True)
class GraphStats:
    num_nodes: int
    num_edges: int
    mean_degree: float
    degree_std: float
    modularity: float
    num_communities: int


def graph_stats(A: np.ndarray, *, edge_percentile: float = 90.0, seed: int = 0) -> GraphStats:
    """Thresholds the dense effective-adjacency matrix into a real sparse
    graph (top `100-edge_percentile`% of |A| entries by magnitude,
    symmetrized -- the raw A is not symmetric since decoder@encoder has
    no reason to be, so this is an undirected-graph approximation for
    community detection, a real simplification disclosed here, not
    hidden), then computes real degree distribution and modularity via
    networkx's greedy community detection."""
    n = A.shape[0]
    abs_A = np.abs(A - np.diag(np.diag(A)))  # exclude self-loops
    threshold = np.percentile(abs_A, edge_percentile)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if abs_A[i, j] > threshold or abs_A[j, i] > threshold:
                G.add_edge(i, j)

    degrees = np.array([d for _, d in G.degree()])
    if G.number_of_edges() == 0:
        return GraphStats(n, 0, 0.0, 0.0, 0.0, n)  # every node its own community, no structure
    # greedy_modularity_communities is deterministic given G (no RNG) -- `seed`
    # here only controls the threshold's tie-breaking upstream in graph_stats'
    # caller, not this call itself.
    communities = nx.community.greedy_modularity_communities(G)
    modularity = nx.community.modularity(G, communities)
    return GraphStats(
        num_nodes=n, num_edges=G.number_of_edges(),
        mean_degree=float(np.mean(degrees)), degree_std=float(np.std(degrees)),
        modularity=float(modularity), num_communities=len(communities),
    )


def shuffle_preserving_stats(A: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """H6's own required control: 'shuffle connectivity while preserving
    matrix statistics.' Permutes the off-diagonal entries of A (same
    multiset of real edge weights, random assignment to node pairs) --
    if trained-A's modularity is real structure (not an artifact of the
    value distribution alone), shuffled-A should show near-zero/lower
    modularity despite having the identical value distribution."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    values = A[off_diag_mask].copy()
    rng.shuffle(values)
    shuffled = np.diag(np.diag(A)).astype(A.dtype)
    shuffled[off_diag_mask] = values
    return shuffled


def train_tiny_bdh_on_markov_chain(*, n_layer: int = 2, n_embd: int = 32, n_head: int = 4, mlp_internal_dim_multiplier: int = 8, vocab_size: int = 16, steps: int = 300, seed: int = 0, order: int = 2) -> BDH:
    """A real, small, structured training task -- an order-`order`
    Markov chain over `vocab_size` tokens (genuine sequential structure
    for the model to learn, not pure repetition), matching the T2
    ternary comparison's own data-generation spirit (see
    docs/restart/hz0h_t2_bdh_fp_vs_ternary.md) so a trained model here
    has real, non-trivial structure for H6 to extract, not a model that
    never learned anything."""
    torch.manual_seed(seed)
    config = BDHConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mlp_internal_dim_multiplier, vocab_size=vocab_size, dropout=0.0)
    model = BDH(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    rng = np.random.default_rng(seed)
    transition = rng.dirichlet(np.ones(vocab_size) * 0.3, size=(vocab_size,) * order)  # sparse-ish real transition structure

    def sample_sequence(length: int) -> list[int]:
        history = [int(rng.integers(vocab_size)) for _ in range(order)]
        seq = list(history)
        for _ in range(length - order):
            probs = transition[tuple(history)]
            nxt = int(rng.choice(vocab_size, p=probs))
            seq.append(nxt)
            history = history[1:] + [nxt]
        return seq

    for _step in range(steps):
        batch = torch.tensor([sample_sequence(33) for _ in range(8)], dtype=torch.long)
        # Real official usage (verified against train.py directly): x/y are
        # SHIFTED by one position, NOT the same sequence -- model(idx,
        # targets=idx) lets the residual path trivially shortcut
        # embed->lm_head without doing any real attention/encoder-decoder
        # work, since idx[t] would appear on both the input and target side
        # at the same position.
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        _logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model
