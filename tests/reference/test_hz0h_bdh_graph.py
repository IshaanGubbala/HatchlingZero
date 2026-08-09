"""HZ-0H H6: effective graph extraction and topology tests."""
from __future__ import annotations

import numpy as np

from reference.hz0h_bdh_graph import extract_effective_graph, graph_stats, shuffle_preserving_stats, train_tiny_bdh_on_markov_chain
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _tiny_config() -> BDHConfig:
    return BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=16, dropout=0.0)


def test_extract_effective_graph_shape():
    model = BDH(_tiny_config())
    N = _tiny_config().mlp_internal_dim_multiplier * _tiny_config().n_embd // _tiny_config().n_head
    A = extract_effective_graph(model, head_index=0)
    assert A.shape == (N, N)
    assert np.all(np.isfinite(A))


def test_extract_effective_graph_differs_per_head():
    """Real sanity check: different heads have independently-initialized
    encoder/decoder slices, so their effective graphs must differ --
    catches an indexing bug that accidentally extracted the same head
    twice."""
    model = BDH(_tiny_config())
    A0 = extract_effective_graph(model, head_index=0)
    A1 = extract_effective_graph(model, head_index=1)
    assert not np.allclose(A0, A1)


def test_shuffle_preserves_value_distribution():
    model = BDH(_tiny_config())
    A = extract_effective_graph(model, head_index=0)
    shuffled = shuffle_preserving_stats(A, seed=0)
    n = A.shape[0]
    off_diag = ~np.eye(n, dtype=bool)
    assert np.allclose(sorted(A[off_diag]), sorted(shuffled[off_diag]))
    assert np.allclose(np.diag(A), np.diag(shuffled))  # diagonal untouched
    assert not np.allclose(A, shuffled)  # but the assignment to node pairs really changed


def test_shuffle_is_deterministic_given_seed():
    model = BDH(_tiny_config())
    A = extract_effective_graph(model, head_index=0)
    s1 = shuffle_preserving_stats(A, seed=7)
    s2 = shuffle_preserving_stats(A, seed=7)
    assert np.array_equal(s1, s2)


def test_graph_stats_finite_and_real_on_random_matrix():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(64, 64))
    stats = graph_stats(A, edge_percentile=90.0)
    assert stats.num_nodes == 64
    assert stats.num_edges > 0
    assert 0.0 <= stats.modularity <= 1.0
    assert stats.num_communities >= 1


def test_trained_vs_untrained_vs_shuffled_modularity_real_and_reproducible():
    """The real H6 finding, locked in as a regression: at this tiny
    scale (n_embd=32, n_head=4, mlp_internal_dim_multiplier=8 -- similar
    order of magnitude to T2's 819K-param setup), trained modularity
    does NOT exceed the statistics-preserving shuffled control across 3
    seeds. A real, reproducible negative result at this scale -- NOT
    a claim that BDH has no graph structure at any scale (the plan's own
    caution: don't reject from a tiny toy run when the paper's evidence
    spans 10M-1B params). This test locks in the reproducibility of the
    negative result itself, not the broader claim."""
    results = []
    for seed in (0, 1, 2):
        model = train_tiny_bdh_on_markov_chain(steps=300, seed=seed)
        A = extract_effective_graph(model, head_index=0)
        trained_stats = graph_stats(A, edge_percentile=90.0)
        shuffled_stats = graph_stats(shuffle_preserving_stats(A, seed=seed), edge_percentile=90.0)
        results.append((trained_stats.modularity, shuffled_stats.modularity))

    for trained_mod, shuffled_mod in results:
        assert trained_mod == trained_mod and shuffled_mod == shuffled_mod  # finite (NaN check)
    # The real, reproducible finding: trained doesn't systematically beat shuffled by a wide margin.
    mean_gap = sum(t - s for t, s in results) / len(results)
    assert abs(mean_gap) < 0.1, f"if this fails, real graph structure may have emerged -- verify before assuming a bug: {results}"
