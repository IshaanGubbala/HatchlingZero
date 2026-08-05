"""HZ-0E E2: isolated router simulator tests
(reference/hz0e_e2_router_simulator.py). Checked against the ACTUAL
frozen HZ-0A checkpoint and REAL corpus domain data, matching this
project's established convention. Skips if either is missing locally.
Locks in the real, measured findings from
`docs/restart/hz0e_e2_router_simulator_results.md` as regression
tests -- E2's own exit gate ("multiple experts remain active without
collapse") plus every named measurement (utilization, balance,
overflow, entropy, collapse, stability) across code/prose/math/JSON/
tools, mixed domains, imbalance, domain shifts, and noisy inputs.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0e_moe_contract import MoeConfig, init_moe_layer
from reference.hz0e_e2_router_simulator import DOMAIN_DATA_PATHS, collect_real_ffn_input, route_with_stats
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import load_real_sequences

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not all(Path(p).exists() for p in DOMAIN_DATA_PATHS.values()),
    reason="frozen HZ-0A checkpoint / real domain corpus files not present locally (gitignored)",
)

CONFIG = MoeConfig()
TARGET_LAYERS = (27, 28, 30)

# Real thresholds derived from a 300-configuration sweep (20 seeds x 5
# domains x 3 layers) reported in the results doc: observed max raw
# routing share ranged 0.277-0.574, observed min per-expert share never
# hit 0 (min ever seen: 0.0786), observed clean-data entropy ranged
# 1.76-1.86 bits (theoretical max 2.0). Bounds below keep real margin
# above/below every observed extreme, not set to just barely pass.
COLLAPSE_MAX_SHARE_BOUND = 0.75
MIN_ENTROPY_BITS_CLEAN = 1.5


def test_collect_real_ffn_input_matches_independent_manual_replay():
    """A real bug lived in `collect_real_ffn_input` during development
    (applied `norm2` to the block's INPUT instead of its post-mixer-
    residual state) -- caught by exactly this comparison, not by
    eyeballing output statistics (the buggy and correct versions have
    nearly identical std). Locks the fix in as a permanent regression
    test: `collect_real_ffn_input` must match a from-scratch manual
    replay of `reference/hz0a_mlx_model.py::Block.__call__`'s own
    control flow, bit-exactly."""
    model, _payload = load_frozen_model()
    tokens = mx.array([[1, 45, 982, 12, 7, 300, 44, 1023]])
    layer_index = 27

    x = model.embedding(tokens)
    for i in range(layer_index):
        x, _ = model.blocks[i](x, None)
    block = model.blocks[layer_index]
    mixed, _ = block.mixer(block.norm1(x), None)
    expected = block.norm2(x + mixed)
    mx.eval(expected)

    actual = collect_real_ffn_input(model, tokens, layer_index)
    mx.eval(actual)
    assert bool(mx.array_equal(actual, expected))


def _real_domain_x(model, domain: str, layer: int, n_seq: int = 8, max_len: int = 256) -> mx.array:
    seqs = load_real_sequences(DOMAIN_DATA_PATHS[domain], n_seq)
    min_len = min(min(len(s) for s in seqs), max_len)
    tokens = mx.array([s[:min_len] for s in seqs])
    x = collect_real_ffn_input(model, tokens, layer)
    mx.eval(x)
    return x


def test_all_five_real_domains_avoid_collapse_at_the_first_target_layer():
    """The plan's own named domain list (code, prose, math, JSON,
    tools -- matched to the closest real corpus file for each, see the
    module docstring), each checked directly: no single expert may
    dominate routing on ANY real domain's content."""
    model, _payload = load_frozen_model()
    params = init_moe_layer(CONFIG)
    for domain in DOMAIN_DATA_PATHS:
        x = _real_domain_x(model, domain, 27)
        out, diag, stats = route_with_stats(x, params, CONFIG)
        mx.eval(out)
        assert stats.max_expert_share < COLLAPSE_MAX_SHARE_BOUND, f"{domain}: max_expert_share={stats.max_expert_share}"
        assert stats.entropy_bits > MIN_ENTROPY_BITS_CLEAN, f"{domain}: entropy_bits={stats.entropy_bits}"
        assert all(u >= 0.0 for u in stats.utilization)


def test_no_dead_experts_across_a_multi_seed_multi_domain_multi_layer_sweep():
    """The real 'collapse' check, at scale: across 5 router-init seeds
    x 5 real domains x 3 target layers (75 real configurations, a
    reduced version of the results doc's own 300-config sweep, kept
    smaller here for test speed), no expert may EVER receive zero raw
    routing mass (a dead expert), and the single largest raw share must
    stay within the real, generously-margined collapse bound."""
    model, _payload = load_frozen_model()
    domain_x = {domain: {layer: _real_domain_x(model, domain, layer) for layer in TARGET_LAYERS} for domain in DOMAIN_DATA_PATHS}

    dead_events = 0
    max_share_ever = 0.0
    for seed in range(5):
        params = init_moe_layer(MoeConfig(init_seed=seed))
        for domain in DOMAIN_DATA_PATHS:
            for layer in TARGET_LAYERS:
                x = domain_x[domain][layer]
                _, diag, _ = route_with_stats(x, params, CONFIG)
                mx.eval(diag.expert_idx)
                n = diag.expert_idx.shape[0]
                counts = [int(mx.sum((diag.expert_idx == e).astype(mx.int32))) for e in range(CONFIG.num_experts)]
                if min(counts) == 0:
                    dead_events += 1
                max_share_ever = max(max_share_ever, max(counts) / n)

    assert dead_events == 0, f"{dead_events} dead-expert events across the sweep"
    assert max_share_ever < COLLAPSE_MAX_SHARE_BOUND, f"max observed raw share {max_share_ever}"


def test_mixed_domain_batch_avoids_collapse():
    """Different real domains as different rows in the SAME batch
    (code + math sequences together) -- must not collapse."""
    model, _payload = load_frozen_model()
    params = init_moe_layer(CONFIG)
    code_seqs = load_real_sequences(DOMAIN_DATA_PATHS["code"], 4)
    math_seqs = load_real_sequences(DOMAIN_DATA_PATHS["math"], 4)
    min_len = min(min(len(s) for s in code_seqs + math_seqs), 256)
    tokens = mx.array([s[:min_len] for s in code_seqs] + [s[:min_len] for s in math_seqs])
    x = collect_real_ffn_input(model, tokens, 27)
    mx.eval(x)
    _, _, stats = route_with_stats(x, params, CONFIG)
    assert stats.max_expert_share < COLLAPSE_MAX_SHARE_BOUND
    assert stats.entropy_bits > MIN_ENTROPY_BITS_CLEAN


def test_heavily_imbalanced_batch_avoids_collapse():
    """A batch heavily skewed by real domain composition (15 code
    sequences to 1 math sequence) -- a real, disproportionate token-
    count imbalance, not just a small perturbation."""
    model, _payload = load_frozen_model()
    params = init_moe_layer(CONFIG)
    code_seqs = load_real_sequences(DOMAIN_DATA_PATHS["code"], 15)
    math_seqs = load_real_sequences(DOMAIN_DATA_PATHS["math"], 1)
    min_len = min(min(len(s) for s in code_seqs + math_seqs), 256)
    tokens = mx.array([s[:min_len] for s in code_seqs] + [s[:min_len] for s in math_seqs])
    x = collect_real_ffn_input(model, tokens, 27)
    mx.eval(x)
    _, _, stats = route_with_stats(x, params, CONFIG)
    assert stats.max_expert_share < COLLAPSE_MAX_SHARE_BOUND


def test_within_sequence_domain_shift_avoids_collapse():
    """A domain shift WITHIN a single sequence (not just different rows
    in a batch): the first half of each row is real code, the second
    half is real math -- the literal "domain shift" scenario the plan
    names, not just domain-mixing across independent rows."""
    model, _payload = load_frozen_model()
    params = init_moe_layer(CONFIG)
    code_seqs = load_real_sequences(DOMAIN_DATA_PATHS["code"], 4)
    math_seqs = load_real_sequences(DOMAIN_DATA_PATHS["math"], 4)
    half = 128
    rows = [c[:half] + m[:half] for c, m in zip(code_seqs, math_seqs)]
    tokens = mx.array(rows)
    x = collect_real_ffn_input(model, tokens, 27)
    mx.eval(x)
    _, diag, stats = route_with_stats(x, params, CONFIG)
    mx.eval(diag.expert_idx)
    n = diag.expert_idx.shape[0]
    counts = [int(mx.sum((diag.expert_idx == e).astype(mx.int32))) for e in range(CONFIG.num_experts)]
    assert max(counts) / n < COLLAPSE_MAX_SHARE_BOUND
    assert min(counts) > 0, "no expert should go dead across a within-sequence domain shift"


def test_noise_injection_keeps_utilization_balanced_even_as_entropy_collapses():
    """A real, disclosed, non-obvious finding: as injected Gaussian
    noise grows (0x to 100x the real activation std), per-token routing
    ENTROPY drops sharply toward zero (each token's decision becomes
    increasingly sharp/confident, since softmax sharpens as input
    magnitude grows relative to a fixed router scale) -- but AGGREGATE
    utilization stays balanced, converging toward uniform (~25% each)
    rather than collapsing onto one expert. These are different
    signals: entropy is about per-token decision confidence/
    brittleness, utilization/max_expert_share is the direct collapse
    signal E2's exit gate cares about. Both are checked here, and their
    divergence at high noise is itself the point of this test."""
    model, _payload = load_frozen_model()
    params = init_moe_layer(CONFIG)
    prose_seqs = load_real_sequences(DOMAIN_DATA_PATHS["prose"], 8)
    min_len = min(min(len(s) for s in prose_seqs), 256)
    tokens = mx.array([s[:min_len] for s in prose_seqs])
    x_clean = collect_real_ffn_input(model, tokens, 27)
    mx.eval(x_clean)
    real_std = float(mx.std(x_clean))

    _, _, clean_stats = route_with_stats(x_clean, params, CONFIG)
    assert clean_stats.entropy_bits > MIN_ENTROPY_BITS_CLEAN

    noise = mx.random.normal(x_clean.shape, key=mx.random.key(999)) * real_std * 100.0
    x_extreme_noise = x_clean + noise
    _, _, noisy_stats = route_with_stats(x_extreme_noise, params, CONFIG)

    assert noisy_stats.entropy_bits < clean_stats.entropy_bits * 0.5, (
        "expected entropy to drop substantially under extreme noise (a real, disclosed effect)"
    )
    assert noisy_stats.max_expert_share < COLLAPSE_MAX_SHARE_BOUND, (
        "expected utilization to stay balanced (no collapse) even though per-token entropy drops"
    )


def test_capacity_is_never_exceeded_on_real_domain_content():
    """`reference/hz0e_moe_contract.py`'s own capacity bound, re-checked
    here specifically against real domain activations (not just the
    synthetic toy data E1's own tests used)."""
    model, _payload = load_frozen_model()
    params = init_moe_layer(CONFIG)
    import math as _math
    for domain in DOMAIN_DATA_PATHS:
        x = _real_domain_x(model, domain, 27)
        _, diag, _ = route_with_stats(x, params, CONFIG)
        mx.eval(diag.expert_counts)
        n = x.shape[0] * x.shape[1]
        capacity = _math.ceil(CONFIG.capacity_factor * n / CONFIG.num_experts)
        for count in diag.expert_counts.tolist():
            assert count <= capacity, f"{domain}: expert served {count} tokens, exceeding capacity {capacity}"
